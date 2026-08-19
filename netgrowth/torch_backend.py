"""CPU Torch evaluation backend for the frozen direct-policy protocol."""

from __future__ import annotations

import io
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch._higher_order_ops.scan import scan

from .config import PolicyConfig
from .evidence import CandidateResult, FoldResult, TemporalCandidate, choose_candidate
from .market_data import CanonicalDataset, build_market_state
from .simulation import (
    EquityPoint,
    MarketMinute,
    ReplayResult,
    SimulationConfig,
    SimulationState,
    Trade,
    advance_simulation,
    marked_weights,
    schedule_portfolio_change,
    simulation_config_for_policy,
)
from .training import (
    LinearDirectPolicy,
    TemporalConvPolicy,
    constrain_weights,
    drawdown_constraint_violation,
    net_log_growth_loss,
    sunday_retraining_boundaries,
    training_episode_slices,
    update_drawdown_multiplier,
    walk_forward_folds,
)
from .workflow import EvaluationOutcome, PaperPolicyDecision

ModelFactory = Callable[[int], nn.Module]


@dataclass(frozen=True)
class _ModelContract:
    kind: str
    metadata: dict[str, int]
    receptive_bars: int
    member_count: int
    factory: ModelFactory

    def serialize(self, models: tuple[nn.Module, ...]) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            {
                "kind": self.kind,
                "metadata": self.metadata,
                "state_dicts": [model.state_dict() for model in models],
            },
            buffer,
        )
        return buffer.getvalue()

    def restore(self, payload: dict[str, object], feature_count: int) -> tuple[nn.Module, ...]:
        if payload["kind"] != self.kind or payload["metadata"] != self.metadata:
            raise ValueError("Fitted Policy does not match the validated architecture")
        states = cast(list[dict[str, Any]], payload["state_dicts"])
        models = tuple(self.factory(feature_count) for _ in states)
        for model, state_dict in zip(models, states, strict=True):
            model.load_state_dict(state_dict)
            model.eval()
        return models


def _linear_contract(config: PolicyConfig) -> _ModelContract:
    def factory(feature_count: int) -> nn.Module:
        return LinearDirectPolicy(feature_count, len(config.tickers))

    return _ModelContract("linear", {}, 1, 1, factory)


def _temporal_contract(config: PolicyConfig, width: int, days: int) -> _ModelContract:
    receptive_bars = days * 96

    def factory(feature_count: int) -> nn.Module:
        return TemporalConvPolicy(
            feature_count,
            len(config.tickers),
            width=width,
            receptive_bars=receptive_bars,
        )

    return _ModelContract(
        "tcn-ensemble",
        {"width": width, "days": days},
        receptive_bars,
        len(config.seeds),
        factory,
    )


def _model_contract(selected: dict[str, object], config: PolicyConfig) -> _ModelContract:
    if selected["kind"] == "linear":
        return _linear_contract(config)
    metadata = cast(dict[str, int], selected["metadata"])
    return _temporal_contract(config, int(metadata["width"]), int(metadata["days"]))


@dataclass(frozen=True)
class _PreparedEvaluation:
    state: pd.DataFrame
    latency_returns: pd.DataFrame
    returns: pd.DataFrame
    funding: pd.DataFrame


@dataclass(frozen=True)
class _TrainingRollout:
    weights: torch.Tensor
    turnover: torch.Tensor
    simple_growth: torch.Tensor
    current_portfolios: torch.Tensor


@dataclass(frozen=True)
class _FittingProtocol:
    transaction_cost_rate: float
    minimum_turnover: float
    drawdown_limit: float
    receptive_bars: int
    episode_bars: int
    epochs: int


def _fitting_protocol(config: PolicyConfig, receptive_bars: int) -> _FittingProtocol:
    return _FittingProtocol(
        transaction_cost_rate=config.transaction_cost_rate,
        minimum_turnover=config.minimum_turnover,
        drawdown_limit=config.drawdown_limit,
        receptive_bars=receptive_bars,
        episode_bars=config.training_episode_days * 96,
        epochs=config.training_epochs,
    )


def _training_step(
    current: torch.Tensor,
    inputs: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    (
        market_logits,
        latency_step,
        return_step,
        funding_step,
        current_matrix,
        transaction_cost,
        minimum_turnover,
    ) = inputs
    desired = constrain_weights(market_logits + torch.nn.functional.linear(current, current_matrix))
    latency_factor = 1.0 + (current * latency_step).sum()
    current_at_fill = current * (1.0 + latency_step) / latency_factor.clamp_min(1e-6)
    post_cost_ratio = torch.ones((), device=current.device)
    for _ in range(16):
        desired_turnover = (desired * post_cost_ratio - current_at_fill).abs().sum()
        post_cost_ratio = 1.0 - transaction_cost * desired_turnover
    executes = desired_turnover >= minimum_turnover
    executed = torch.where(executes, desired, current_at_fill)
    realized_turnover = torch.where(executes, desired_turnover, torch.zeros_like(desired_turnover))
    post_cost_ratio = torch.where(executes, post_cost_ratio, torch.ones_like(post_cost_ratio))
    held_return = (executed * return_step).sum()
    funding_cashflow = (executed * funding_step).sum()
    holding_factor = post_cost_ratio * (1.0 + held_return - funding_cashflow)
    total_factor = latency_factor * holding_factor
    simple_growth = total_factor - 1.0
    next_notional = post_cost_ratio * executed * (1.0 + return_step)
    next_current = next_notional / holding_factor.clamp_min(1e-6)
    return next_current.clone(), (
        executed.clone(),
        realized_turnover.clone(),
        simple_growth.clone(),
        current.clone(),
    )


def _scan_training_rollout(
    market_logits: torch.Tensor,
    current_matrix: torch.Tensor,
    latency_returns: torch.Tensor,
    returns: torch.Tensor,
    funding: torch.Tensor,
    transaction_cost_rate: torch.Tensor,
    minimum_turnover: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    steps = len(returns)
    matrices = current_matrix.unsqueeze(0).expand(steps, -1, -1)
    costs = transaction_cost_rate.expand(steps)
    thresholds = minimum_turnover.expand(steps)
    _, outputs = scan(
        _training_step,
        torch.zeros(returns.shape[1], device=returns.device),
        (market_logits, latency_returns, returns, funding, matrices, costs, thresholds),
    )
    return cast(tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], outputs)


_compiled_cpu_training_rollout = torch.compile(_scan_training_rollout, fullgraph=True)


def _training_rollout(
    market_logits: torch.Tensor,
    current_matrix: torch.Tensor,
    latency_returns: torch.Tensor,
    returns: torch.Tensor,
    funding: torch.Tensor,
    *,
    transaction_cost_rate: float,
    minimum_turnover: float,
) -> _TrainingRollout:
    """Roll a differentiable policy over the actual marked portfolio seen at each step."""
    if market_logits.shape != returns.shape or returns.shape != funding.shape or returns.shape != latency_returns.shape:
        raise ValueError("training state, latency, returns, and funding must share one timeline")
    arguments = (
        market_logits,
        current_matrix,
        latency_returns,
        returns,
        funding,
        torch.tensor(transaction_cost_rate, device=returns.device),
        torch.tensor(minimum_turnover, device=returns.device),
    )
    if returns.is_cuda and torch.version.hip is not None:
        cpu_outputs = _compiled_cpu_training_rollout(*(value.to("cpu") for value in arguments))
        weights, turnovers, simple_growth, current_portfolios = (value.to(returns.device) for value in cpu_outputs)
    else:
        weights, turnovers, simple_growth, current_portfolios = _scan_training_rollout(*arguments)
    return _TrainingRollout(
        weights=weights,
        turnover=turnovers,
        simple_growth=simple_growth,
        current_portfolios=current_portfolios,
    )


def _prepare(canonical: CanonicalDataset) -> _PreparedEvaluation:
    state = build_market_state(canonical)
    state_index = cast(pd.DatetimeIndex, state.index)
    closes = _decision_closes(canonical, state_index)
    fill_references = _decision_fill_references(canonical, state_index)
    latency_returns = fill_references.div(closes).sub(1.0).fillna(0.0)
    returns = closes.shift(-1).div(fill_references).sub(1.0).fillna(0.0)
    funding = _training_funding(canonical, fill_references)
    return _PreparedEvaluation(state, latency_returns, returns, funding)


def _training_funding(canonical: CanonicalDataset, fill_references: pd.DataFrame) -> pd.DataFrame:
    """Convert rates to event-mark notional cashflows per unit of interval-start portfolio weight."""
    effective: dict[str, pd.Series] = {}
    for ticker in canonical.tickers:
        instrument = canonical.instruments[ticker]
        rates = (
            instrument.funding.set_axis(cast(pd.DatetimeIndex, instrument.funding.index).floor("min"))
            .groupby(level=0)
            .sum()
        )
        event_marks = instrument.perpetual["close"].reindex(rates.index)
        marked_rates = rates * event_marks
        due_notional = marked_rates.resample("15min", closed="right", label="left").sum()
        effective[ticker] = due_notional.reindex(fill_references.index).fillna(0.0) / fill_references[ticker]
    return pd.DataFrame(effective, index=fill_references.index).fillna(0.0)


def _recent_paper_context(canonical: CanonicalDataset, signal_time: pd.Timestamp) -> CanonicalDataset:
    """Bound inference preparation while retaining normalization plus the largest receptive field."""
    start = signal_time - pd.Timedelta(days=10)
    return CanonicalDataset(
        instruments={
            ticker: type(data)(
                perpetual=data.perpetual.loc[start:],
                spot=data.spot.loc[start:],
                funding=data.funding.loc[start:],
                open_interest=data.open_interest.loc[start:],
                premium=data.premium.loc[start:],
            )
            for ticker, data in canonical.instruments.items()
        },
        tickers=canonical.tickers,
    )


def _decision_closes(canonical: CanonicalDataset, index: pd.DatetimeIndex) -> pd.DataFrame:
    closes = {}
    for ticker in canonical.tickers:
        minute = canonical.instruments[ticker].perpetual["close"]
        closes[ticker] = minute.resample("15min", closed="left", label="right").last().reindex(index)
    return pd.DataFrame(closes, index=index)


def _decision_fill_references(canonical: CanonicalDataset, index: pd.DatetimeIndex) -> pd.DataFrame:
    eligible_at = index + pd.Timedelta(minutes=1)
    references = {
        ticker: canonical.instruments[ticker].perpetual["open"].reindex(eligible_at).set_axis(index)
        for ticker in canonical.tickers
    }
    return pd.DataFrame(references, index=index)


def _fit(
    factory: ModelFactory,
    features: pd.DataFrame,
    latency_returns: pd.DataFrame,
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    seed: int,
    device: str,
    protocol: _FittingProtocol,
) -> nn.Module:
    torch.manual_seed(seed)
    model = factory(features.shape[1]).to(device)
    if min(len(features), protocol.episode_bars) < max(32, protocol.receptive_bars):
        raise ValueError("insufficient revealed history to fit the Policy Protocol")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    multiplier = torch.tensor(0.0, device=device)
    episodes = training_episode_slices(
        revealed_bars=len(features),
        episode_bars=protocol.episode_bars,
        epochs=protocol.epochs,
    )
    for episode in episodes:
        values = torch.tensor(features.iloc[episode].to_numpy(dtype=np.float32), device=device).unsqueeze(0)
        response = torch.tensor(returns.iloc[episode].to_numpy(dtype=np.float32), device=device)
        latency = torch.tensor(latency_returns.iloc[episode].to_numpy(dtype=np.float32), device=device)
        carry = torch.tensor(funding.iloc[episode].to_numpy(dtype=np.float32), device=device)
        optimizer.zero_grad()
        encoded = cast(Any, model).encode_trajectory(values)
        market_logits, current_matrix = cast(Any, model).rollout_components(encoded)
        rollout = _training_rollout(
            market_logits,
            current_matrix,
            latency,
            response,
            carry,
            transaction_cost_rate=protocol.transaction_cost_rate,
            minimum_turnover=protocol.minimum_turnover,
        )
        loss = net_log_growth_loss(
            rollout.weights,
            response,
            carry,
            transaction_cost_rate=protocol.transaction_cost_rate,
            drawdown_limit=protocol.drawdown_limit,
            constraint_multiplier=multiplier,
            turnover=rollout.turnover,
            simple_growth=rollout.simple_growth,
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        with torch.no_grad():
            violation = drawdown_constraint_violation(
                rollout.weights,
                response,
                carry,
                transaction_cost_rate=protocol.transaction_cost_rate,
                drawdown_limit=protocol.drawdown_limit,
                turnover=rollout.turnover,
                simple_growth=rollout.simple_growth,
            )
            multiplier = update_drawdown_multiplier(multiplier, violation)
    return model.cpu().eval()


def _ensemble_target(
    models: tuple[nn.Module, ...],
    encoded_members: tuple[torch.Tensor, ...],
    current_weights: dict[str, float],
    tickers: tuple[str, ...],
) -> dict[str, float]:
    """Average member outputs produced from one shared, actually marked portfolio."""
    if len(models) != len(encoded_members) or not models:
        raise ValueError("ensemble models and encodings must have the same non-zero size")
    current = torch.tensor([[current_weights[ticker] for ticker in tickers]], dtype=torch.float32)
    with torch.no_grad():
        outputs = [
            cast(Any, model).target_from_encoded(encoded.unsqueeze(0), current).squeeze(0)
            for model, encoded in zip(models, encoded_members, strict=True)
        ]
    averaged = torch.stack(outputs).mean(dim=0)
    return dict(zip(tickers, averaged.tolist(), strict=True))


def _market_minutes(
    canonical: CanonicalDataset,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: PolicyConfig,
) -> list[MarketMinute]:
    raw_start = start - pd.Timedelta(minutes=1)
    frames = [canonical.instruments[ticker].perpetual.loc[raw_start:end] for ticker in config.tickers]
    index = frames[0].index
    for frame in frames[1:]:
        index = index.intersection(frame.index)
    event_index = index + pd.Timedelta(minutes=1)
    selected = (event_index >= start) & (event_index < end)
    event_index = event_index[selected]
    closes = np.column_stack([frame.reindex(index)["close"].to_numpy(dtype=float)[selected] for frame in frames])
    opens = np.column_stack([frame["open"].reindex(event_index).to_numpy(dtype=float) for frame in frames])
    funding = np.column_stack(
        [
            canonical.instruments[ticker]
            .funding.set_axis(cast(pd.DatetimeIndex, canonical.instruments[ticker].funding.index).floor("min"))
            .groupby(level=0)
            .sum()
            .reindex(event_index)
            .fillna(0.0)
            .to_numpy(dtype=float)
            for ticker in config.tickers
        ]
    )
    rows = []
    for position, timestamp in enumerate(event_index):
        marks = dict(zip(config.tickers, closes[position], strict=True))
        references = dict(zip(config.tickers, opens[position], strict=True))
        due = {
            ticker: float(rate) for ticker, rate in zip(config.tickers, funding[position], strict=True) if rate != 0.0
        }
        rows.append(
            MarketMinute(
                timestamp=timestamp.to_pydatetime(),
                mark_prices=marks,
                historical_open=references,
                funding_rates=due,
            )
        )
    return rows


@dataclass
class _ReplayTrack:
    state: SimulationState | None = None
    trades: list[Trade] = field(default_factory=list)
    equity: list[EquityPoint] = field(default_factory=list)

    def advance(self, minutes: list[MarketMinute], config: SimulationConfig) -> None:
        if self.state is not None and self.state.flattened:
            return
        result, self.state = advance_simulation(minutes, config, self.state)
        self.trades.extend(result.trades)
        self.equity.extend(result.equity)

    def result(self, initial_equity: float) -> ReplayResult:
        if self.state is None:
            raise ValueError("evaluation period contains no executable market minutes")
        return ReplayResult(
            initial_equity=initial_equity,
            final_equity=self.state.equity,
            max_drawdown=self.state.max_drawdown,
            max_gross_exposure=self.state.max_gross_exposure,
            max_instrument_exposure=self.state.max_instrument_exposure.copy(),
            turnover_notional=self.state.turnover_notional,
            transaction_cost=self.state.transaction_cost,
            funding_cashflow=self.state.funding_cashflow,
            qualifying_portfolio_changes=self.state.qualifying_portfolio_changes,
            risk_stop_triggered=self.state.risk_stop_time is not None,
            risk_stop_time=self.state.risk_stop_time,
            trades=tuple(self.trades),
            equity=tuple(self.equity),
        )


@dataclass(frozen=True)
class _PeriodEvaluation:
    member_replays: tuple[ReplayResult, ...]
    ensemble_replay: ReplayResult
    models: tuple[nn.Module, ...]


def _outcome(replay: ReplayResult, model_bytes: bytes) -> EvaluationOutcome:
    return EvaluationOutcome(
        net_return=replay.compounded_net_return,
        max_drawdown=replay.max_drawdown,
        qualifying_changes=replay.qualifying_portfolio_changes,
        model_bytes=model_bytes,
        equity_rows=tuple(
            {
                "timestamp": point.timestamp.isoformat(),
                "equity": point.equity,
                "drawdown": point.drawdown,
                "gross_exposure": point.gross_exposure,
            }
            for point in replay.equity
        ),
        trade_rows=tuple(
            {
                "timestamp": trade.timestamp.isoformat(),
                "ticker": trade.ticker,
                "quantity": trade.quantity,
                "reference_price": trade.reference_price,
                "effective_fill": trade.effective_fill,
                "target_weight": trade.target_weight,
            }
            for trade in replay.trades
        ),
        diagnostics={
            "transaction_cost": float(replay.transaction_cost),
            "funding_cashflow": float(replay.funding_cashflow),
            "turnover_notional": float(replay.turnover_notional),
            "risk_stop_triggered": replay.risk_stop_triggered,
            "maximum_gross_exposure": float(replay.max_gross_exposure),
            "maximum_instrument_exposure": replay.max_instrument_exposure,
        },
    )


@dataclass
class TorchEvaluationBackend:
    state_root: Path

    def _prepared(self, canonical: CanonicalDataset) -> _PreparedEvaluation:
        cache_directory = self.state_root / ".cache"
        cache_path = cache_directory / f"prepared-{canonical.identity_hash}.pkl"
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)  # noqa: S301
            if isinstance(cached, _PreparedEvaluation):
                return cached
        prepared = _prepare(canonical)
        cache_directory.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(cache_path)
        return prepared

    def _evaluate_period(
        self,
        canonical: CanonicalDataset,
        config: PolicyConfig,
        device: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        factory: ModelFactory,
        *,
        seeds: tuple[int, ...],
        receptive_bars: int,
        initial_training_end: pd.Timestamp | None = None,
        prepared: _PreparedEvaluation | None = None,
    ) -> _PeriodEvaluation:
        """Fit prequential members and replay each candidate against its actual portfolio."""
        if not seeds:
            raise ValueError("evaluation requires at least one fitted-policy seed")
        prepared = prepared or self._prepared(canonical)
        state, latency_returns, returns, funding = (
            prepared.state,
            prepared.latency_returns,
            prepared.returns,
            prepared.funding,
        )
        simulation_config = simulation_config_for_policy(config, mode="historical")
        minutes = _market_minutes(canonical, start, end, config)
        minute_index = pd.DatetimeIndex([minute.timestamp for minute in minutes])
        member_tracks = tuple(_ReplayTrack() for _ in seeds)
        ensemble_track = _ReplayTrack() if len(seeds) > 1 else member_tracks[0]
        fitted: tuple[nn.Module, ...] = ()
        boundaries = sunday_retraining_boundaries(start, end)
        for boundary, next_boundary in zip(boundaries, boundaries[1:], strict=False):
            training_end = initial_training_end if boundary == start and initial_training_end else boundary
            revealed = state.loc[state.index < training_end]
            fitted = tuple(
                _fit(
                    factory,
                    revealed,
                    latency_returns.reindex(revealed.index),
                    returns.reindex(revealed.index),
                    funding.reindex(revealed.index),
                    seed=seed,
                    device=device,
                    protocol=_fitting_protocol(config, receptive_bars),
                )
                for seed in seeds
            )
            decision_slice = state.loc[(state.index >= boundary) & (state.index < next_boundary)]
            context = state.loc[state.index < next_boundary].tail(len(decision_slice) + receptive_bars - 1)
            values = torch.tensor(context.to_numpy(dtype=np.float32), device=device).unsqueeze(0)
            encoded_members = []
            with torch.no_grad():
                for model in fitted:
                    encoded_members.append(cast(Any, model.to(device)).encode_trajectory(values).cpu())
                    model.cpu()
            encoded_by_time = {
                timestamp: tuple(encoded[position] for encoded in encoded_members)
                for position, timestamp in enumerate(context.index)
                if position >= receptive_bars - 1 and boundary <= timestamp < next_boundary
            }

            segment_start = minute_index.searchsorted(boundary, side="left")
            segment_end = minute_index.searchsorted(next_boundary, side="left")
            pending_minutes: list[MarketMinute] = []
            for minute in minutes[segment_start:segment_end]:
                pending_minutes.append(minute)
                timestamp = pd.Timestamp(minute.timestamp)
                encodings = encoded_by_time.get(timestamp)
                if encodings is None:
                    continue
                for track in member_tracks:
                    track.advance(pending_minutes, simulation_config)
                if len(seeds) > 1:
                    ensemble_track.advance(pending_minutes, simulation_config)
                pending_minutes = []
                for member_index, track in enumerate(member_tracks):
                    if track.state is None or track.state.risk_stop_time is not None:
                        continue
                    target = _ensemble_target(
                        (fitted[member_index],),
                        (encodings[member_index],),
                        marked_weights(track.state, minute.mark_prices, config.tickers),
                        config.tickers,
                    )
                    schedule_portfolio_change(track.state, minute.timestamp, target, simulation_config)
                if len(fitted) > 1 and ensemble_track.state is not None and ensemble_track.state.risk_stop_time is None:
                    target = _ensemble_target(
                        fitted,
                        encodings,
                        marked_weights(ensemble_track.state, minute.mark_prices, config.tickers),
                        config.tickers,
                    )
                    schedule_portfolio_change(ensemble_track.state, minute.timestamp, target, simulation_config)
            if pending_minutes:
                for track in member_tracks:
                    track.advance(pending_minutes, simulation_config)
                if len(seeds) > 1:
                    ensemble_track.advance(pending_minutes, simulation_config)
        if not fitted:
            raise ValueError("evaluation period is empty")
        member_replays = tuple(track.result(config.initial_equity) for track in member_tracks)
        return _PeriodEvaluation(
            member_replays=member_replays,
            ensemble_replay=ensemble_track.result(config.initial_equity),
            models=fitted,
        )

    def validate(self, canonical: CanonicalDataset, config: PolicyConfig, device: str) -> EvaluationOutcome:
        prepared = self._prepared(canonical)
        folds = walk_forward_folds(
            common_start=canonical.common_trading_start,
            holdout_start=pd.Timestamp(config.development_evidence_end, tz="UTC"),
            fold_count=config.validation_folds,
            fold_days=config.fold_days,
            purge_days=max(config.receptive_field_days),
        )

        linear_contract = _linear_contract(config)

        linear_replays = []
        last_model: nn.Module | None = None
        for fold in folds:
            period = self._evaluate_period(
                canonical,
                config,
                device,
                fold.validation_start,
                fold.validation_end,
                linear_contract.factory,
                seeds=(config.seeds[0],),
                receptive_bars=linear_contract.receptive_bars,
                initial_training_end=fold.training_end,
                prepared=prepared,
            )
            linear_replays.append(period.member_replays[0])
            last_model = period.models[0]
        linear = CandidateResult(
            "linear",
            tuple(FoldResult(item.compounded_net_return, item.max_drawdown) for item in linear_replays),
        )

        temporal_results = []
        temporal_models: dict[str, tuple[nn.Module, ...]] = {}
        temporal_contracts: dict[str, _ModelContract] = {}
        temporal_ensemble_replays: dict[str, list[ReplayResult]] = {}
        for width in config.temporal_widths:
            for days in config.receptive_field_days:
                contract = _temporal_contract(config, width, days)
                receptive = contract.receptive_bars
                seed_replays: list[list[ReplayResult]] = [[] for _ in config.seeds]
                ensemble_replays: list[ReplayResult] = []
                last_seed_models: tuple[nn.Module, ...] = ()

                for fold in folds:
                    period = self._evaluate_period(
                        canonical,
                        config,
                        device,
                        fold.validation_start,
                        fold.validation_end,
                        contract.factory,
                        seeds=config.seeds,
                        receptive_bars=receptive,
                        initial_training_end=fold.training_end,
                        prepared=prepared,
                    )
                    for seed_index, replay in enumerate(period.member_replays):
                        seed_replays[seed_index].append(replay)
                    ensemble_replays.append(period.ensemble_replay)
                    last_seed_models = period.models
                seeds = [
                    CandidateResult(
                        f"tcn-{width}-{days}-seed-{seed}",
                        tuple(FoldResult(item.compounded_net_return, item.max_drawdown) for item in replays),
                    )
                    for seed, replays in zip(config.seeds, seed_replays, strict=True)
                ]
                architecture = f"tcn-{width}-{days}"
                temporal_models[architecture] = last_seed_models
                temporal_contracts[architecture] = contract
                temporal_ensemble_replays[architecture] = ensemble_replays
                ensemble_folds = tuple(
                    FoldResult(replay.compounded_net_return, replay.max_drawdown) for replay in ensemble_replays
                )
                seed_tuple = cast(tuple[CandidateResult, CandidateResult, CandidateResult], tuple(seeds))
                temporal_results.append(
                    TemporalCandidate(
                        architecture,
                        seed_tuple,
                        CandidateResult(f"tcn-{width}-{days}-ensemble", ensemble_folds),
                    )
                )
        selected = choose_candidate(linear, tuple(temporal_results), drawdown_limit=config.drawdown_limit)
        if selected.name == "linear":
            assert last_model is not None
            model_bytes = linear_contract.serialize((last_model,))
            chosen_replays = linear_replays
        else:
            architecture = selected.name.removesuffix("-ensemble")
            models = temporal_models[architecture]
            model_bytes = temporal_contracts[architecture].serialize(models)
            chosen_replays = temporal_ensemble_replays[architecture]
        aggregate_return = selected.compounded_net_return
        max_drawdown = max(fold.max_drawdown for fold in selected.folds)
        fold_outcomes = [_outcome(replay, b"") for replay in chosen_replays]
        equity_rows = tuple(
            {"fold": fold_index + 1, **row}
            for fold_index, outcome in enumerate(fold_outcomes)
            for row in outcome.equity_rows
        )
        trade_rows = tuple(
            {"fold": fold_index + 1, **row}
            for fold_index, outcome in enumerate(fold_outcomes)
            for row in outcome.trade_rows
        )
        fold_metrics = [
            {
                "fold": index + 1,
                "validation_start": fold.validation_start.isoformat(),
                "validation_end": fold.validation_end.isoformat(),
                "compounded_net_return": float(replay.compounded_net_return),
                "maximum_drawdown": float(replay.max_drawdown),
                "qualifying_portfolio_changes": replay.qualifying_portfolio_changes,
                "turnover_notional": float(replay.turnover_notional),
                "transaction_cost": float(replay.transaction_cost),
                "funding_cashflow": float(replay.funding_cashflow),
                "maximum_gross_exposure": float(replay.max_gross_exposure),
                "maximum_instrument_exposure": replay.max_instrument_exposure,
                "risk_stop_triggered": replay.risk_stop_triggered,
            }
            for index, (fold, replay) in enumerate(zip(folds, chosen_replays, strict=True))
        ]
        return EvaluationOutcome(
            aggregate_return,
            max_drawdown,
            sum(replay.qualifying_portfolio_changes for replay in chosen_replays),
            model_bytes,
            equity_rows,
            trade_rows,
            {
                "selected_candidate": selected.name,
                "folds": fold_metrics,
                "turnover_notional": float(sum(replay.turnover_notional for replay in chosen_replays)),
                "transaction_cost": float(sum(replay.transaction_cost for replay in chosen_replays)),
                "funding_cashflow": float(sum(replay.funding_cashflow for replay in chosen_replays)),
                "maximum_gross_exposure": float(max(replay.max_gross_exposure for replay in chosen_replays)),
                "maximum_instrument_exposure": {
                    ticker: float(max(replay.max_instrument_exposure[ticker] for replay in chosen_replays))
                    for ticker in config.tickers
                },
            },
        )

    @staticmethod
    def _selected_metadata(validated_model: bytes) -> dict[str, object]:
        return cast(
            dict[str, object],
            torch.load(io.BytesIO(validated_model), map_location="cpu", weights_only=True),
        )

    def _post_validation(
        self,
        canonical: CanonicalDataset,
        config: PolicyConfig,
        device: str,
        *,
        validated_model: bytes,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> EvaluationOutcome:
        selected = self._selected_metadata(validated_model)
        prepared = self._prepared(canonical)
        contract = _model_contract(selected, config)
        seeds = config.seeds[: contract.member_count]
        period = self._evaluate_period(
            canonical,
            config,
            device,
            start,
            end,
            contract.factory,
            seeds=seeds,
            receptive_bars=contract.receptive_bars,
            prepared=prepared,
        )
        return _outcome(
            period.ensemble_replay,
            contract.serialize(period.models),
        )

    def holdout(
        self,
        canonical: CanonicalDataset,
        config: PolicyConfig,
        device: str,
        validated_model: bytes,
    ) -> EvaluationOutcome:
        start = pd.Timestamp(config.holdout_start, tz="UTC")
        end = pd.Timestamp(config.holdout_end, tz="UTC") + pd.Timedelta(days=1)
        return self._post_validation(
            canonical,
            config,
            device,
            validated_model=validated_model,
            start=start,
            end=end,
        )

    def paper(
        self,
        canonical: CanonicalDataset,
        config: PolicyConfig,
        device: str,
        *,
        validated_model: bytes,
        fitted_model: bytes | None,
        observed_at: datetime,
        current_weights: dict[str, float],
    ) -> PaperPolicyDecision:
        """Fit or restore the frozen architecture and emit one causal paper target."""
        selected = self._selected_metadata(validated_model)
        contract = _model_contract(selected, config)
        signal_time = pd.Timestamp(observed_at)
        prepared = (
            self._prepared(canonical)
            if fitted_model is None
            else _prepare(_recent_paper_context(canonical, signal_time))
        )
        revealed = prepared.state.loc[prepared.state.index < signal_time]
        if fitted_model is None:
            models = tuple(
                _fit(
                    contract.factory,
                    revealed,
                    prepared.latency_returns.reindex(revealed.index),
                    prepared.returns.reindex(revealed.index),
                    prepared.funding.reindex(revealed.index),
                    seed=config.seeds[index],
                    device=device,
                    protocol=_fitting_protocol(config, contract.receptive_bars),
                )
                for index in range(contract.member_count)
            )
            model_bytes = contract.serialize(models)
            refitted = True
        else:
            fitted = self._selected_metadata(fitted_model)
            models = contract.restore(fitted, prepared.state.shape[1])
            model_bytes = fitted_model
            refitted = False

        context = prepared.state.loc[prepared.state.index <= signal_time].tail(contract.receptive_bars)
        if len(context) < contract.receptive_bars:
            raise ValueError("insufficient revealed Market State for paper decision")
        values = torch.tensor(context.to_numpy(dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            encodings = tuple(cast(Any, model).encode_trajectory(values)[-1] for model in models)
        target = _ensemble_target(models, encodings, current_weights, config.tickers)
        return PaperPolicyDecision(target_weights=target, model_bytes=model_bytes, refitted=refitted)
