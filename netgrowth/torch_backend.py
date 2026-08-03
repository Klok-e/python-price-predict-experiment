"""CPU Torch evaluation backend for the frozen direct-policy protocol."""

from __future__ import annotations

import io
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from .config import PolicyConfig
from .evidence import CandidateResult, FoldResult, TemporalCandidate, choose_candidate
from .market_data import CanonicalDataset, build_market_state
from .simulation import MarketMinute, ReplayResult, SimulationConfig, simulate
from .training import (
    LinearDirectPolicy,
    TemporalConvPolicy,
    net_log_growth_loss,
    walk_forward_folds,
)
from .workflow import EvaluationOutcome

ModelFactory = Callable[[int], nn.Module]


def _decision_closes(canonical: CanonicalDataset, index: pd.DatetimeIndex) -> pd.DataFrame:
    closes = {}
    for ticker in canonical.tickers:
        minute = canonical.instruments[ticker].perpetual["close"]
        closes[ticker] = minute.resample("15min", closed="left", label="right").last().reindex(index)
    return pd.DataFrame(closes, index=index)


def _fit(
    factory: ModelFactory,
    features: pd.DataFrame,
    returns: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    seed: int,
    device: str,
    cost: float,
    drawdown_limit: float,
    receptive_bars: int,
) -> nn.Module:
    torch.manual_seed(seed)
    model = factory(features.shape[1]).to(device)
    usable = min(len(features) - receptive_bars, 256)
    if usable < 32:
        raise ValueError("insufficient revealed history to fit the Policy Protocol")
    values = torch.tensor(features.to_numpy(dtype=np.float32), device=device)
    windows = values.unfold(0, receptive_bars, 1).transpose(1, 2)[-usable:]
    response = torch.tensor(returns.to_numpy(dtype=np.float32), device=device)[-usable:]
    carry = torch.tensor(funding.to_numpy(dtype=np.float32), device=device)[-usable:]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    multiplier = torch.tensor(0.0, device=device)
    for _ in range(4):
        optimizer.zero_grad()
        current = torch.zeros((1, returns.shape[1]), device=device)
        trajectory = []
        for window in windows:
            current = model(window.unsqueeze(0), current)
            trajectory.append(current.squeeze(0))
        weights = torch.stack(trajectory)
        loss = net_log_growth_loss(
            weights,
            response,
            carry,
            transaction_cost_rate=cost,
            drawdown_limit=drawdown_limit,
            constraint_multiplier=multiplier,
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        with torch.no_grad():
            multiplier.add_(torch.relu(-loss.detach()) * 0.01).clamp_(min=0.0)
    return model.cpu().eval()


def _targets(
    model: nn.Module,
    features: pd.DataFrame,
    receptive_bars: int,
    tickers: tuple[str, ...],
    initial_weights: np.ndarray | None = None,
) -> dict[pd.Timestamp, dict[str, float]]:
    values = torch.tensor(features.to_numpy(dtype=np.float32))
    starting_weights = (
        np.zeros((1, len(tickers)), dtype=np.float32) if initial_weights is None else initial_weights.reshape(1, -1)
    )
    current = torch.tensor(starting_weights, dtype=torch.float32)
    targets: dict[pd.Timestamp, dict[str, float]] = {}
    with torch.no_grad():
        for position in range(receptive_bars - 1, len(features)):
            window = values[position - receptive_bars + 1 : position + 1].unsqueeze(0)
            output = model(window, current).squeeze(0)
            targets[features.index[position]] = dict(zip(tickers, output.tolist(), strict=True))
            current = output.unsqueeze(0)
    return targets


def _average_targets(
    members: list[dict[pd.Timestamp, dict[str, float]]], tickers: tuple[str, ...]
) -> dict[pd.Timestamp, dict[str, float]]:
    timestamps = set.intersection(*(set(member) for member in members))
    return {
        timestamp: {ticker: float(np.mean([member[timestamp][ticker] for member in members])) for ticker in tickers}
        for timestamp in timestamps
    }


def _replay(
    canonical: CanonicalDataset,
    targets: dict[pd.Timestamp, dict[str, float]],
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: PolicyConfig,
    *,
    mode: str = "historical",
) -> ReplayResult:
    frames = [canonical.instruments[ticker].perpetual.loc[start:end] for ticker in config.tickers]
    index = frames[0].index
    for frame in frames[1:]:
        index = index.intersection(frame.index)
    funding = {ticker: canonical.instruments[ticker].funding for ticker in config.tickers}
    rows = []
    for timestamp in index:
        marks = {
            ticker: float(cast(float, canonical.instruments[ticker].perpetual.at[timestamp, "close"]))
            for ticker in config.tickers
        }
        opens = {
            ticker: float(cast(float, canonical.instruments[ticker].perpetual.at[timestamp, "open"]))
            for ticker in config.tickers
        }
        due = {
            ticker: float(cast(float, series.at[timestamp]))
            for ticker, series in funding.items()
            if timestamp in series.index
        }
        rows.append(
            MarketMinute(
                timestamp=timestamp.to_pydatetime(),
                mark_prices=marks,
                historical_open=opens,
                bid={ticker: price for ticker, price in opens.items()} if mode == "paper" else None,
                ask={ticker: price for ticker, price in opens.items()} if mode == "paper" else None,
                funding_rates=due,
                target_weights=targets.get(timestamp),
            )
        )
    return simulate(
        rows,
        SimulationConfig(
            tickers=config.tickers,
            initial_equity=config.initial_equity,
            decision_latency_seconds=config.decision_latency_seconds,
            transaction_cost_rate=config.transaction_cost_rate,
            minimum_turnover=config.minimum_turnover,
            drawdown_limit=config.drawdown_limit,
            mode="paper" if mode == "paper" else "historical",
        ),
    )


def _outcome(replay: ReplayResult, model_bytes: bytes) -> EvaluationOutcome:
    return EvaluationOutcome(
        net_return=replay.compounded_net_return,
        max_drawdown=replay.max_drawdown,
        qualifying_changes=replay.qualifying_portfolio_changes,
        model_bytes=model_bytes,
        equity_rows=tuple(
            {"timestamp": point.timestamp.isoformat(), "equity": point.equity, "drawdown": point.drawdown}
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
    )


@dataclass
class TorchEvaluationBackend:
    state_root: Path

    def _evaluate_period(
        self,
        canonical: CanonicalDataset,
        config: PolicyConfig,
        device: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        factory: ModelFactory,
        *,
        seed: int,
        receptive_bars: int,
        mode: str = "historical",
        initial_training_end: pd.Timestamp | None = None,
    ) -> tuple[ReplayResult, nn.Module, dict[pd.Timestamp, dict[str, float]]]:
        state = build_market_state(canonical)
        state_index = cast(pd.DatetimeIndex, state.index)
        closes = _decision_closes(canonical, state_index)
        returns = closes.pct_change().shift(-1).fillna(0.0)
        funding = pd.DataFrame(
            {
                ticker: canonical.instruments[ticker].funding.reindex(state.index).fillna(0.0)
                for ticker in config.tickers
            },
            index=state.index,
        )
        all_targets: dict[pd.Timestamp, dict[str, float]] = {}
        fitted: nn.Module | None = None
        handoff_weights = np.zeros(len(config.tickers), dtype=np.float32)
        boundary = start
        while boundary < end:
            next_boundary = min(boundary + pd.Timedelta(days=7), end)
            training_end = initial_training_end if boundary == start and initial_training_end else boundary
            revealed = state.loc[state.index < training_end]
            fitted = _fit(
                factory,
                revealed,
                returns.reindex(revealed.index),
                funding.reindex(revealed.index),
                seed=seed,
                device=device,
                cost=config.transaction_cost_rate,
                drawdown_limit=config.drawdown_limit,
                receptive_bars=receptive_bars,
            )
            context = state.loc[:next_boundary].tail(len(state.loc[boundary:next_boundary]) + receptive_bars - 1)
            weekly_targets = {
                timestamp: weights
                for timestamp, weights in _targets(
                    fitted,
                    context,
                    receptive_bars,
                    config.tickers,
                    initial_weights=handoff_weights,
                ).items()
                if boundary <= timestamp < next_boundary
            }
            all_targets.update(weekly_targets)
            if weekly_targets:
                handoff_weights = np.asarray(list(weekly_targets[max(weekly_targets)].values()), dtype=np.float32)
            boundary = next_boundary
        if fitted is None:
            raise ValueError("evaluation period is empty")
        replay = _replay(canonical, all_targets, start, end, config, mode=mode)
        return replay, fitted, all_targets

    @staticmethod
    def _serialize(kind: str, models: tuple[nn.Module, ...], **metadata: int) -> bytes:
        buffer = io.BytesIO()
        torch.save(
            {
                "kind": kind,
                "metadata": metadata,
                "state_dicts": [model.state_dict() for model in models],
            },
            buffer,
        )
        return buffer.getvalue()

    def validate(self, canonical: CanonicalDataset, config: PolicyConfig, device: str) -> EvaluationOutcome:
        folds = walk_forward_folds(
            common_start=canonical.common_trading_start,
            holdout_start=pd.Timestamp(config.holdout_start, tz="UTC"),
            fold_count=config.validation_folds,
            fold_days=config.fold_days,
            purge_days=max(config.receptive_field_days),
        )

        def linear_factory(count: int) -> nn.Module:
            return LinearDirectPolicy(count, len(config.tickers))

        linear_replays = []
        last_model: nn.Module | None = None
        for fold in folds:
            replay, last_model, _ = self._evaluate_period(
                canonical,
                config,
                device,
                fold.validation_start,
                fold.validation_end,
                linear_factory,
                seed=config.seeds[0],
                receptive_bars=1,
                initial_training_end=fold.training_end,
            )
            linear_replays.append(replay)
        linear = CandidateResult(
            "linear",
            tuple(FoldResult(item.compounded_net_return, item.max_drawdown) for item in linear_replays),
        )

        temporal_results = []
        temporal_models: dict[str, tuple[nn.Module, ...]] = {}
        temporal_ensemble_replays: dict[str, list[ReplayResult]] = {}
        for width in config.temporal_widths:
            for days in config.receptive_field_days:
                receptive = days * 96
                seeds = []
                seed_replays: list[list[ReplayResult]] = []
                seed_targets: list[list[dict[pd.Timestamp, dict[str, float]]]] = []
                last_seed_models: list[nn.Module] = []
                for seed in config.seeds:
                    replays = []
                    targets_by_fold = []
                    for fold in folds:

                        def factory(
                            count: int,
                            selected_width: int = width,
                            selected_receptive: int = receptive,
                        ) -> nn.Module:
                            return TemporalConvPolicy(
                                count,
                                len(config.tickers),
                                width=selected_width,
                                receptive_bars=selected_receptive,
                            )

                        replay, model, targets = self._evaluate_period(
                            canonical,
                            config,
                            device,
                            fold.validation_start,
                            fold.validation_end,
                            factory,
                            seed=seed,
                            receptive_bars=receptive,
                            initial_training_end=fold.training_end,
                        )
                        replays.append(replay)
                        targets_by_fold.append(targets)
                    last_seed_models.append(model)
                    seed_replays.append(replays)
                    seed_targets.append(targets_by_fold)
                    seeds.append(
                        CandidateResult(
                            f"tcn-{width}-{days}-seed-{seed}",
                            tuple(FoldResult(item.compounded_net_return, item.max_drawdown) for item in replays),
                        )
                    )
                architecture = f"tcn-{width}-{days}"
                temporal_models[architecture] = tuple(last_seed_models)
                ensemble_replays = [
                    _replay(
                        canonical,
                        _average_targets(
                            [seed_targets[seed_index][fold_index] for seed_index in range(3)],
                            config.tickers,
                        ),
                        fold.validation_start,
                        fold.validation_end,
                        config,
                    )
                    for fold_index, fold in enumerate(folds)
                ]
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
            model_bytes = self._serialize("linear", (last_model,))
            chosen_replays = linear_replays
        else:
            architecture = selected.name.removesuffix("-ensemble")
            models = temporal_models[architecture]
            width, days = (int(value) for value in architecture.split("-")[1:])
            model_bytes = self._serialize("tcn-ensemble", models, width=width, days=days)
            chosen_replays = temporal_ensemble_replays[architecture]
        aggregate_return = selected.compounded_net_return
        max_drawdown = max(fold.max_drawdown for fold in selected.folds)
        return EvaluationOutcome(
            aggregate_return,
            max_drawdown,
            sum(replay.qualifying_portfolio_changes for replay in chosen_replays),
            model_bytes,
            (),
            (),
        )

    def _selected_metadata(self) -> dict[str, object]:
        models = sorted(self.state_root.glob("validation/*/model.pt"))
        if not models:
            raise ValueError("validation must freeze a Policy Protocol before evaluation")
        return cast(
            dict[str, object],
            torch.load(models[-1], map_location="cpu", weights_only=True),
        )

    def _post_validation(
        self,
        canonical: CanonicalDataset,
        config: PolicyConfig,
        device: str,
        *,
        start: pd.Timestamp,
        end: pd.Timestamp,
        mode: str,
    ) -> EvaluationOutcome:
        selected = self._selected_metadata()
        metadata = cast(dict[str, int], selected["metadata"])
        models: tuple[nn.Module, ...]
        if selected["kind"] == "linear":

            def factory(count: int) -> nn.Module:
                return LinearDirectPolicy(count, len(config.tickers))

            receptive = 1
        else:
            width, days = int(metadata["width"]), int(metadata["days"])
            receptive = days * 96

            def factory(count: int) -> nn.Module:
                return TemporalConvPolicy(count, len(config.tickers), width=width, receptive_bars=receptive)

        if selected["kind"] == "linear":
            replay, fitted, _ = self._evaluate_period(
                canonical,
                config,
                device,
                start,
                end,
                factory,
                seed=config.seeds[0],
                receptive_bars=receptive,
                mode=mode,
            )
            models = (fitted,)
        else:
            member_targets = []
            fitted_members = []
            for seed in config.seeds:
                _, fitted, targets = self._evaluate_period(
                    canonical,
                    config,
                    device,
                    start,
                    end,
                    factory,
                    seed=seed,
                    receptive_bars=receptive,
                    mode=mode,
                )
                member_targets.append(targets)
                fitted_members.append(fitted)
            replay = _replay(
                canonical,
                _average_targets(member_targets, config.tickers),
                start,
                end,
                config,
                mode=mode,
            )
            models = tuple(fitted_members)
        return _outcome(
            replay,
            self._serialize(str(selected["kind"]), models, **metadata),
        )

    def holdout(self, canonical: CanonicalDataset, config: PolicyConfig, device: str) -> EvaluationOutcome:
        start = pd.Timestamp(config.holdout_start, tz="UTC")
        end = pd.Timestamp(config.holdout_end, tz="UTC") + pd.Timedelta(days=1)
        return self._post_validation(canonical, config, device, start=start, end=end, mode="historical")

    def paper(self, canonical: CanonicalDataset, config: PolicyConfig, device: str) -> EvaluationOutcome:
        del canonical, config, device
        raise RuntimeError(
            "Forward Paper Proof requires a persistent contemporaneous midpoint session; "
            "historical archive replay is Development Evidence and cannot advance the Proof Clock"
        )
