"""Production wiring for the localhost-only Paper Account service."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Any, cast

import pandas as pd
import torch
import uvicorn
from torch import nn

from netgrowth.attribution import feature_metadata_from_names, integrated_gradients
from netgrowth.binance import PaperFeedObservation, PublicPaperAdapter
from netgrowth.config import PolicyConfig, load_config
from netgrowth.market_data import CanonicalDataset
from netgrowth.simulation import simulation_config_for_policy
from netgrowth.torch_backend import (
    TorchEvaluationBackend,
    _model_contract,
    _prepare,
    _recent_paper_context,
)

from .application import FittedPolicyCandidate, create_application
from .domain import MarketObservation, PolicyDecision


@dataclass
class ProductionMarketFeed:
    adapter: PublicPaperAdapter
    _recovered: list[MarketObservation] = field(default_factory=list, init=False)

    def observe(self, after: datetime | None) -> MarketObservation:
        observed = self.adapter.mark(after)
        observations = self._observations(observed)
        if not observations:
            raise RuntimeError("public Paper Account feed returned no closed one-minute mark")
        self._recovered = observations[:-1] if after is not None else []
        return observations[-1]

    def recover(self, after: datetime, before: datetime) -> list[MarketObservation]:
        recovered = [observation for observation in self._recovered if after < observation.timestamp < before]
        self._recovered = []
        return recovered

    def _observations(self, observed: PaperFeedObservation) -> list[MarketObservation]:
        indexes = [set(instrument.perpetual.index) for instrument in observed.instruments.values()]
        common = sorted(set.intersection(*indexes)) if indexes else []
        result: list[MarketObservation] = []
        for index, open_time in enumerate(common):
            signal_time = open_time + pd.Timedelta(minutes=1)
            timestamp = signal_time.to_pydatetime()
            marks = {
                ticker: float(observed.instruments[ticker].perpetual.loc[open_time, "close"])
                for ticker in observed.tickers
            }
            candles = {
                ticker: {
                    field: float(observed.instruments[ticker].perpetual.loc[open_time, field])
                    for field in ("open", "high", "low", "close")
                }
                for ticker in observed.tickers
            }
            latest = index == len(common) - 1
            bid = {ticker: observed.quotes[ticker].bid for ticker in observed.tickers} if latest else marks
            ask = {ticker: observed.quotes[ticker].ask for ticker in observed.tickers} if latest else marks
            quote_times = (
                {ticker: observed.quotes[ticker].exchange_time.to_pydatetime() for ticker in observed.tickers}
                if latest
                else dict.fromkeys(observed.tickers, timestamp)
            )
            funding_rates: dict[str, float] = {}
            funding_marks: dict[str, float] = {}
            for ticker in observed.tickers:
                for funding in observed.instruments[ticker].settled_funding:
                    if funding.event_time.floor("min").to_pydatetime() == timestamp:
                        funding_rates[ticker] = funding.rate
                        funding_marks[ticker] = funding.mark_price
            result.append(
                MarketObservation(
                    timestamp=timestamp,
                    mark_prices=marks,
                    bid=bid,
                    ask=ask,
                    quote_exchange_times=quote_times,
                    quote_observed_at=(observed.observed_at.to_pydatetime() if latest else timestamp),
                    funding_rates=funding_rates,
                    funding_mark_prices=funding_marks,
                    input_id=sha256(f"{timestamp.isoformat()}:{','.join(sorted(marks))}".encode()).hexdigest(),
                    decision_bar_closed=latest and timestamp.minute % 15 == 0,
                    reconstructed=not latest,
                    candles=candles,
                )
            )
        return result


@dataclass
class ProductionPolicyBackend:
    adapter: PublicPaperAdapter
    backend: TorchEvaluationBackend
    fitting_adapter: PublicPaperAdapter
    attribution_adapter: PublicPaperAdapter
    fitting_backend: TorchEvaluationBackend
    checkpoint_directory: Path
    config: PolicyConfig
    device: str
    fitted_model: bytes
    _attribution_inputs: dict[
        str,
        tuple[CanonicalDataset, dict[str, float], bytes, datetime],
    ] = field(default_factory=dict, init=False, repr=False)
    _inference_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def decide(self, observation: MarketObservation, current_weights: dict[str, float]) -> PolicyDecision:
        with self._inference_lock:
            canonical = self.adapter.load()
            input_id = sha256(
                (
                    canonical.identity_hash
                    + ":"
                    + observation.timestamp.isoformat()
                    + ":"
                    + json.dumps(current_weights, sort_keys=True, separators=(",", ":"))
                ).encode()
            ).hexdigest()
            result = self.backend.paper(
                canonical,
                self.config,
                self.device,
                validated_model=self.fitted_model,
                fitted_model=self.fitted_model,
                observed_at=observation.timestamp,
                current_weights=current_weights,
            )
        self._attribution_inputs[input_id] = (
            canonical,
            current_weights.copy(),
            self.fitted_model,
            observation.timestamp,
        )
        if result.refitted:
            self.fitted_model = result.model_bytes
        model_id = sha256(self.fitted_model).hexdigest()
        return PolicyDecision(
            raw_target_weights=result.target_weights,
            target_weights=result.target_weights,
            model_id=model_id,
            input_id=input_id,
            protocol_id=self.config.identity_hash,
        )

    def prepare(self, *, observed_at: datetime | None = None) -> None:
        """Warm canonical data, model restoration, and device kernels before a Decision Bar."""
        with self._inference_lock:
            canonical = self.adapter.load()
            signal_time = observed_at
            if signal_time is None:
                last_open = min(instrument.perpetual.index.max() for instrument in canonical.instruments.values())
                signal_time = (last_open + pd.Timedelta(minutes=1)).to_pydatetime()
            result = self.backend.paper(
                canonical,
                self.config,
                self.device,
                validated_model=self.fitted_model,
                fitted_model=self.fitted_model,
                observed_at=signal_time,
                current_weights=dict.fromkeys(self.config.tickers, 0.0),
            )
            if result.refitted:
                raise RuntimeError("policy preparation must not fit or replace the active Fitted Policy")

    def is_due(self, now: datetime, fitting: Mapping[str, Any]) -> bool:
        weekdays = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
        try:
            target_weekday = weekdays.index(self.config.retrain_weekday_utc.lower())
        except ValueError as error:
            raise ValueError(f"unknown UTC fitting weekday {self.config.retrain_weekday_utc!r}") from error
        utc_now = now.astimezone(UTC)
        deadline = (utc_now - timedelta(days=(utc_now.weekday() - target_weekday) % 7)).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        completed_at = fitting.get("completed_at")
        if not isinstance(completed_at, str):
            return True
        return datetime.fromisoformat(completed_at).astimezone(UTC) < deadline

    def fit(
        self,
        observed_at: datetime,
        current_weights: dict[str, float],
    ) -> FittedPolicyCandidate:
        canonical = self.fitting_adapter.load()
        result = self.fitting_backend.paper(
            canonical,
            self.config,
            self.device,
            validated_model=self.fitted_model,
            fitted_model=None,
            observed_at=observed_at,
            current_weights=current_weights,
        )
        if not result.refitted:
            raise RuntimeError("scheduled fitting did not produce a new Fitted Policy")
        model_id = sha256(result.model_bytes).hexdigest()
        self.checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint = self.checkpoint_directory / f"{model_id}.pt"
        if checkpoint.exists():
            if checkpoint.read_bytes() != result.model_bytes:
                raise RuntimeError("immutable Fitted Policy checkpoint hash collision")
        else:
            temporary = checkpoint.with_suffix(".tmp")
            temporary.write_bytes(result.model_bytes)
            temporary.replace(checkpoint)
        return FittedPolicyCandidate(
            model_id=model_id,
            checkpoint=str(checkpoint.resolve()),
            payload=result.model_bytes,
        )

    def activate(self, candidate: FittedPolicyCandidate) -> Callable[[], None]:
        if not isinstance(candidate.payload, bytes):
            raise TypeError("production Policy Handoff requires serialized Fitted Policy bytes")
        if sha256(candidate.payload).hexdigest() != candidate.model_id:
            raise ValueError("Fitted Policy candidate identity does not match its checkpoint")
        with self._inference_lock:
            previous = self.fitted_model
            self.fitted_model = candidate.payload

        def rollback() -> None:
            with self._inference_lock:
                self.fitted_model = previous

        return rollback

    def attribute(self, input_id: str, decision: Mapping[str, Any]) -> dict[str, Any]:
        retained = self._attribution_inputs.pop(input_id, None)
        if retained is None:
            retained = self._recover_attribution_input(decision)
        canonical, current_weights, model_bytes, observed_at = retained
        selected = self.backend._selected_metadata(model_bytes)
        contract = _model_contract(selected, self.config)
        prepared = _prepare(_recent_paper_context(canonical, pd.Timestamp(observed_at)))
        revealed = prepared.state.loc[prepared.state.index <= pd.Timestamp(observed_at)]
        context = revealed.tail(contract.receptive_bars)
        if len(context) < contract.receptive_bars:
            raise ValueError("insufficient revealed Market State for Model Attribution")
        models = contract.restore(selected, context.shape[1])
        ensemble = _AttributionEnsemble(models)
        market_input = torch.tensor(
            context.to_numpy(dtype="float32"),
            dtype=torch.float32,
        ).unsqueeze(0)
        current = torch.tensor(
            [[current_weights[ticker] for ticker in self.config.tickers]],
            dtype=torch.float32,
        )
        result = integrated_gradients(
            ensemble,
            market_input,
            current,
            feature_metadata=feature_metadata_from_names(tuple(context.columns), self.config.tickers),
            target_names=self.config.tickers,
            portfolio_names=self.config.tickers,
        )
        top_influences = [
            {
                "label": f"{target.target}: {influence.label}",
                "value": influence.value,
            }
            for target in result.targets
            for influence in target.top_influences
        ]
        return {
            "method": result.method,
            "baseline": result.baseline,
            "parameters": {"steps": result.steps, "top_k": result.top_k},
            "input_hash": result.input_hash,
            "model_hash": sha256(model_bytes).hexdigest(),
            "top_influences": top_influences,
            "targets": [asdict(target) for target in result.targets],
        }

    def _recover_attribution_input(
        self,
        decision: Mapping[str, Any],
    ) -> tuple[CanonicalDataset, dict[str, float], bytes, datetime]:
        signal_time = decision.get("signal_time")
        if isinstance(signal_time, str):
            observed_at = datetime.fromisoformat(signal_time)
        elif isinstance(signal_time, datetime):
            observed_at = signal_time
        else:
            raise ValueError("durable Decision Record lacks Signal Time")
        current = decision.get("current_portfolio")
        if not isinstance(current, Mapping):
            raise ValueError("durable Decision Record lacks Current Portfolio")
        model_id = decision.get("model_id")
        if not isinstance(model_id, str):
            raise ValueError("durable Decision Record lacks Fitted Policy identity")
        return (
            self.attribution_adapter.load(),
            {ticker: float(current[ticker]) for ticker in self.config.tickers},
            self._model_bytes(model_id),
            observed_at,
        )

    def _model_bytes(self, model_id: str) -> bytes:
        if sha256(self.fitted_model).hexdigest() == model_id:
            return self.fitted_model
        for checkpoint in (
            self.checkpoint_directory / f"{model_id}.pt",
            self.checkpoint_directory / "current.pt",
        ):
            if checkpoint.is_file():
                payload = checkpoint.read_bytes()
                if sha256(payload).hexdigest() == model_id:
                    return payload
        raise RuntimeError(f"immutable Fitted Policy checkpoint {model_id} is unavailable")


class _AttributionEnsemble(nn.Module):
    def __init__(self, models: tuple[nn.Module, ...]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, market_input: torch.Tensor, current_portfolio: torch.Tensor) -> torch.Tensor:
        outputs = []
        for model in self.models:
            encoded = cast(Any, model).encode_trajectory(market_input)[-1].unsqueeze(0)
            outputs.append(cast(Any, model).target_from_encoded(encoded, current_portfolio))
        return torch.stack(outputs).mean(dim=0)


@dataclass(frozen=True)
class DesktopNotifications:
    """Best-effort local desktop delivery; failures remain an operational overlay."""

    def notify(self, kind: str, payload: dict[str, object]) -> None:
        title = {
            "executed_portfolio_change": "Paper Account portfolio change",
            "risk_stop": "Paper Account Risk Stop",
            "manual_reset_completed": "Paper Account reset completed",
            "data_stale": "Paper Account data is stale",
            "fitting_failure": "Paper Account fitting failed",
            "incompatible_policy_revision": "Paper Account migration required",
        }.get(kind, "Paper Account")
        subprocess.run(
            ["notify-send", "--app-name=Net Growth", title, json.dumps(payload, sort_keys=True)],
            check=True,
            timeout=10,
        )


def run_dashboard(
    config_path: str | Path,
    data_directory: str | Path,
    operational_directory: str | Path,
    host: str,
    port: int,
    device: str,
) -> None:
    """Run the one uvicorn process; never launch a browser or bind beyond the requested host."""
    if host != "127.0.0.1":
        raise ValueError("the Paper Account dashboard is localhost-only")
    config = load_config(config_path)
    operational = Path(operational_directory)
    operational.mkdir(parents=True, exist_ok=True)
    adapter = PublicPaperAdapter(Path(data_directory), config.tickers)
    fitting_adapter = PublicPaperAdapter(Path(data_directory), config.tickers)
    attribution_adapter = PublicPaperAdapter(Path(data_directory), config.tickers)
    fitted_model = _selected_fitted_policy(operational)
    policy = ProductionPolicyBackend(
        adapter=adapter,
        backend=TorchEvaluationBackend(operational / "checkpoints"),
        fitting_adapter=fitting_adapter,
        attribution_adapter=attribution_adapter,
        fitting_backend=TorchEvaluationBackend(operational / "fitting-cache"),
        checkpoint_directory=operational / "checkpoints",
        config=config,
        device=device,
        fitted_model=fitted_model,
    )
    app = create_application(
        database_path=operational / "paper-account.sqlite3",
        backup_directory=operational / "backups",
        tickers=config.tickers,
        market_feed=ProductionMarketFeed(adapter),
        policy_backend=policy,
        policy_fitter=policy,
        attribution_backend=policy,
        notifications=DesktopNotifications(),
        simulation_config=simulation_config_for_policy(config, mode="paper"),
        starting_equity=config.initial_equity,
        operator_interval_seconds=float(config.mark_minutes * 60),
        policy_preparer=policy.prepare,
    )
    paper = app.state.paper_dashboard
    if paper.state.model_checkpoint is not None:
        persisted_checkpoint = Path(paper.state.model_checkpoint)
        persisted_model = persisted_checkpoint.read_bytes()
        if sha256(persisted_model).hexdigest() != paper.state.model_id:
            raise RuntimeError("persisted Fitted Policy checkpoint does not match Paper Account state")
        fitted_model = persisted_model
        policy.fitted_model = persisted_model
    compatible = (
        not paper.state.compatibility_manifest or paper.state.compatibility_manifest == config.compatibility_manifest
    )
    if paper.state.protocol_id != config.identity_hash or not paper.state.compatibility_manifest:
        paper.register_policy_revision(
            protocol_id=config.identity_hash,
            compatible=compatible,
            compatibility=config.compatibility_manifest,
        )
    model_id = sha256(fitted_model).hexdigest()
    checkpoint = str((operational / "checkpoints" / "current.pt").resolve())
    if compatible and paper.state.model_id == "unknown":
        paper.register_initial_fitted_policy(
            model_id=model_id,
            checkpoint=checkpoint,
        )
    elif compatible and paper.state.model_id != model_id:
        raise RuntimeError("the durable Fitted Policy differs from the selected production checkpoint")
    uvicorn.run(app, host=host, port=port)


def _selected_fitted_policy(operational_directory: Path) -> bytes:
    candidates = (
        operational_directory / "checkpoints" / "current.pt",
        operational_directory / "current-fitted-policy.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            payload = candidate.read_bytes()
            if candidate != candidates[0]:
                candidates[0].parent.mkdir(parents=True, exist_ok=True)
                candidates[0].write_bytes(payload)
            return payload
    evidence_state = operational_directory.parent / "evidence" / "evidence-state.json"
    if evidence_state.is_file():
        raw = json.loads(evidence_state.read_text(encoding="utf-8"))
        artifact = raw.get("validated_artifact")
        expected_hash = raw.get("validated_model_hash")
        if isinstance(artifact, str) and isinstance(expected_hash, str):
            source = Path(artifact) / "model.pt"
            payload = source.read_bytes()
            if sha256(payload).hexdigest() != expected_hash:
                raise RuntimeError("validated Fitted Policy checkpoint hash does not match evidence state")
            candidates[0].parent.mkdir(parents=True, exist_ok=True)
            candidates[0].write_bytes(payload)
            return payload
    raise RuntimeError(f"no selected Fitted Policy is available at {candidates[0]}")
