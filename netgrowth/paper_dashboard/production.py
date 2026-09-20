"""Production wiring for the localhost-only Paper Account service."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import uvicorn
from torch import nn

from netgrowth.attribution import feature_metadata_from_names, integrated_gradients
from netgrowth.binance import PaperFeedObservation, PublicPaperAdapter
from netgrowth.config import PolicyConfig, load_config
from netgrowth.market_data import CanonicalDataset, InstrumentData
from netgrowth.simulation import simulation_config_for_policy
from netgrowth.torch_backend import (
    TorchEvaluationBackend,
    _model_contract,
    _prepare,
    _recent_paper_context,
)

from .application import FittedPolicyCandidate, create_application
from .domain import (
    LifecycleState,
    MarketObservation,
    PolicyDecision,
    eligibility_revision_is_supported,
    policy_revision_is_compatible,
)
from .persistence import SQLitePaperStore
from .revision_evidence import validate_revision_evidence


def _decision_input_id(
    canonical_identity: str,
    signal_time: datetime,
    current_weights: Mapping[str, float],
) -> str:
    payload = (
        canonical_identity
        + ":"
        + signal_time.isoformat()
        + ":"
        + json.dumps(current_weights, sort_keys=True, separators=(",", ":"))
    )
    return sha256(payload.encode()).hexdigest()


def _causal_policy_input(canonical: CanonicalDataset, signal_time: datetime) -> CanonicalDataset:
    """Retain exactly the recent raw observations available at Signal Time."""
    cutoff = pd.Timestamp(signal_time)
    recent = _recent_paper_context(canonical, cutoff)
    return CanonicalDataset(
        instruments={
            ticker: InstrumentData(
                perpetual=data.perpetual.loc[data.perpetual.index < cutoff],
                spot=data.spot.loc[data.spot.index < cutoff],
                funding=data.funding.loc[:cutoff],
                open_interest=data.open_interest.loc[:cutoff],
                premium=data.premium.loc[data.premium.index < cutoff],
            )
            for ticker, data in recent.instruments.items()
        },
        tickers=recent.tickers,
    )


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
    _attribution_inputs: dict[str, bytes] = field(default_factory=dict, init=False, repr=False)
    _inference_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def decide(self, observation: MarketObservation, current_weights: dict[str, float]) -> PolicyDecision:
        with self._inference_lock:
            decision_model = self.fitted_model
            canonical = _causal_policy_input(self.adapter.load(), observation.timestamp)
            input_id = _decision_input_id(canonical.identity_hash, observation.timestamp, current_weights)
            result = self.backend.paper(
                canonical,
                self.config,
                self.device,
                validated_model=decision_model,
                fitted_model=decision_model,
                observed_at=observation.timestamp,
                current_weights=current_weights,
            )
        if result.refitted:
            raise RuntimeError("ordinary decision must not replace the active Fitted Policy")
        model_id = sha256(decision_model).hexdigest()
        self._attribution_inputs[input_id] = self._serialize_attribution_input(
            input_id=input_id,
            canonical=canonical,
            current_weights=current_weights,
            model_id=model_id,
            model_bytes=decision_model,
            observed_at=observation.timestamp,
        )
        return PolicyDecision(
            raw_target_weights=result.target_weights,
            target_weights=result.target_weights,
            model_id=model_id,
            input_id=input_id,
            protocol_id=self.config.protocol_id,
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
        if fitting.get("status") == "failed":
            retry_at = fitting.get("next_retry_at")
            return not isinstance(retry_at, str) or utc_now >= datetime.fromisoformat(retry_at)
        attempted_at = fitting.get("completed_at") or fitting.get("due_at")
        if not isinstance(attempted_at, str):
            return True
        return datetime.fromisoformat(attempted_at).astimezone(UTC) < deadline

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
        _retain_checkpoint(self.checkpoint_directory, candidate.model_id, candidate.payload)
        with self._inference_lock:
            previous = self.fitted_model
            self.fitted_model = candidate.payload

        def rollback() -> None:
            with self._inference_lock:
                self.fitted_model = previous

        return rollback

    def export_attribution_input(self, input_id: str) -> bytes:
        """Transfer the exact prepared explanation input to the Decision Record transaction."""
        try:
            return self._attribution_inputs.pop(input_id)
        except KeyError as error:
            raise RuntimeError("exact prepared attribution input is unavailable for this new decision") from error

    def attribute(self, input_id: str, decision: Mapping[str, Any]) -> dict[str, Any]:
        payload = decision.get("_durable_attribution_input")
        if not isinstance(payload, bytes):
            raise RuntimeError("exact durable attribution input is unavailable for this Decision Record")
        metadata, market_values = self._deserialize_attribution_input(input_id, decision, payload)
        model_id = str(metadata["model_id"])
        model_bytes = self._model_bytes(model_id)
        selected = self.backend._selected_metadata(model_bytes)
        contract = _model_contract(selected, self.config)
        if market_values.shape != (contract.receptive_bars, len(metadata["feature_names"])):
            raise ValueError("durable attribution input does not match the selected model contract")
        models = contract.restore(selected, market_values.shape[1])
        ensemble = _AttributionEnsemble(models)
        market_input = torch.tensor(
            market_values,
            dtype=torch.float32,
        ).unsqueeze(0)
        current = torch.tensor(
            [[float(metadata["current_weights"][ticker]) for ticker in self.config.tickers]],
            dtype=torch.float32,
        )
        result = integrated_gradients(
            ensemble,
            market_input,
            current,
            feature_metadata=feature_metadata_from_names(tuple(metadata["feature_names"]), self.config.tickers),
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

    def _serialize_attribution_input(
        self,
        *,
        input_id: str,
        canonical: CanonicalDataset,
        current_weights: Mapping[str, float],
        model_id: str,
        model_bytes: bytes,
        observed_at: datetime,
    ) -> bytes:
        selected = self.backend._selected_metadata(model_bytes)
        contract = _model_contract(selected, self.config)
        prepared = _prepare(_recent_paper_context(canonical, pd.Timestamp(observed_at)))
        revealed = prepared.state.loc[prepared.state.index <= pd.Timestamp(observed_at)]
        context = revealed.tail(contract.receptive_bars)
        if len(context) < contract.receptive_bars:
            raise ValueError("insufficient revealed Market State for durable Model Attribution")
        metadata = {
            "input_id": input_id,
            "model_id": model_id,
            "model_hash": sha256(model_bytes).hexdigest(),
            "observed_at": observed_at.isoformat(),
            "current_weights": {ticker: float(current_weights[ticker]) for ticker in self.config.tickers},
            "feature_names": list(context.columns),
            "tickers": list(self.config.tickers),
        }
        buffer = BytesIO()
        encoded_metadata = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        np.savez_compressed(
            buffer,
            market_values=context.to_numpy(dtype="float32"),
            metadata=np.frombuffer(encoded_metadata, dtype=np.uint8),
        )
        return buffer.getvalue()

    def _deserialize_attribution_input(
        self,
        input_id: str,
        decision: Mapping[str, Any],
        payload: bytes,
    ) -> tuple[dict[str, Any], np.ndarray]:
        try:
            with np.load(BytesIO(payload), allow_pickle=False) as archive:
                market_values = archive["market_values"]
                metadata = json.loads(bytes(archive["metadata"].tolist()).decode())
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError("durable attribution input is malformed") from error
        if not isinstance(metadata, dict):
            raise RuntimeError("durable attribution metadata is malformed")
        if metadata.get("input_id") != input_id or decision.get("input_id") != input_id:
            raise RuntimeError("durable attribution input identity differs from the Decision Record")
        model_id = decision.get("model_id")
        if not isinstance(model_id, str) or metadata.get("model_id") != model_id:
            raise RuntimeError("durable attribution model identity differs from the Decision Record")
        if metadata.get("model_hash") != model_id:
            raise RuntimeError("durable attribution model hash does not match its identity")
        if metadata.get("tickers") != list(self.config.tickers):
            raise RuntimeError("durable attribution input Trading Universe differs from the active policy")
        current_weights = metadata.get("current_weights")
        feature_names = metadata.get("feature_names")
        if not isinstance(current_weights, dict) or set(current_weights) != set(self.config.tickers):
            raise RuntimeError("durable attribution input lacks the exact Current Portfolio")
        signal_time = decision.get("signal_time")
        expected_signal_time = signal_time.isoformat() if isinstance(signal_time, datetime) else signal_time
        if not isinstance(expected_signal_time, str) or metadata.get("observed_at") != expected_signal_time:
            raise RuntimeError("durable attribution input Signal Time differs from the Decision Record")
        recorded_portfolio = decision.get("current_portfolio")
        if not isinstance(recorded_portfolio, Mapping):
            raise RuntimeError("durable Decision Record lacks the exact Current Portfolio")
        try:
            expected_portfolio = {ticker: float(recorded_portfolio[ticker]) for ticker in self.config.tickers}
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("durable Decision Record lacks the exact Current Portfolio") from error
        if {ticker: float(current_weights[ticker]) for ticker in self.config.tickers} != expected_portfolio:
            raise RuntimeError("durable attribution input Current Portfolio differs from the Decision Record")
        if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
            raise RuntimeError("durable attribution input lacks feature metadata")
        if market_values.dtype != np.dtype("float32") or market_values.ndim != 2:
            raise RuntimeError("durable attribution market input is not a two-dimensional float32 tensor")
        return metadata, market_values

    def diagnostics(self) -> dict[str, str]:
        """Report the configured and available fitting/inference runtime."""
        requested = str(torch.device(self.device))
        device_type = torch.device(requested).type
        rocm_version = torch.version.hip
        available = device_type != "cuda" or torch.cuda.is_available()
        backend = "ROCm" if device_type == "cuda" and rocm_version else "CUDA" if device_type == "cuda" else "CPU"
        actual = requested if available else "unavailable"
        device_name = "CPU"
        if device_type == "cuda" and available:
            device_name = torch.cuda.get_device_name(requested)
        readiness = "ready" if available else "unavailable"
        return {
            "backend": backend,
            "requested_device": requested,
            "actual_device": actual,
            "device_name": device_name,
            "rocm_version": str(rocm_version) if rocm_version else "not-detected",
            "inference": readiness,
            "fitting": readiness,
        }

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


def _retain_checkpoint(directory: Path, model_id: str, payload: bytes) -> Path:
    if sha256(payload).hexdigest() != model_id:
        raise ValueError("checkpoint checksum mismatch")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{model_id}.pt"
    if destination.exists() and destination.read_bytes() != payload:
        raise ValueError("retained checkpoint does not match candidate identity")
    if not destination.exists():
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.replace(destination)
    return destination.resolve()


def run_dashboard(
    config_path: str | Path,
    data_directory: str | Path,
    operational_directory: str | Path,
    host: str,
    port: int,
    device: str,
    revision_bundle: str | Path | None = None,
) -> None:
    """Run the one uvicorn process; never launch a browser or bind beyond the requested host."""
    if host != "127.0.0.1":
        raise ValueError("the Paper Account dashboard is localhost-only")
    config = load_config(config_path)
    operational = Path(operational_directory)
    operational.mkdir(parents=True, exist_ok=True)
    database_path = operational / "paper-account.sqlite3"
    with SQLitePaperStore(database_path, backup_directory=operational / "backups") as store:
        restored = store.load_active_account()
    account_tickers = restored.tickers if restored is not None else config.tickers
    restored_manifest = restored.compatibility_manifest if restored is not None else {}
    restored_lifecycle = restored.lifecycle if restored is not None else LifecycleState.TRADING
    compatible = policy_revision_is_compatible(
        restored_manifest,
        config.compatibility_manifest,
        lifecycle=restored_lifecycle,
    )
    revision_required = restored is None or (
        restored.protocol_id != config.protocol_id or restored.compatibility_manifest != config.compatibility_manifest
    )
    adapter = PublicPaperAdapter(Path(data_directory), account_tickers)
    fitting_adapter = PublicPaperAdapter(Path(data_directory), account_tickers)
    attribution_adapter = PublicPaperAdapter(Path(data_directory), account_tickers)
    checkpoint = (operational / "checkpoints" / "current.pt").resolve()
    if restored is not None and restored.model_id != "unknown":
        if restored.model_checkpoint is None:
            raise RuntimeError("durable Fitted Policy identity has no checkpoint")
        checkpoint = Path(restored.model_checkpoint)
        fitted_model = checkpoint.read_bytes()
        if sha256(fitted_model).hexdigest() != restored.model_id:
            raise RuntimeError("persisted Fitted Policy checkpoint does not match Paper Account state")
    else:
        fitted_model = b"" if revision_bundle is not None else _selected_fitted_policy(operational)
    if restored is None and revision_bundle is not None:
        initial_bundle = json.loads(Path(revision_bundle).read_text())
        validate_revision_evidence(
            initial_bundle,
            protocol_id=config.protocol_id,
            model_id=str(initial_bundle.get("model_id")),
            drawdown_limit=config.drawdown_limit,
            fold_count=config.validation_folds,
        )
        checkpoint = Path(initial_bundle["checkpoint"])
        fitted_model = checkpoint.read_bytes()
        checkpoint = _retain_checkpoint(operational / "checkpoints", str(initial_bundle["model_id"]), fitted_model)
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
    bundle = json.loads(Path(revision_bundle).read_text()) if revision_bundle is not None else None
    if restored is not None and revision_required:
        if restored.revision.get("status") == "draining":
            bundle = restored.revision["validation"]
        if bundle is None or bundle.get("protocol_id") != config.protocol_id:
            raise RuntimeError(
                "Policy Revision requires an offline validated revision bundle; retained model is not relabelled"
            )
    if bundle is not None:
        validate_revision_evidence(
            bundle,
            protocol_id=config.protocol_id,
            model_id=str(bundle.get("model_id")),
            drawdown_limit=config.drawdown_limit,
            fold_count=config.validation_folds,
        )
        if bundle.get("compatibility") != config.compatibility_manifest:
            raise ValueError("revision bundle has different account semantics")
    if restored is None:
        evidence_path = operational.parent / "evidence" / "evidence-state.json"
        evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else {}
        if bundle is None and (
            not evidence.get("validation_passed") or evidence.get("validated_protocol") != config.protocol_id
        ):
            raise RuntimeError("Flat Start requires validation for the current Policy Protocol")
        if bundle is None and sha256(fitted_model).hexdigest() != evidence.get("validated_model_hash"):
            raise RuntimeError("Flat Start checkpoint does not match the validated model identity")
    app = create_application(
        database_path=database_path,
        backup_directory=operational / "backups",
        tickers=account_tickers,
        market_feed=ProductionMarketFeed(adapter),
        policy_backend=policy,
        policy_fitter=policy,
        attribution_backend=policy,
        notifications=DesktopNotifications(),
        simulation_config=replace(simulation_config_for_policy(config, mode="paper"), tickers=account_tickers),
        starting_equity=config.initial_equity,
        operator_interval_seconds=float(config.mark_minutes * 60),
        policy_preparer=policy.prepare
        if compatible or eligibility_revision_is_supported(restored_manifest, config.compatibility_manifest)
        else None,
    )
    paper = app.state.paper_dashboard
    model_id = sha256(fitted_model).hexdigest()
    if restored is not None and revision_required:
        assert bundle is not None
        candidate_path = Path(bundle["checkpoint"])
        payload = candidate_path.read_bytes()
        candidate_path = _retain_checkpoint(operational / "checkpoints", str(bundle["model_id"]), payload)
        candidate = FittedPolicyCandidate(str(bundle["model_id"]), str(candidate_path), payload)
        paper.stage_policy_revision(
            protocol_id=config.protocol_id,
            compatibility=config.compatibility_manifest,
            candidate=candidate,
            validation=bundle,
        )
    elif paper.state.model_id == "unknown":
        paper.register_policy_revision(protocol_id=config.protocol_id, compatibility=config.compatibility_manifest)
        paper.register_initial_fitted_policy(
            model_id=model_id, checkpoint=str(checkpoint), fitted_at=bundle.get("fitted_at") if bundle else None
        )
    elif paper.state.model_id != model_id:
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
