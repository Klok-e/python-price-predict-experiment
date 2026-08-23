"""Historical Development Evidence workflow."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from .artifacts import RunIdentity, write_artifacts
from .config import PolicyConfig, load_config
from .evidence import EvidenceState, HoldoutEvidence
from .market_data import CanonicalDataset, MarketDataAdapter


@dataclass(frozen=True)
class EvaluationOutcome:
    net_return: float
    max_drawdown: float
    executable_changes: int
    model_bytes: bytes
    equity_rows: tuple[dict[str, Any], ...]
    trade_rows: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PaperPolicyDecision:
    """A fitted or inferred policy result consumed by the Paper Account application."""

    target_weights: dict[str, float]
    model_bytes: bytes
    refitted: bool


class EvaluationBackend(Protocol):
    def validate(self, canonical: CanonicalDataset, config: PolicyConfig, device: str) -> EvaluationOutcome: ...

    def holdout(
        self,
        canonical: CanonicalDataset,
        config: PolicyConfig,
        device: str,
        validated_model: bytes,
    ) -> EvaluationOutcome: ...

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
    ) -> PaperPolicyDecision: ...


class DataSynchronizer(Protocol):
    def sync(self) -> str: ...


@dataclass(frozen=True)
class WorkflowResult:
    summary: str
    artifact_directory: Path


_POLICY_RUNTIME_FILES = (
    "config.py",
    "market_data.py",
    "policy.py",
    "simulation.py",
    "torch_backend.py",
    "training.py",
)


def _policy_code_hash() -> str:
    """Hash Policy Protocol behavior without coupling it to UI or dependency metadata."""
    digest = sha256()
    package = Path(__file__).parent
    for name in _POLICY_RUNTIME_FILES:
        path = package / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class NetGrowthWorkflow:
    """Own synchronization, validation, and one-time Historical Holdout evidence."""

    def __init__(
        self,
        *,
        config: PolicyConfig,
        historical: MarketDataAdapter,
        backend: EvaluationBackend,
        output_directory: str | Path,
        control_directory: str | Path | None = None,
        device: str,
        synchronizer: DataSynchronizer | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.historical = historical
        self.backend = backend
        self.output_directory = Path(output_directory)
        self.control_directory = Path(control_directory) if control_directory is not None else self.output_directory
        self.device = device
        self.synchronizer = synchronizer
        self.now = now or (lambda: datetime.now(UTC))
        self.state_path = self.control_directory / "evidence-state.json"
        self.state = self._load_state()

    @property
    def protocol_hash(self) -> str:
        """Policy Revision identity; Fitted Policy weights and dashboard code are excluded."""
        return sha256(f"{self.config.identity_hash}:{_policy_code_hash()}".encode()).hexdigest()

    @classmethod
    def from_paths(
        cls,
        *,
        config_path: str | Path,
        data_directory: str | Path,
        output_directory: str | Path,
        device: str,
    ) -> NetGrowthWorkflow:
        from .binance import BinanceDataSync, HistoricalArchiveAdapter
        from .torch_backend import TorchEvaluationBackend

        config = load_config(config_path)
        repository = Path(__file__).resolve().parent.parent
        return cls(
            config=config,
            historical=HistoricalArchiveAdapter(Path(data_directory), config.tickers),
            backend=TorchEvaluationBackend(Path(output_directory)),
            output_directory=output_directory,
            control_directory=repository / "computed-data/evidence",
            device=device,
            synchronizer=BinanceDataSync(Path(data_directory), config),
        )

    def _load_state(self) -> EvidenceState:
        if not self.state_path.exists():
            return EvidenceState(drawdown_limit=self.config.drawdown_limit)
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        state = EvidenceState(
            validated_protocol=raw.get("validated_protocol"),
            validation_passed=raw.get("validation_passed", False),
            validated_artifact=raw.get("validated_artifact"),
            validated_model_hash=raw.get("validated_model_hash"),
            drawdown_limit=self.config.drawdown_limit,
        )
        if raw.get("holdout"):
            state.holdout = HoldoutEvidence(**raw["holdout"])
        return state

    def _save_state(self) -> None:
        self.control_directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "validated_protocol": self.state.validated_protocol,
            "validation_passed": self.state.validation_passed,
            "validated_artifact": self.state.validated_artifact,
            "validated_model_hash": self.state.validated_model_hash,
            "holdout": asdict(self.state.holdout) if self.state.holdout else None,
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self.state_path)

    def _write(self, stage: str, canonical: CanonicalDataset, outcome: EvaluationOutcome) -> Path:
        model_hash = sha256(outcome.model_bytes).hexdigest()
        identity = RunIdentity(
            config_hash=self.config.identity_hash,
            code_hash=_policy_code_hash(),
            data_hash=canonical.identity_hash,
            model_hash=model_hash,
        )
        report = {
            "stage": stage,
            "compounded_net_return": float(outcome.net_return),
            "maximum_drawdown": float(outcome.max_drawdown),
            "executable_portfolio_changes": int(outcome.executable_changes),
        }
        report.update(outcome.diagnostics)
        return write_artifacts(
            self.output_directory / stage,
            identity=identity,
            report=report,
            equity_rows=outcome.equity_rows,
            trade_rows=outcome.trade_rows,
            model_bytes=outcome.model_bytes,
        )

    def _existing(self, stage: str) -> Path:
        candidates = sorted((self.output_directory / stage).glob("*/manifest.json"))
        if not candidates:
            raise RuntimeError(f"{stage} evidence state exists but its artifacts are missing")
        return candidates[-1].parent

    def data_sync(self) -> WorkflowResult:
        if self.synchronizer is None:
            raise RuntimeError("no data synchronizer is configured")
        summary = self.synchronizer.sync()
        return WorkflowResult(summary=summary, artifact_directory=Path("."))

    def validate(self) -> WorkflowResult:
        if self.config.development_evidence_end > self.config.holdout_start and self.state.holdout is None:
            raise ValueError(
                "development may include the Historical Holdout period only after its immutable result is recorded"
            )
        canonical = self.historical.load()
        outcome = self.backend.validate(canonical, self.config, self.device)
        passed = bool(outcome.net_return > 0.0 and outcome.max_drawdown <= self.config.drawdown_limit)
        artifact = self._write("validation", canonical, outcome)
        model_hash = sha256(outcome.model_bytes).hexdigest()
        self.state.record_validation(
            self.protocol_hash,
            passed=passed,
            artifact=str(artifact.resolve()) if passed else None,
            model_hash=model_hash if passed else None,
        )
        self._save_state()
        label = "Validated Policy Protocol" if passed else "Validation failed"
        return WorkflowResult(f"{label}: net return {outcome.net_return:.2%}", artifact)

    def _validated_model(self) -> bytes:
        if not self.state.validated_artifact or not self.state.validated_model_hash:
            raise ValueError("validation evidence does not identify its frozen Fitted Policy")
        artifact = Path(self.state.validated_artifact)
        if not artifact.is_absolute():
            artifact = self.output_directory / artifact
        artifact = artifact.resolve()
        payload = (artifact / "model.pt").read_bytes()
        if sha256(payload).hexdigest() != self.state.validated_model_hash:
            raise ValueError("validated Fitted Policy hash does not match its artifact")
        return payload

    def holdout(self) -> WorkflowResult:
        if self.state.holdout is not None:
            label = "Holdout-Passing Protocol" if self.state.holdout.status == "passed" else "Consumed Holdout"
            artifact = Path(self.state.holdout.artifact) if self.state.holdout.artifact else self._existing("holdout")
            if not artifact.is_absolute():
                artifact = self.output_directory / artifact
            if not (artifact / "manifest.json").exists():
                raise RuntimeError("holdout evidence state exists but its exact artifact is missing")
            return WorkflowResult(f"{label}: recorded result", artifact)
        if not self.state.validation_passed or self.state.validated_protocol != self.protocol_hash:
            raise ValueError("successful validation is required before Historical Holdout")
        canonical = self.historical.load()
        outcome = self.backend.holdout(canonical, self.config, self.device, self._validated_model())
        artifact = self._write("holdout", canonical, outcome)
        evidence = self.state.consume_holdout(
            protocol_hash=self.protocol_hash,
            net_return=float(outcome.net_return),
            max_drawdown=float(outcome.max_drawdown),
            artifact=str(artifact.resolve()),
        )
        self._save_state()
        label = "Holdout-Passing Protocol" if evidence.status == "passed" else "Consumed Holdout"
        return WorkflowResult(f"{label}: net return {outcome.net_return:.2%}", artifact)
