"""One deep interface owning data, training/replay, evidence, and artifacts."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

from .artifacts import RunIdentity, write_artifacts
from .config import PolicyConfig, load_config
from .evidence import EvidenceState, HoldoutEvidence, PaperEvidence
from .market_data import CanonicalDataset, MarketDataAdapter


@dataclass(frozen=True)
class EvaluationOutcome:
    net_return: float
    max_drawdown: float
    qualifying_changes: int
    model_bytes: bytes
    equity_rows: tuple[dict[str, Any], ...]
    trade_rows: tuple[dict[str, Any], ...]


class EvaluationBackend(Protocol):
    def validate(self, canonical: CanonicalDataset, config: PolicyConfig, device: str) -> EvaluationOutcome: ...

    def holdout(self, canonical: CanonicalDataset, config: PolicyConfig, device: str) -> EvaluationOutcome: ...

    def paper(self, canonical: CanonicalDataset, config: PolicyConfig, device: str) -> EvaluationOutcome: ...


class DataSynchronizer(Protocol):
    def sync(self) -> str: ...


@dataclass(frozen=True)
class WorkflowResult:
    summary: str
    artifact_directory: Path


def _code_hash() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--no-ext-diff", "--binary", "HEAD"],
            check=True,
            capture_output=True,
        ).stdout
        return sha256(commit.encode() + diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


class NetGrowthWorkflow:
    def __init__(
        self,
        *,
        config: PolicyConfig,
        historical: MarketDataAdapter,
        live: MarketDataAdapter,
        backend: EvaluationBackend,
        output_directory: str | Path,
        device: str,
        synchronizer: DataSynchronizer | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.historical = historical
        self.live = live
        self.backend = backend
        self.output_directory = Path(output_directory)
        self.device = device
        self.synchronizer = synchronizer
        self.now = now or (lambda: datetime.now(UTC))
        self.state_path = self.output_directory / "evidence-state.json"
        self.state = self._load_state()

    @property
    def protocol_hash(self) -> str:
        """Policy Revision identity; scheduled Fitted Policy weights are deliberately excluded."""
        return sha256(f"{self.config.identity_hash}:{_code_hash()}".encode()).hexdigest()

    @classmethod
    def from_paths(
        cls,
        *,
        config_path: str | Path,
        data_directory: str | Path,
        output_directory: str | Path,
        device: str,
    ) -> NetGrowthWorkflow:
        from .binance import BinanceDataSync, HistoricalArchiveAdapter, PublicPaperAdapter
        from .torch_backend import TorchEvaluationBackend

        config = load_config(config_path)
        return cls(
            config=config,
            historical=HistoricalArchiveAdapter(Path(data_directory), config.tickers),
            live=PublicPaperAdapter(Path(data_directory), config.tickers),
            backend=TorchEvaluationBackend(Path(output_directory)),
            output_directory=output_directory,
            device=device,
            synchronizer=BinanceDataSync(Path(data_directory), config),
        )

    def _load_state(self) -> EvidenceState:
        if not self.state_path.exists():
            return EvidenceState(
                proof_days=self.config.proof_days,
                proof_changes=self.config.proof_changes,
                drawdown_limit=self.config.drawdown_limit,
            )
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        state = EvidenceState(
            validated_protocol=raw.get("validated_protocol"),
            validation_passed=raw.get("validation_passed", False),
            proof_days=self.config.proof_days,
            proof_changes=self.config.proof_changes,
            drawdown_limit=self.config.drawdown_limit,
        )
        if raw.get("holdout"):
            state.holdout = HoldoutEvidence(**raw["holdout"])
        if raw.get("paper"):
            paper = raw["paper"]
            paper["started_at"] = datetime.fromisoformat(paper["started_at"])
            paper["observed_at"] = datetime.fromisoformat(paper["observed_at"])
            state.paper = PaperEvidence(**paper)
        return state

    def _save_state(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "validated_protocol": self.state.validated_protocol,
            "validation_passed": self.state.validation_passed,
            "holdout": asdict(self.state.holdout) if self.state.holdout else None,
            "paper": asdict(self.state.paper) if self.state.paper else None,
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, default=lambda value: value.isoformat()) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def _write(self, stage: str, canonical: CanonicalDataset, outcome: EvaluationOutcome) -> Path:
        model_hash = sha256(outcome.model_bytes).hexdigest()
        identity = RunIdentity(
            config_hash=self.config.identity_hash,
            code_hash=_code_hash(),
            data_hash=canonical.identity_hash,
            model_hash=model_hash,
        )
        report = {
            "stage": stage,
            "compounded_net_return": outcome.net_return,
            "maximum_drawdown": outcome.max_drawdown,
            "qualifying_portfolio_changes": outcome.qualifying_changes,
        }
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
        canonical = self.historical.load()
        outcome = self.backend.validate(canonical, self.config, self.device)
        passed = outcome.net_return > 0.0 and outcome.max_drawdown <= self.config.drawdown_limit
        self.state.record_validation(self.protocol_hash, passed=passed)
        self._save_state()
        artifact = self._write("validation", canonical, outcome)
        label = "Validated Policy Protocol" if passed else "Validation failed"
        return WorkflowResult(f"{label}: net return {outcome.net_return:.2%}", artifact)

    def holdout(self) -> WorkflowResult:
        if self.state.holdout is not None:
            label = "Holdout-Passing Protocol" if self.state.holdout.status == "passed" else "Consumed Holdout"
            return WorkflowResult(f"{label}: recorded result", self._existing("holdout"))
        if not self.state.validation_passed or self.state.validated_protocol != self.protocol_hash:
            raise ValueError("successful validation is required before Historical Holdout")
        canonical = self.historical.load()
        outcome = self.backend.holdout(canonical, self.config, self.device)
        evidence = self.state.consume_holdout(
            protocol_hash=self.protocol_hash,
            net_return=outcome.net_return,
            max_drawdown=outcome.max_drawdown,
        )
        self._save_state()
        artifact = self._write("holdout", canonical, outcome)
        label = "Holdout-Passing Protocol" if evidence.status == "passed" else "Consumed Holdout"
        return WorkflowResult(f"{label}: net return {outcome.net_return:.2%}", artifact)

    def paper(self) -> WorkflowResult:
        canonical = self.live.load()
        observed_at = self.now()
        self.state.start_paper(self.protocol_hash, observed_at)
        outcome = self.backend.paper(canonical, self.config, self.device)
        evidence = self.state.record_paper_progress(
            self.protocol_hash,
            observed_at,
            changes=outcome.qualifying_changes,
            net_return=outcome.net_return,
            max_drawdown=outcome.max_drawdown,
        )
        self._save_state()
        artifact = self._write("paper", canonical, outcome)
        status = "passed" if evidence.passed else "failed" if evidence.failed else "in progress"
        return WorkflowResult(f"Forward Paper Proof {status}: net return {outcome.net_return:.2%}", artifact)
