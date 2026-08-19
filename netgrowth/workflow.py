"""One deep interface owning data, training/replay, evidence, and artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .artifacts import RunIdentity, write_artifacts
from .config import PolicyConfig, load_config
from .evidence import EvidenceState, HoldoutEvidence, PaperEvidence
from .market_data import CanonicalDataset, MarketDataAdapter
from .simulation import (
    MarketMinute,
    SimulationState,
    advance_simulation,
    marked_weights,
    schedule_portfolio_change,
    simulation_config_for_policy,
)


@dataclass(frozen=True)
class EvaluationOutcome:
    net_return: float
    max_drawdown: float
    qualifying_changes: int
    model_bytes: bytes
    equity_rows: tuple[dict[str, Any], ...]
    trade_rows: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PaperPolicyDecision:
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


class PaperObservation(Protocol):
    def market_minute(self, after: datetime | None) -> MarketMinute: ...


class PaperMarketDataAdapter(MarketDataAdapter, Protocol):
    def observe(self, after: datetime | None) -> PaperObservation: ...

    def mark(self, after: datetime | None) -> PaperObservation: ...


@dataclass(frozen=True)
class WorkflowResult:
    summary: str
    artifact_directory: Path
    terminal: bool = False


@dataclass
class _PaperSession:
    protocol_hash: str
    simulation: SimulationState
    fitted_at: datetime | None = None
    fitted_model_hash: str | None = None
    data_hash: str | None = None
    fitted_data_hashes: tuple[str, ...] = ()
    inference_data_hashes: tuple[str, ...] = ()
    pending_equity_rows: tuple[dict[str, Any], ...] = ()
    pending_trade_rows: tuple[dict[str, Any], ...] = ()


def _policy_code_hash() -> str:
    """Hash runtime policy behavior without coupling identity to Git or documentation state."""
    digest = sha256()
    package = Path(__file__).parent
    for path in sorted(package.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    repository = package.parent
    for name in ("pyproject.toml", "uv.lock"):
        path = repository / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


class NetGrowthWorkflow:
    def __init__(
        self,
        *,
        config: PolicyConfig,
        historical: MarketDataAdapter,
        live: PaperMarketDataAdapter,
        backend: EvaluationBackend,
        output_directory: str | Path,
        control_directory: str | Path | None = None,
        device: str,
        synchronizer: DataSynchronizer | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self.historical = historical
        self.live = live
        self.backend = backend
        self.output_directory = Path(output_directory)
        self.control_directory = Path(control_directory) if control_directory is not None else self.output_directory
        self.device = device
        self.synchronizer = synchronizer
        self.now = now or (lambda: datetime.now(UTC))
        self.state_path = self.control_directory / "evidence-state.json"
        self.paper_session_path = self.control_directory / "paper-session.json"
        self.paper_models_directory = self.control_directory / "paper-models"
        self.legacy_paper_model_path = self.control_directory / "paper-fitted-policy.pt"
        self._paper_refit_executor: ThreadPoolExecutor | None = None
        self._paper_refit_future: Future[PaperPolicyDecision] | None = None
        self._paper_refit_context: tuple[datetime, str] | None = None
        self.state = self._load_state()

    @property
    def protocol_hash(self) -> str:
        """Policy Revision identity; scheduled Fitted Policy weights are deliberately excluded."""
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
        from .binance import BinanceDataSync, HistoricalArchiveAdapter, PublicPaperAdapter
        from .torch_backend import TorchEvaluationBackend

        config = load_config(config_path)
        repository = Path(__file__).resolve().parent.parent
        return cls(
            config=config,
            historical=HistoricalArchiveAdapter(Path(data_directory), config.tickers),
            live=PublicPaperAdapter(Path(data_directory), config.tickers),
            backend=TorchEvaluationBackend(Path(output_directory)),
            output_directory=output_directory,
            control_directory=repository / "computed-data/evidence",
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
            validated_artifact=raw.get("validated_artifact"),
            validated_model_hash=raw.get("validated_model_hash"),
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
        self.control_directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "validated_protocol": self.state.validated_protocol,
            "validation_passed": self.state.validation_passed,
            "validated_artifact": self.state.validated_artifact,
            "validated_model_hash": self.state.validated_model_hash,
            "holdout": asdict(self.state.holdout) if self.state.holdout else None,
            "paper": asdict(self.state.paper) if self.state.paper else None,
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, default=lambda value: value.isoformat()) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    def _load_paper_session(self) -> _PaperSession | None:
        if not self.paper_session_path.exists():
            return None
        raw = json.loads(self.paper_session_path.read_text(encoding="utf-8"))
        if raw["protocol_hash"] != self.protocol_hash:
            return None
        session = _PaperSession(
            protocol_hash=raw["protocol_hash"],
            simulation=SimulationState.from_payload(raw["simulation"]),
            fitted_at=datetime.fromisoformat(raw["fitted_at"]) if raw.get("fitted_at") else None,
            fitted_model_hash=raw.get("fitted_model_hash"),
            data_hash=raw.get("data_hash"),
            fitted_data_hashes=tuple(raw.get("fitted_data_hashes", ())),
            inference_data_hashes=tuple(raw.get("inference_data_hashes", ())),
            pending_equity_rows=tuple(raw.get("pending_equity_rows", ())),
            pending_trade_rows=tuple(raw.get("pending_trade_rows", ())),
        )
        if session.fitted_model_hash:
            self._fitted_model_bytes(session.fitted_model_hash)
        return session

    def _model_checkpoint(self, model_hash: str) -> Path:
        return self.paper_models_directory / f"{model_hash}.pt"

    def _fitted_model_bytes(self, model_hash: str) -> bytes:
        path = self._model_checkpoint(model_hash)
        if not path.exists() and self.legacy_paper_model_path.exists():
            legacy = self.legacy_paper_model_path.read_bytes()
            if sha256(legacy).hexdigest() == model_hash:
                self.paper_models_directory.mkdir(parents=True, exist_ok=True)
                path.write_bytes(legacy)
        payload = path.read_bytes()
        if sha256(payload).hexdigest() != model_hash:
            raise ValueError("paper Fitted Policy hash does not match its immutable checkpoint")
        return payload

    def _save_paper_session(self, session: _PaperSession, model_bytes: bytes | None = None) -> None:
        self.control_directory.mkdir(parents=True, exist_ok=True)
        if model_bytes is not None:
            model_hash = sha256(model_bytes).hexdigest()
            if session.fitted_model_hash != model_hash:
                raise ValueError("paper session pointer must match the Fitted Policy checkpoint")
            self.paper_models_directory.mkdir(parents=True, exist_ok=True)
            checkpoint = self._model_checkpoint(model_hash)
            if not checkpoint.exists():
                model_temporary = checkpoint.with_suffix(".tmp")
                model_temporary.write_bytes(model_bytes)
                model_temporary.replace(checkpoint)
        payload = {
            "protocol_hash": session.protocol_hash,
            "simulation": session.simulation.to_payload(),
            "fitted_at": session.fitted_at,
            "fitted_model_hash": session.fitted_model_hash,
            "data_hash": session.data_hash,
            "fitted_data_hashes": session.fitted_data_hashes,
            "inference_data_hashes": session.inference_data_hashes,
            "pending_equity_rows": session.pending_equity_rows,
            "pending_trade_rows": session.pending_trade_rows,
        }
        temporary = self.paper_session_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, default=lambda value: value.isoformat()) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.paper_session_path)

    def _start_paper_refit(
        self,
        canonical: CanonicalDataset,
        *,
        observed_at: datetime,
        current_weights: dict[str, float],
    ) -> None:
        if self._paper_refit_future is not None:
            return
        if self._paper_refit_executor is None:
            self._paper_refit_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="policy-refit")
        self._paper_refit_context = (observed_at, canonical.identity_hash)
        self._paper_refit_future = self._paper_refit_executor.submit(
            self.backend.paper,
            canonical,
            self.config,
            self.device,
            validated_model=self._validated_model(),
            fitted_model=None,
            observed_at=observed_at,
            current_weights=current_weights,
        )

    def _complete_paper_refit(self, session: _PaperSession) -> bytes | None:
        future = self._paper_refit_future
        if future is None or not future.done():
            return None
        context = self._paper_refit_context
        self._paper_refit_future = None
        self._paper_refit_context = None
        if context is None:
            raise RuntimeError("completed Fitted Policy is missing its causal refit identity")
        decision = future.result()
        if not decision.refitted:
            raise RuntimeError("scheduled Policy Handoff requires a newly fitted checkpoint")
        fitted_at, data_hash = context
        session.fitted_at = fitted_at
        session.fitted_data_hashes += (data_hash,)
        session.data_hash = sha256(
            (session.data_hash or "").encode() + b":fitted-data:" + data_hash.encode()
        ).hexdigest()
        session.fitted_model_hash = sha256(decision.model_bytes).hexdigest()
        self.state.record_fitted_policy_handoff(
            self.protocol_hash,
            fitted_policy_hash=session.fitted_model_hash,
        )
        self._save_paper_session(session, decision.model_bytes)
        self._save_state()
        return decision.model_bytes

    @staticmethod
    def _append_csv(path: Path, rows: tuple[dict[str, Any], ...], fields: tuple[str, ...]) -> None:
        if not rows:
            if not path.exists():
                with path.open("w", newline="", encoding="utf-8") as handle:
                    csv.DictWriter(handle, fieldnames=fields).writeheader()
            return
        existing: set[tuple[str, ...]] = set()
        if path.exists():
            with path.open("rb") as handle:
                header = handle.readline().decode()
                handle.seek(0, 2)
                tail_start = max(len(header.encode()), handle.tell() - 65_536)
                handle.seek(tail_start)
                if tail_start > len(header.encode()):
                    handle.readline()
                tail = [line.decode() for line in handle.readlines()[-max(1, len(rows)) :]]
            existing = {tuple(row.get(field, "") for field in fields) for row in csv.DictReader([header, *tail])}
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if path.stat().st_size == 0:
                writer.writeheader()
            for row in rows:
                key = tuple(str(row.get(field, "")) for field in fields)
                if key not in existing:
                    writer.writerow(row)

    def _write_active_paper(self, session: _PaperSession) -> Path:
        directory = self.output_directory / "paper" / f"active-{self.protocol_hash[:16]}"
        directory.mkdir(parents=True, exist_ok=True)
        equity_fields = ("timestamp", "equity", "drawdown", "gross_exposure")
        trade_fields = (
            "timestamp",
            "ticker",
            "quantity",
            "reference_price",
            "effective_fill",
            "target_weight",
            "bid",
            "ask",
            "quote_exchange_time",
            "quote_observed_at",
        )
        self._append_csv(directory / "equity.csv", session.pending_equity_rows, equity_fields)
        self._append_csv(directory / "trades.csv", session.pending_trade_rows, trade_fields)
        manifest_path = directory / "manifest.json"
        try:
            previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        except json.JSONDecodeError:
            previous_manifest = {}
        previous_model_hash = previous_manifest.get("identity", {}).get("model_hash")
        if session.fitted_model_hash and previous_model_hash != session.fitted_model_hash:
            _atomic_write(directory / "model.pt", self._fitted_model_bytes(session.fitted_model_hash))
        report = self._paper_report(session, "active")
        report_bytes = (json.dumps(report, sort_keys=True) + "\n").encode()
        _atomic_write(directory / "report.json", report_bytes)
        result_digest = sha256(report_bytes)
        result_digest.update((session.data_hash or "pending").encode())
        result_digest.update((session.fitted_model_hash or "pending").encode())
        result_digest.update(str(session.simulation.previous_timestamp).encode())
        result_digest.update(str(session.simulation.qualifying_portfolio_changes).encode())
        manifest = {
            "identity": {
                "config_hash": self.config.identity_hash,
                "code_hash": _policy_code_hash(),
                "data_hash": session.data_hash or "pending",
                "model_hash": session.fitted_model_hash or "pending",
            },
            "protocol_hash": self.protocol_hash,
            "active": True,
            "result_hash": result_digest.hexdigest(),
        }
        _atomic_write(manifest_path, (json.dumps(manifest, sort_keys=True) + "\n").encode())
        return directory

    def _finalize_paper(self, session: _PaperSession, active: Path, status: str) -> Path:
        if not session.data_hash or not session.fitted_model_hash:
            raise RuntimeError("Forward Paper Proof cannot freeze without data and Fitted Policy identities")

        def rows(path: Path) -> tuple[dict[str, Any], ...]:
            with path.open(newline="", encoding="utf-8") as handle:
                return tuple(dict(row) for row in csv.DictReader(handle))

        report = self._paper_report(session, status)
        return write_artifacts(
            self.output_directory / "paper",
            identity=RunIdentity(
                config_hash=self.config.identity_hash,
                code_hash=_policy_code_hash(),
                data_hash=session.data_hash,
                model_hash=session.fitted_model_hash,
            ),
            report=report,
            equity_rows=rows(active / "equity.csv"),
            trade_rows=rows(active / "trades.csv"),
            model_bytes=self._fitted_model_bytes(session.fitted_model_hash),
        )

    def _paper_report(self, session: _PaperSession, status: str) -> dict[str, Any]:
        return {
            "stage": "paper",
            "status": status,
            "compounded_net_return": session.simulation.equity / self.config.initial_equity - 1.0,
            "maximum_drawdown": session.simulation.max_drawdown,
            "maximum_gross_exposure": session.simulation.max_gross_exposure,
            "maximum_instrument_exposure": session.simulation.max_instrument_exposure,
            "qualifying_portfolio_changes": session.simulation.qualifying_portfolio_changes,
            "turnover_notional": session.simulation.turnover_notional,
            "transaction_cost": session.simulation.transaction_cost,
            "funding_cashflow": session.simulation.funding_cashflow,
            "risk_stop_triggered": session.simulation.risk_stop_time is not None,
            "fitted_data_hashes": session.fitted_data_hashes,
            "inference_data_hashes": session.inference_data_hashes,
        }

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
            "qualifying_portfolio_changes": int(outcome.qualifying_changes),
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

    def paper(self) -> WorkflowResult:
        if not self.state.validation_passed or self.state.validated_protocol != self.protocol_hash:
            raise ValueError(
                "successful validation of the current Policy Protocol is required before Forward Paper Proof"
            )
        session = self._load_paper_session()
        if session is not None and (session.pending_equity_rows or session.pending_trade_rows):
            self._write_active_paper(session)
            session.pending_equity_rows = ()
            session.pending_trade_rows = ()
            self._save_paper_session(session)
        if (
            self.state.paper is not None
            and self.state.paper.protocol_hash == self.protocol_hash
            and (
                self.state.paper.passed
                or (
                    self.state.paper.failed
                    and session is not None
                    and (session.simulation.flattened or session.simulation.risk_stop_time is None)
                )
            )
        ):
            assert session is not None
            status = "passed" if self.state.paper.passed else "failed"
            return WorkflowResult(
                f"Forward Paper Proof {status}: recorded result",
                self._write_active_paper(session),
                terminal=True,
            )

        feed = self.live
        simulation_config = simulation_config_for_policy(self.config, mode="paper")
        bootstrap_model: bytes | None = None
        bootstrap_time: datetime | None = None
        bootstrap_data_hash: str | None = None
        if session is None:
            probe = feed.mark(None)
            probe_minute = probe.market_minute(None)
            canonical = self.live.load()
            bootstrap = self.backend.paper(
                canonical,
                self.config,
                self.device,
                validated_model=self._validated_model(),
                fitted_model=None,
                observed_at=probe_minute.timestamp,
                current_weights=dict.fromkeys(self.config.tickers, 0.0),
            )
            bootstrap_model = bootstrap.model_bytes
            bootstrap_time = probe_minute.timestamp
            bootstrap_data_hash = canonical.identity_hash

        after = session.simulation.previous_timestamp if session is not None else None
        observation = feed.mark(after)
        minute = observation.market_minute(after)
        replay, simulation = advance_simulation(
            [minute],
            simulation_config,
            session.simulation if session is not None else None,
        )
        if session is None:
            assert bootstrap_model is not None and bootstrap_time is not None and bootstrap_data_hash is not None
            session = _PaperSession(
                protocol_hash=self.protocol_hash,
                simulation=simulation,
                fitted_at=bootstrap_time,
                fitted_model_hash=sha256(bootstrap_model).hexdigest(),
                data_hash=bootstrap_data_hash,
                fitted_data_hashes=(bootstrap_data_hash,),
                inference_data_hashes=(),
            )
        else:
            session.simulation = simulation

        self.state.start_paper(self.protocol_hash, minute.timestamp)
        if bootstrap_model is not None:
            assert session.fitted_model_hash is not None
            self.state.record_fitted_policy_handoff(
                self.protocol_hash,
                fitted_policy_hash=session.fitted_model_hash,
            )
        self._save_state()

        completed_model = self._complete_paper_refit(session)
        fitted_model = (
            completed_model
            or bootstrap_model
            or (self._fitted_model_bytes(session.fitted_model_hash) if session.fitted_model_hash else None)
        )
        signal = pd.Timestamp(minute.timestamp)
        latest_sunday = signal.normalize() - pd.Timedelta(days=(signal.dayofweek - 6) % 7)
        refit_due = session.fitted_at is None or pd.Timestamp(session.fitted_at) < latest_sunday <= signal
        decision_due = signal.minute % self.config.decision_minutes == 0
        inference_data_hash: str | None = None
        if simulation.risk_stop_time is None and (refit_due or decision_due):
            canonical = self.live.load()
            inference_data_hash = canonical.identity_hash
            session.inference_data_hashes += (inference_data_hash,)
            if refit_due:
                self._start_paper_refit(
                    canonical,
                    observed_at=minute.timestamp,
                    current_weights=marked_weights(simulation, minute.mark_prices, self.config.tickers),
                )
            if decision_due:
                if fitted_model is None:
                    raise RuntimeError("Forward Paper Proof decision requires an active Fitted Policy")
                decision = self.backend.paper(
                    canonical,
                    self.config,
                    self.device,
                    validated_model=self._validated_model(),
                    fitted_model=fitted_model,
                    observed_at=minute.timestamp,
                    current_weights=marked_weights(simulation, minute.mark_prices, self.config.tickers),
                )
                if decision.refitted:
                    raise RuntimeError("ordinary paper inference cannot replace the scheduled Fitted Policy")
                schedule_portfolio_change(simulation, minute.timestamp, decision.target_weights, simulation_config)

        observation_payload = json.dumps(
            {
                "timestamp": minute.timestamp,
                "mark_prices": minute.mark_prices,
                "bid": minute.bid,
                "ask": minute.ask,
                "quote_exchange_times": minute.quote_exchange_times,
                "quote_observed_at": minute.quote_observed_at,
                "funding_rates": minute.funding_rates,
                "funding_mark_prices": minute.funding_mark_prices,
            },
            sort_keys=True,
            default=lambda value: value.isoformat(),
        ).encode()
        identity_payload = (session.data_hash or "").encode() + observation_payload
        if inference_data_hash is not None:
            identity_payload += b":inference:" + inference_data_hash.encode()
        session.data_hash = sha256(identity_payload).hexdigest()

        session.pending_equity_rows = tuple(
            {
                "timestamp": point.timestamp.isoformat(),
                "equity": point.equity,
                "drawdown": point.drawdown,
                "gross_exposure": point.gross_exposure,
            }
            for point in replay.equity
        )
        session.pending_trade_rows = tuple(
            {
                "timestamp": trade.timestamp.isoformat(),
                "ticker": trade.ticker,
                "quantity": trade.quantity,
                "reference_price": trade.reference_price,
                "effective_fill": trade.effective_fill,
                "target_weight": trade.target_weight,
                "bid": trade.bid,
                "ask": trade.ask,
                "quote_exchange_time": trade.quote_exchange_time.isoformat() if trade.quote_exchange_time else None,
                "quote_observed_at": trade.quote_observed_at.isoformat() if trade.quote_observed_at else None,
            }
            for trade in replay.trades
        )
        self._save_paper_session(session, fitted_model)
        evidence = self.state.record_paper_progress(
            self.protocol_hash,
            minute.timestamp,
            changes=simulation.qualifying_portfolio_changes,
            net_return=simulation.equity / self.config.initial_equity - 1.0,
            max_drawdown=simulation.max_drawdown,
        )
        self._save_state()
        artifact = self._write_active_paper(session)
        session.pending_equity_rows = ()
        session.pending_trade_rows = ()
        self._save_paper_session(session)
        status = "passed" if evidence.passed else "failed" if evidence.failed else "in progress"
        terminal = evidence.passed or (evidence.failed and (simulation.flattened or simulation.risk_stop_time is None))
        if terminal:
            artifact = self._finalize_paper(session, artifact, status)
        net_return = simulation.equity / self.config.initial_equity - 1.0
        return WorkflowResult(
            f"Forward Paper Proof {status}: net return {net_return:.2%}",
            artifact,
            terminal=terminal,
        )
