from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from threading import Event
from time import monotonic

import numpy as np
import pandas as pd
import pytest

import netgrowth.workflow as workflow_module
from netgrowth.binance import BookTicker, PaperFeedObservation, PaperInstrumentCatchup
from netgrowth.config import load_config
from netgrowth.market_data import InMemoryMarketData
from netgrowth.workflow import EvaluationOutcome, NetGrowthWorkflow, PaperPolicyDecision

from .test_market_data import dataset


@dataclass
class FakeBackend:
    holdout_calls: int = 0
    holdout_model: bytes | None = None

    def validate(self, canonical, config, device):
        del canonical, config, device
        return EvaluationOutcome(np.float64(0.03), np.float64(0.10), 12, b"linear", (), ())

    def holdout(self, canonical, config, device, validated_model):
        del canonical, config, device
        self.holdout_calls += 1
        self.holdout_model = validated_model
        return EvaluationOutcome(-0.01, 0.05, 0, b"linear", (), ())

    fitted_models: list[bytes | None] | None = None

    def paper(
        self,
        canonical,
        config,
        device,
        *,
        validated_model,
        fitted_model,
        observed_at,
        current_weights,
    ):
        del canonical, device, validated_model, observed_at, current_weights
        if self.fitted_models is None:
            self.fitted_models = []
        self.fitted_models.append(fitted_model)
        return PaperPolicyDecision(
            target_weights={**dict.fromkeys(config.tickers, 0.0), "BTCUSDT": 0.5},
            model_bytes=b"paper-fit",
            refitted=fitted_model is None,
        )


@dataclass
class FakePaperFeed:
    canonical: object
    observation: PaperFeedObservation
    mode: str = "live"

    def load(self):
        return self.canonical

    def observe(self, after):
        del after
        return self.observation

    def mark(self, after):
        del after
        return self.observation


@dataclass
class BlockingRefitBackend(FakeBackend):
    refit_started: Event = field(default_factory=Event)
    release_refit: Event = field(default_factory=Event)
    refit_finished: Event = field(default_factory=Event)
    initial_fits: int = 0

    def paper(
        self,
        canonical,
        config,
        device,
        *,
        validated_model,
        fitted_model,
        observed_at,
        current_weights,
    ):
        del canonical, device, validated_model, observed_at, current_weights
        if fitted_model is None:
            self.initial_fits += 1
            if self.initial_fits > 1:
                self.refit_started.set()
                self.release_refit.wait(timeout=2.0)
                self.refit_finished.set()
                return PaperPolicyDecision(
                    target_weights=dict.fromkeys(config.tickers, 0.0),
                    model_bytes=b"sunday-fit",
                    refitted=True,
                )
            return PaperPolicyDecision(
                target_weights=dict.fromkeys(config.tickers, 0.0),
                model_bytes=b"bootstrap-fit",
                refitted=True,
            )
        return PaperPolicyDecision(
            target_weights=dict.fromkeys(config.tickers, 0.0),
            model_bytes=fitted_model,
            refitted=False,
        )


def paper_observation(open_time: pd.Timestamp, close: float = 100.0) -> PaperFeedObservation:
    canonical = dataset()
    observed_at = open_time + pd.Timedelta(minutes=1, seconds=1)
    instruments = {}
    quotes = {}
    for ticker in canonical.tickers:
        row = canonical.instruments[ticker].perpetual.iloc[[-1]].copy()
        row.index = pd.DatetimeIndex([open_time])
        row.loc[:, ["open", "high", "low", "close"]] = close
        instruments[ticker] = PaperInstrumentCatchup(
            perpetual=row,
            spot=row.copy(),
            premium=canonical.instruments[ticker].premium.iloc[:0],
            settled_funding=(),
            open_interest=canonical.instruments[ticker].open_interest.iloc[:0],
        )
        quotes[ticker] = BookTicker(ticker, close - 1.0, close + 1.0, observed_at, observed_at)
    return PaperFeedObservation(observed_at, observed_at, quotes, instruments, canonical.tickers)


def test_deep_workflow_owns_validation_artifacts_and_single_use_holdout(tmp_path) -> None:
    canonical = dataset()
    backend = FakeBackend()
    config = load_config("policy.toml")
    workflow = NetGrowthWorkflow(
        config=replace(config, development_evidence_end=config.holdout_start),
        historical=InMemoryMarketData(canonical, mode="historical"),
        live=InMemoryMarketData(canonical, mode="live"),
        backend=backend,
        output_directory=tmp_path,
        device="cpu",
        now=lambda: datetime(2026, 8, 1, tzinfo=UTC),
    )

    validation = workflow.validate()
    holdout = workflow.holdout()
    repeated = workflow.holdout()

    assert validation.summary.startswith("Validated Policy Protocol")
    assert holdout.summary.startswith("Consumed Holdout")
    assert repeated.artifact_directory == holdout.artifact_directory
    assert backend.holdout_calls == 1
    assert backend.holdout_model == b"linear"
    assert {path.name for path in validation.artifact_directory.iterdir()} == {
        "manifest.json",
        "report.json",
        "equity.csv",
        "trades.csv",
        "model.pt",
    }


def test_holdout_lock_is_independent_of_artifact_output_directory(tmp_path) -> None:
    canonical = dataset()
    backend = FakeBackend()
    base_config = load_config("policy.toml")
    config = replace(base_config, development_evidence_end=base_config.holdout_start)
    control = tmp_path / "repository-control"
    first = NetGrowthWorkflow(
        config=config,
        historical=InMemoryMarketData(canonical, mode="historical"),
        live=InMemoryMarketData(canonical, mode="live"),
        backend=backend,
        output_directory=tmp_path / "run-one",
        control_directory=control,
        device="cpu",
    )
    first.validate()
    consumed = first.holdout()

    second = NetGrowthWorkflow(
        config=config,
        historical=InMemoryMarketData(canonical, mode="historical"),
        live=InMemoryMarketData(canonical, mode="live"),
        backend=backend,
        output_directory=tmp_path / "run-two",
        control_directory=control,
        device="cpu",
    )
    repeated = second.holdout()

    assert repeated.artifact_directory == consumed.artifact_directory
    assert backend.holdout_calls == 1


def test_non_policy_repository_edits_do_not_reset_the_proof_clock(tmp_path, monkeypatch) -> None:
    read_paths: list[Path] = []
    original_read_bytes = Path.read_bytes

    def tracked_read_bytes(path: Path) -> bytes:
        read_paths.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", tracked_read_bytes)
    canonical = dataset()
    workflow = NetGrowthWorkflow(
        config=load_config("policy.toml"),
        historical=InMemoryMarketData(canonical, mode="historical"),
        live=InMemoryMarketData(canonical, mode="live"),
        backend=FakeBackend(),
        output_directory=tmp_path,
        device="cpu",
    )
    protocol = workflow.protocol_hash

    assert workflow.protocol_hash == protocol
    assert read_paths
    assert all(path.parent.name == "netgrowth" or path.name in {"pyproject.toml", "uv.lock"} for path in read_paths)


def test_holdout_is_not_consumed_until_its_immutable_artifact_exists(tmp_path, monkeypatch) -> None:
    canonical = dataset()
    config = load_config("policy.toml")
    workflow = NetGrowthWorkflow(
        config=replace(config, development_evidence_end=config.holdout_start),
        historical=InMemoryMarketData(canonical, mode="historical"),
        live=InMemoryMarketData(canonical, mode="live"),
        backend=FakeBackend(),
        output_directory=tmp_path,
        device="cpu",
    )
    workflow.validate()
    monkeypatch.setattr(workflow, "_write", lambda *args: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        workflow.holdout()

    assert workflow.state.holdout is None


def test_consumed_period_cannot_be_used_as_development_before_holdout_record_exists(tmp_path) -> None:
    canonical = dataset()
    workflow = NetGrowthWorkflow(
        config=load_config("policy.toml"),
        historical=InMemoryMarketData(canonical, mode="historical"),
        live=InMemoryMarketData(canonical, mode="live"),
        backend=FakeBackend(),
        output_directory=tmp_path,
        device="cpu",
    )

    with pytest.raises(ValueError, match="immutable result"):
        workflow.validate()


def test_paper_operation_resumes_pending_fill_and_current_portfolio_without_flattening(tmp_path) -> None:
    canonical = dataset()
    backend = FakeBackend()
    first_open = canonical.instruments["BTCUSDT"].perpetual.index[-1]
    first_feed = FakePaperFeed(canonical, paper_observation(first_open))
    base_config = load_config("policy.toml")
    config = replace(base_config, development_evidence_end=base_config.holdout_start)
    workflow = NetGrowthWorkflow(
        config=config,
        historical=InMemoryMarketData(canonical, mode="historical"),
        live=first_feed,  # type: ignore[arg-type]
        backend=backend,
        output_directory=tmp_path,
        device="cpu",
    )
    workflow.validate()

    first = workflow.paper()
    started_at = workflow.state.paper.started_at if workflow.state.paper else None

    second_feed = FakePaperFeed(canonical, paper_observation(first_open + pd.Timedelta(minutes=1)))
    resumed = NetGrowthWorkflow(
        config=config,
        historical=InMemoryMarketData(canonical, mode="historical"),
        live=second_feed,  # type: ignore[arg-type]
        backend=backend,
        output_directory=tmp_path,
        device="cpu",
    )
    second = resumed.paper()

    assert first.summary.startswith("Forward Paper Proof in progress")
    assert second.summary.startswith("Forward Paper Proof in progress")
    assert resumed.state.paper is not None and resumed.state.paper.started_at == started_at
    assert resumed.state.paper.changes == 1
    assert backend.fitted_models == [None, b"paper-fit"]
    assert {path.read_bytes() for path in Path(tmp_path, "paper-models").glob("*.pt")} == {b"paper-fit"}
    assert Path(second.artifact_directory, "trades.csv").read_text().count("BTCUSDT") == 1
    session = resumed._load_paper_session()
    assert session is not None and session.inference_data_hashes == (canonical.identity_hash,)


def test_sunday_refit_does_not_block_minute_collection_or_reset_proof_clock(tmp_path) -> None:
    canonical = dataset()
    backend = BlockingRefitBackend(refit_started=Event(), release_refit=Event(), refit_finished=Event())
    base_config = load_config("policy.toml")
    config = replace(base_config, development_evidence_end=base_config.holdout_start)
    saturday_open = pd.Timestamp("2026-01-03 23:58:00", tz="UTC")
    workflow = NetGrowthWorkflow(
        config=config,
        historical=InMemoryMarketData(canonical, mode="historical"),
        live=FakePaperFeed(canonical, paper_observation(saturday_open)),  # type: ignore[arg-type]
        backend=backend,
        output_directory=tmp_path,
        device="cpu",
    )
    workflow.validate()
    workflow.paper()
    started_at = workflow.state.paper.started_at if workflow.state.paper else None

    workflow.live = FakePaperFeed(  # type: ignore[assignment]
        canonical,
        paper_observation(saturday_open + pd.Timedelta(minutes=1)),
    )
    before = monotonic()
    workflow.paper()
    elapsed = monotonic() - before

    assert backend.refit_started.wait(timeout=1.0)
    assert elapsed < 1.0
    assert workflow.state.paper is not None and workflow.state.paper.started_at == started_at

    backend.release_refit.set()
    assert backend.refit_finished.wait(timeout=1.0)
    workflow.live = FakePaperFeed(  # type: ignore[assignment]
        canonical,
        paper_observation(saturday_open + pd.Timedelta(minutes=2)),
    )
    workflow.paper()

    assert b"sunday-fit" in {path.read_bytes() for path in Path(tmp_path, "paper-models").glob("*.pt")}
    assert workflow.state.paper is not None and workflow.state.paper.started_at == started_at
    session = workflow._load_paper_session()
    assert session is not None and len(session.fitted_data_hashes) == 2
    assert session.data_hash != canonical.identity_hash


def test_crash_between_checkpoint_write_and_session_switch_keeps_old_checkpoint_loadable(tmp_path, monkeypatch) -> None:
    canonical = dataset()
    backend = FakeBackend()
    base_config = load_config("policy.toml")
    config = replace(base_config, development_evidence_end=base_config.holdout_start)
    open_time = pd.Timestamp("2026-01-03 23:58:00", tz="UTC")
    workflow = NetGrowthWorkflow(
        config=config,
        historical=InMemoryMarketData(canonical, mode="historical"),
        live=FakePaperFeed(canonical, paper_observation(open_time)),  # type: ignore[arg-type]
        backend=backend,
        output_directory=tmp_path,
        device="cpu",
    )
    workflow.validate()
    workflow.paper()
    old_session = workflow._load_paper_session()
    assert old_session is not None and old_session.fitted_model_hash is not None
    old_hash = old_session.fitted_model_hash
    new_model = b"new-sunday-model"
    old_session.fitted_model_hash = sha256(new_model).hexdigest()
    original_replace = Path.replace

    def crash_before_pointer_switch(path: Path, target: Path) -> Path:
        if Path(target) == workflow.paper_session_path:
            raise OSError("simulated crash")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", crash_before_pointer_switch)
    with pytest.raises(OSError, match="simulated crash"):
        workflow._save_paper_session(old_session, new_model)

    resumed = NetGrowthWorkflow(
        config=config,
        historical=InMemoryMarketData(canonical, mode="historical"),
        live=FakePaperFeed(canonical, paper_observation(open_time + pd.Timedelta(minutes=1))),  # type: ignore[arg-type]
        backend=backend,
        output_directory=tmp_path,
        device="cpu",
    )
    recovered = resumed._load_paper_session()

    assert recovered is not None and recovered.fitted_model_hash == old_hash
    assert resumed._fitted_model_bytes(old_hash) == b"paper-fit"


def test_active_paper_append_deduplicates_from_bounded_tail(tmp_path, monkeypatch) -> None:
    path = tmp_path / "equity.csv"
    path.write_text(
        "timestamp,equity\n"
        + "".join(f"2026-01-01T00:{index % 60:02d}:00+00:00,{10_000 + index}\n" for index in range(10_000)),
        encoding="utf-8",
    )
    original = workflow_module.csv.DictReader
    rows_read: list[int] = []

    def bounded_reader(lines, *args, **kwargs):
        materialized = list(lines)
        rows_read.append(len(materialized))
        return original(materialized, *args, **kwargs)

    monkeypatch.setattr(workflow_module.csv, "DictReader", bounded_reader)
    pending = ({"timestamp": "2026-01-01T00:39:00+00:00", "equity": 19_999},)

    NetGrowthWorkflow._append_csv(path, pending, ("timestamp", "equity"))

    assert rows_read == [2]
    assert path.read_text(encoding="utf-8").count("2026-01-01T00:39:00+00:00,19999") == 1


def test_active_paper_recovers_a_truncated_manifest(tmp_path) -> None:
    canonical = dataset()
    config = replace(load_config("policy.toml"), development_evidence_end=load_config("policy.toml").holdout_start)
    workflow = NetGrowthWorkflow(
        config=config,
        historical=InMemoryMarketData(canonical, mode="historical"),
        live=FakePaperFeed(canonical, paper_observation(canonical.instruments["BTCUSDT"].perpetual.index[-1])),  # type: ignore[arg-type]
        backend=FakeBackend(),
        output_directory=tmp_path,
        device="cpu",
    )
    workflow.validate()
    workflow.paper()
    session = workflow._load_paper_session()
    assert session is not None
    active = tmp_path / "paper" / f"active-{workflow.protocol_hash[:16]}"
    (active / "manifest.json").write_text('{"identity":', encoding="utf-8")

    workflow._write_active_paper(session)

    assert json.loads((active / "manifest.json").read_text(encoding="utf-8"))["active"] is True
