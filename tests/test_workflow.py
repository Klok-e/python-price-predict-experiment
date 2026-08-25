from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from netgrowth.config import load_config
from netgrowth.market_data import InMemoryMarketData
from netgrowth.workflow import EvaluationOutcome, NetGrowthWorkflow

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


def test_deep_workflow_owns_validation_artifacts_and_single_use_holdout(tmp_path) -> None:
    canonical = dataset()
    backend = FakeBackend()
    config = load_config("policy.toml")
    workflow = NetGrowthWorkflow(
        config=replace(config, development_evidence_end=config.holdout_start),
        historical=InMemoryMarketData(canonical, mode="historical"),
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
        backend=backend,
        output_directory=tmp_path / "run-two",
        control_directory=control,
        device="cpu",
    )
    repeated = second.holdout()

    assert repeated.artifact_directory == consumed.artifact_directory
    assert backend.holdout_calls == 1


def test_dashboard_and_dependency_files_do_not_change_policy_revision(tmp_path, monkeypatch) -> None:
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
        backend=FakeBackend(),
        output_directory=tmp_path,
        device="cpu",
    )
    protocol = workflow.protocol_hash

    assert workflow.protocol_hash == protocol
    assert protocol == workflow.config.protocol_id
    assert {path.name for path in read_paths} == {
        "config.py",
        "market_data.py",
        "policy.py",
        "simulation.py",
        "torch_backend.py",
        "training.py",
    }


def test_holdout_is_not_consumed_until_its_immutable_artifact_exists(tmp_path, monkeypatch) -> None:
    canonical = dataset()
    config = load_config("policy.toml")
    workflow = NetGrowthWorkflow(
        config=replace(config, development_evidence_end=config.holdout_start),
        historical=InMemoryMarketData(canonical, mode="historical"),
        backend=FakeBackend(),
        output_directory=tmp_path,
        device="cpu",
    )
    workflow.validate()
    monkeypatch.setattr(workflow, "_write", lambda *args: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError, match="disk full"):
        workflow.holdout()

    assert workflow.state.holdout is None
