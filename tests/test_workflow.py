from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from netgrowth.config import load_config
from netgrowth.market_data import InMemoryMarketData
from netgrowth.workflow import EvaluationOutcome, NetGrowthWorkflow

from .test_market_data import dataset


@dataclass
class FakeBackend:
    holdout_calls: int = 0

    def validate(self, canonical, config, device):
        del canonical, config, device
        return EvaluationOutcome(0.03, 0.10, 12, b"linear", (), ())

    def holdout(self, canonical, config, device):
        del canonical, config, device
        self.holdout_calls += 1
        return EvaluationOutcome(-0.01, 0.05, 0, b"linear", (), ())

    def paper(self, canonical, config, device):
        del canonical, config, device
        return EvaluationOutcome(0.02, 0.05, 101, b"linear", (), ())


def test_deep_workflow_owns_validation_artifacts_and_single_use_holdout(tmp_path) -> None:
    canonical = dataset()
    backend = FakeBackend()
    workflow = NetGrowthWorkflow(
        config=load_config("policy.toml"),
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
    assert {path.name for path in validation.artifact_directory.iterdir()} == {
        "manifest.json",
        "report.json",
        "equity.csv",
        "trades.csv",
        "model.pt",
    }
