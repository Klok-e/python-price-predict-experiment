from __future__ import annotations

from pathlib import Path

import pytest

from netgrowth.cli import build_parser, main
from netgrowth.market_data import PaperObservationGap, PublicDataUnavailable
from netgrowth.workflow import WorkflowResult


def test_cli_exposes_exactly_the_four_policy_workflow_commands() -> None:
    parser = build_parser()
    choices = next(action.choices for action in parser._actions if action.dest == "command")

    assert set(choices) == {"data-sync", "validate", "holdout", "paper"}


def test_cli_fixes_evidence_paths_and_rejects_policy_overrides() -> None:
    parser = build_parser()

    arguments = parser.parse_args(["validate", "--device", "cpu"])
    assert arguments.device == "cpu"

    with pytest.raises(SystemExit):
        parser.parse_args(["validate", "--transaction-cost", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["holdout", "--output-dir", "/fresh-lock"])


def test_paper_cli_retries_transient_public_data_failure_in_process(monkeypatch, capsys) -> None:
    class TransientWorkflow:
        def __init__(self) -> None:
            self.calls = 0

        def paper(self) -> WorkflowResult:
            self.calls += 1
            if self.calls == 1:
                raise PublicDataUnavailable("BTCUSDT book snapshot is stale")
            return WorkflowResult("paper resumed", Path("."), terminal=True)

    workflow = TransientWorkflow()
    sleeps: list[float] = []
    monkeypatch.setattr(
        "netgrowth.workflow.NetGrowthWorkflow.from_paths",
        lambda **_kwargs: workflow,
    )
    monkeypatch.setattr("netgrowth.cli.time.sleep", sleeps.append)

    assert main(["paper", "--device", "cuda"]) == 0
    assert workflow.calls == 2
    assert sleeps == [2.0]
    assert capsys.readouterr().out == "paper resumed\n"


def test_paper_cli_does_not_retry_an_irreconstructible_observation_gap(monkeypatch) -> None:
    class GappedWorkflow:
        def paper(self) -> WorkflowResult:
            raise PaperObservationGap("required paper minute range has a gap")

    monkeypatch.setattr(
        "netgrowth.workflow.NetGrowthWorkflow.from_paths",
        lambda **_kwargs: GappedWorkflow(),
    )
    monkeypatch.setattr(
        "netgrowth.cli.time.sleep",
        lambda _seconds: pytest.fail("an irreconstructible gap must not be retried"),
    )

    with pytest.raises(PaperObservationGap, match="required paper minute range has a gap"):
        main(["paper", "--device", "cuda"])


def test_paper_cli_escalates_persistent_public_data_failure(monkeypatch, capsys) -> None:
    class UnavailableWorkflow:
        def __init__(self) -> None:
            self.calls = 0

        def paper(self) -> WorkflowResult:
            self.calls += 1
            raise PublicDataUnavailable("Binance public endpoint rejected request")

    workflow = UnavailableWorkflow()
    sleeps: list[float] = []

    def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) > 30:
            pytest.fail("persistent public-data failure was not escalated")

    monkeypatch.setattr(
        "netgrowth.workflow.NetGrowthWorkflow.from_paths",
        lambda **_kwargs: workflow,
    )
    monkeypatch.setattr("netgrowth.cli.time.sleep", record_sleep)

    with pytest.raises(PublicDataUnavailable, match="endpoint rejected"):
        main(["paper", "--device", "cuda"])

    assert workflow.calls == 30
    assert sleeps == [2.0] * 29
    assert capsys.readouterr().err == (
        "public paper data unavailable; retrying in-process: Binance public endpoint rejected request\n"
        "public paper data unavailable after 30 attempts; exiting for service recovery: "
        "Binance public endpoint rejected request\n"
    )
