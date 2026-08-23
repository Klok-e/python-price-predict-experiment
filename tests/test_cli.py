from __future__ import annotations

import pytest

from netgrowth.cli import build_parser, main


def test_cli_exposes_exactly_the_four_policy_workflow_commands() -> None:
    parser = build_parser()
    choices = next(action.choices for action in parser._actions if action.dest == "command")

    assert set(choices) == {"data-sync", "validate", "holdout", "serve"}


def test_cli_fixes_evidence_paths_and_rejects_policy_overrides() -> None:
    parser = build_parser()

    arguments = parser.parse_args(["validate", "--device", "cpu"])
    assert arguments.device == "cpu"

    with pytest.raises(SystemExit):
        parser.parse_args(["validate", "--transaction-cost", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["holdout", "--output-dir", "/fresh-lock"])


def test_serve_defaults_to_the_local_dashboard_boundary() -> None:
    arguments = build_parser().parse_args(["serve"])

    assert arguments.host == "127.0.0.1"
    assert arguments.port == 8765
    assert arguments.device == "cpu"
    assert arguments.operational_directory == "computed-data/paper-dashboard"


def test_removed_proof_command_and_options_are_rejected() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["paper"])
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--proof-days", "60"])


def test_serve_routes_only_to_the_dashboard_runner(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "netgrowth.paper_dashboard.production.run_dashboard",
        lambda **kwargs: calls.append(kwargs),
    )

    assert main(["serve", "--device", "cuda"]) == 0
    assert calls == [
        {
            "config_path": "policy.toml",
            "data_directory": "computed-data/dataset",
            "operational_directory": "computed-data/paper-dashboard",
            "host": "127.0.0.1",
            "port": 8765,
            "device": "cuda",
        }
    ]
