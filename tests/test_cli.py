from __future__ import annotations

import pytest

from netgrowth.cli import build_parser


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
