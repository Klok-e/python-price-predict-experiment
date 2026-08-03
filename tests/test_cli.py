from __future__ import annotations

import pytest

from netgrowth.cli import build_parser


def test_cli_exposes_exactly_the_four_policy_workflow_commands() -> None:
    parser = build_parser()
    choices = next(action.choices for action in parser._actions if action.dest == "command")

    assert set(choices) == {"data-sync", "validate", "holdout", "paper"}


def test_cli_allows_paths_and_device_but_rejects_policy_overrides() -> None:
    parser = build_parser()

    arguments = parser.parse_args(["validate", "--data-dir", "/data", "--output-dir", "/runs", "--device", "cpu"])
    assert arguments.data_dir == "/data"
    assert arguments.output_dir == "/runs"
    assert arguments.device == "cpu"

    with pytest.raises(SystemExit):
        parser.parse_args(["validate", "--transaction-cost", "0"])
