"""Thin command-line routing for the one deep workflow interface."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from datetime import UTC, datetime


def _paths(parser: argparse.ArgumentParser, *, device: bool = True) -> None:
    parser.add_argument("--config", default="policy.toml", help="checked-in Policy Protocol TOML")
    if device:
        parser.add_argument("--device", default="cpu")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="netgrowth", description="Direct Net Log Growth workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    _paths(commands.add_parser("data-sync", help="synchronize public Binance-native data"), device=False)
    _paths(commands.add_parser("validate", help="run purged prequential validation"))
    _paths(commands.add_parser("holdout", help="evaluate the frozen Historical Holdout once"))
    _paths(commands.add_parser("paper", help="run public-data-only Forward Paper Proof"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from .workflow import NetGrowthWorkflow

    arguments = build_parser().parse_args(argv)
    workflow = NetGrowthWorkflow.from_paths(
        config_path=arguments.config,
        data_directory="computed-data/dataset",
        output_directory="computed-data/evidence",
        device=getattr(arguments, "device", "cpu"),
    )
    if arguments.command == "paper":
        while True:
            result = workflow.paper()
            print(result.summary, flush=True)
            if result.terminal:
                break
            now = datetime.now(UTC).timestamp()
            next_minute = (int(now) // 60 + 1) * 60 + 2
            time.sleep(max(0.0, next_minute - now))
    else:
        result = getattr(workflow, arguments.command.replace("-", "_"))()
        print(result.summary)
    return 0
