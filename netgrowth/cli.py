"""Thin command-line routing for the one deep workflow interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def _paths(parser: argparse.ArgumentParser, *, device: bool = True) -> None:
    parser.add_argument("--config", default="policy.toml", help="checked-in Policy Protocol TOML")
    parser.add_argument("--data-dir", default="computed-data/dataset")
    parser.add_argument("--output-dir", default="computed-data/evidence")
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
        data_directory=arguments.data_dir,
        output_directory=arguments.output_dir,
        device=getattr(arguments, "device", "cpu"),
    )
    result = getattr(workflow, arguments.command.replace("-", "_"))()
    print(result.summary)
    return 0
