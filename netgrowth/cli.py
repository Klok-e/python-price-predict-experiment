"""Thin command-line routing for the one deep workflow interface."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime

_PAPER_DATA_RETRY_ATTEMPTS = 30
_PAPER_DATA_RETRY_DELAY_SECONDS = 2.0


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
    from .market_data import PaperObservationGap, PublicDataUnavailable
    from .workflow import NetGrowthWorkflow

    arguments = build_parser().parse_args(argv)
    workflow = NetGrowthWorkflow.from_paths(
        config_path=arguments.config,
        data_directory="computed-data/dataset",
        output_directory="computed-data/evidence",
        device=getattr(arguments, "device", "cpu"),
    )
    if arguments.command == "paper":
        consecutive_data_failures = 0
        while True:
            try:
                result = workflow.paper()
            except PaperObservationGap:
                raise
            except PublicDataUnavailable as error:
                consecutive_data_failures += 1
                if consecutive_data_failures == 1:
                    print(f"public paper data unavailable; retrying in-process: {error}", file=sys.stderr, flush=True)
                if consecutive_data_failures >= _PAPER_DATA_RETRY_ATTEMPTS:
                    print(
                        f"public paper data unavailable after {_PAPER_DATA_RETRY_ATTEMPTS} attempts; "
                        f"exiting for service recovery: {error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise
                time.sleep(_PAPER_DATA_RETRY_DELAY_SECONDS)
                continue
            consecutive_data_failures = 0
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
