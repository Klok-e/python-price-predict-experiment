"""Thin command-line routing for the one deep workflow interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


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
    serve = commands.add_parser("serve", help="run the persistent local Paper Account dashboard")
    _paths(serve)
    serve.add_argument("--data-directory", default="computed-data/dataset")
    serve.add_argument("--operational-directory", default="computed-data/paper-dashboard")
    serve.add_argument("--revision-bundle", help="validated revision.json prepared offline")
    prepare = commands.add_parser("prepare-revision", help="evaluate the incumbent architecture and prepare a revision")
    _paths(prepare)
    prepare.add_argument("--data-directory", required=True)
    prepare.add_argument("--checkpoint", required=True)
    prepare.add_argument("--evidence-state", required=True)
    prepare.add_argument("--output-directory", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "prepare-revision":
        from .revision import prepare_revision

        result = prepare_revision(
            config_path=arguments.config,
            data_directory=arguments.data_directory,
            checkpoint=arguments.checkpoint,
            evidence_state=arguments.evidence_state,
            output_directory=arguments.output_directory,
            device=arguments.device,
        )
        print(result)
        return 0
    if arguments.command == "serve":
        from .paper_dashboard.production import run_dashboard

        run_dashboard(
            config_path=arguments.config,
            data_directory=arguments.data_directory,
            operational_directory=arguments.operational_directory,
            host=arguments.host,
            port=arguments.port,
            device=arguments.device,
            **({"revision_bundle": arguments.revision_bundle} if arguments.revision_bundle else {}),
        )
        return 0

    from .workflow import NetGrowthWorkflow

    workflow = NetGrowthWorkflow.from_paths(
        config_path=arguments.config,
        data_directory="computed-data/dataset",
        output_directory="computed-data/evidence",
        device=getattr(arguments, "device", "cpu"),
    )
    result = getattr(workflow, arguments.command.replace("-", "_"))()
    print(result.summary)
    return 0
