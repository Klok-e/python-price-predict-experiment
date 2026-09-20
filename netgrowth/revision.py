"""Offline preparation of a validated, immutable eligibility-revision candidate."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from math import prod
from pathlib import Path
from typing import Any

import pandas as pd

from .binance import HistoricalArchiveAdapter
from .config import load_config
from .torch_backend import TorchEvaluationBackend, _model_contract, _outcome
from .training import walk_forward_folds


def prepare_revision(
    *, config_path: str, data_directory: str, checkpoint: str, evidence_state: str, output_directory: str, device: str
) -> Path:
    config = load_config(config_path)
    prior = json.loads(Path(evidence_state).read_text())
    if config.development_evidence_end > config.holdout_start and not prior.get("holdout"):
        raise ValueError("development period requires recorded consumed holdout evidence")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    incumbent = Path(checkpoint).read_bytes()
    backend = TorchEvaluationBackend(output / "cache")
    selected = backend._selected_metadata(incumbent)
    contract = _model_contract(selected, config)
    canonical = HistoricalArchiveAdapter(Path(data_directory), config.tickers).load()
    prepared = backend._prepared(canonical)
    folds = walk_forward_folds(
        common_start=canonical.common_trading_start,
        holdout_start=pd.Timestamp(config.development_evidence_end, tz="UTC"),
        fold_count=config.validation_folds,
        fold_days=config.fold_days,
        purge_days=max(config.receptive_field_days),
    )
    data_id = canonical.identity_hash
    rows: list[dict[str, Any]] = []
    for index, fold in enumerate(folds):
        cache_key = sha256(
            f"{config.protocol_id}:{data_id}:{sha256(incumbent).hexdigest()}:{index}".encode()
        ).hexdigest()
        fold_path = output / f"fold-{index + 1:02d}-{cache_key[:12]}.json"
        if fold_path.exists():
            rows.append(json.loads(fold_path.read_text()))
            continue
        period = backend._evaluate_period(
            canonical,
            config,
            device,
            fold.validation_start,
            fold.validation_end,
            contract.factory,
            seeds=config.seeds[: contract.member_count],
            receptive_bars=contract.receptive_bars,
            initial_training_end=fold.training_end,
            prepared=prepared,
        )
        replay = period.ensemble_replay
        outcome = _outcome(replay, b"")
        row = {
            "fold": index + 1,
            "start": fold.validation_start.isoformat(),
            "end": fold.validation_end.isoformat(),
            "net_return": replay.compounded_net_return,
            "max_drawdown": replay.max_drawdown,
            "executable_changes": replay.executable_portfolio_changes,
            "equity": outcome.equity_rows,
            "trades": outcome.trade_rows,
        }
        temporary = fold_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(row, sort_keys=True))
        temporary.replace(fold_path)
        rows.append(row)
        print(
            f"Fold {index + 1}/{len(folds)}: return {replay.compounded_net_return:.4%}, "
            f"drawdown {replay.max_drawdown:.4%}",
            flush=True,
        )
    report = {
        "protocol_id": config.protocol_id,
        "incumbent_model_id": sha256(incumbent).hexdigest(),
        "data_id": data_id,
        "architecture": {"kind": contract.kind, **contract.metadata},
        "folds": [{k: v for k, v in row.items() if k not in {"equity", "trades"}} for row in rows],
        "compounded_net_return": prod(1 + row["net_return"] for row in rows) - 1,
        "maximum_drawdown": max(row["max_drawdown"] for row in rows),
    }
    report_path = output / "validation.json"
    report_path.write_text(json.dumps(report, sort_keys=True, indent=2))
    if report["compounded_net_return"] <= 0 or report["maximum_drawdown"] > config.drawdown_limit:
        raise ValueError(f"revision failed validation; evidence retained at {report_path}")
    observed = min(data.perpetual.index.max() for data in canonical.instruments.values()) + pd.Timedelta(minutes=1)
    final = backend.paper(
        canonical,
        config,
        device,
        validated_model=incumbent,
        fitted_model=None,
        observed_at=observed.to_pydatetime(),
        current_weights=dict.fromkeys(config.tickers, 0.0),
    )
    model_id = sha256(final.model_bytes).hexdigest()
    model_path = output / f"{model_id}.pt"
    model_path.write_bytes(final.model_bytes)
    bundle = {
        **report,
        "trained_protocol_id": config.protocol_id,
        "model_id": model_id,
        "fitted_at": datetime.now(UTC).isoformat(),
        "training_observed_at": observed.isoformat(),
        "checkpoint": str(model_path),
        "compatibility": config.compatibility_manifest,
    }
    destination = output / "revision.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(bundle, sort_keys=True, indent=2))
    temporary.replace(destination)
    return destination
