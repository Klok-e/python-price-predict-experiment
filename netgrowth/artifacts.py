"""Bounded, content-addressed reproducibility artifacts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


@dataclass(frozen=True)
class RunIdentity:
    config_hash: str
    code_hash: str
    data_hash: str
    model_hash: str

    @property
    def run_id(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()[:16]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_csv(path: Path, rows: tuple[dict[str, Any], ...], default_fields: tuple[str, ...]) -> None:
    fields = tuple(rows[0]) if rows else default_fields
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_artifacts(
    root: str | Path,
    *,
    identity: RunIdentity,
    report: dict[str, Any],
    equity_rows: tuple[dict[str, Any], ...],
    trade_rows: tuple[dict[str, Any], ...],
    model_bytes: bytes,
) -> Path:
    artifact_root = Path(root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_directory = artifact_root / identity.run_id
    expected_files = {"manifest.json", "report.json", "equity.csv", "trades.csv", "model.pt"}
    if run_directory.exists():
        if {path.name for path in run_directory.iterdir()} != expected_files:
            raise RuntimeError(f"incomplete immutable artifact exists at {run_directory}")
        return run_directory

    report_bytes = _canonical(report)
    result_digest = sha256()
    for payload in (report_bytes, _canonical(equity_rows), _canonical(trade_rows), model_bytes):
        result_digest.update(payload)
    manifest = {"identity": asdict(identity), "result_hash": result_digest.hexdigest()}

    with TemporaryDirectory(prefix=f".{identity.run_id}-", dir=artifact_root) as temporary:
        staging = Path(temporary)
        (staging / "report.json").write_bytes(report_bytes + b"\n")
        _write_csv(
            staging / "equity.csv",
            equity_rows,
            ("timestamp", "equity", "drawdown", "gross_exposure"),
        )
        _write_csv(
            staging / "trades.csv",
            trade_rows,
            ("timestamp", "ticker", "quantity", "reference_price", "effective_fill", "target_weight"),
        )
        (staging / "model.pt").write_bytes(model_bytes)
        (staging / "manifest.json").write_bytes(_canonical(manifest) + b"\n")
        staging.replace(run_directory)
    return run_directory
