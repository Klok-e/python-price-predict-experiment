from __future__ import annotations

import json

from netgrowth.artifacts import RunIdentity, write_artifacts


def test_run_writes_only_the_minimal_reproducibility_artifact_set(tmp_path) -> None:
    identity = RunIdentity(config_hash="c", code_hash="g", data_hash="d", model_hash="m")

    first = write_artifacts(
        tmp_path,
        identity=identity,
        report={"compounded_net_return": 0.02},
        equity_rows=({"timestamp": "2026-01-01T00:00:00+00:00", "equity": 10_000.0},),
        trade_rows=(),
        model_bytes=b"fitted-policy",
    )
    second = write_artifacts(
        tmp_path,
        identity=identity,
        report={"compounded_net_return": 0.02},
        equity_rows=({"timestamp": "2026-01-01T00:00:00+00:00", "equity": 10_000.0},),
        trade_rows=(),
        model_bytes=b"fitted-policy",
    )

    assert first == second
    assert {path.name for path in first.iterdir()} == {
        "manifest.json",
        "report.json",
        "equity.csv",
        "trades.csv",
        "model.pt",
    }
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["identity"] == {
        "code_hash": "g",
        "config_hash": "c",
        "data_hash": "d",
        "model_hash": "m",
    }
    assert manifest["result_hash"]
