from types import SimpleNamespace

import numpy as np
import pandas as pd

import run_paper_forward
from utils.paper_ledger import (
    append_ledger_records,
    idempotency_key,
    paper_policy_id,
    read_ledger,
    reduce_paper_state,
)
from utils.paper_replay import PaperReplayConfig


def config(**overrides):
    values = {
        "tickers": ("AAA",),
        "start_date": "2026-01-01",
        "end_date": None,
        "bar_size": "1h",
        "prediction_horizon_bars": 1,
        "validation_days": 2,
        "min_train_days": 3,
        "selection_cadence_days": 2,
        "rebalance_cadence_bars": 1,
        "selector_policy": "full-validation",
        "model_family": "ridge",
        "include_futures_metrics": False,
        "include_premium_index": False,
        "long_count_grid": "1",
        "short_count_grid": "0",
        "rebalance_bars_grid": "1",
        "long_threshold_grid": "0.0",
        "short_threshold_grid": "-1.0",
        "long_leverage_grid": "1.0",
        "short_leverage_grid": "0.0",
        "min_validation_trades": 1,
        "min_replay_trades": 1,
        "commission": 0.001,
        "cash": 1.0,
        "seed": 42,
    }
    values.update(overrides)
    return PaperReplayConfig(**values)


def args(tmp_path, **overrides):
    values = {
        "decision_time": "2026-01-01 01:00",
        "force_reselect": False,
        "force_reselect_reason": None,
        "max_staleness_hours": 24.0,
        "max_feature_staleness_hours": 48.0,
        "computed_data_dir": str(tmp_path),
        "data_dir": str(tmp_path),
        "refresh_data": False,
        "refresh_lookback_days": 14,
        "slippage": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def raw_minutes():
    index = pd.date_range("2026-01-01", periods=180, freq="min")
    raw = pd.DataFrame(
        {
            "Open": np.full(len(index), 100.0),
            "High": np.full(len(index), 101.0),
            "Low": np.full(len(index), 99.0),
            "Close": np.full(len(index), 100.0),
        },
        index=index,
    )
    return [(raw, "AAA")]


def dataset(prices, feature_index=None):
    index = pd.date_range("2026-01-01", periods=len(prices), freq="h")
    if feature_index is None:
        feature_index = index
    bars = pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": np.ones(len(prices)),
        },
        index=index,
    )
    features = pd.DataFrame({"feature": np.arange(len(feature_index), dtype=float)}, index=feature_index)
    return {"AAA": {"bars": bars, "features": features, "future_return": pd.Series(0.0, index=feature_index)}}


class StaticModel:
    def predict(self, frame):
        return np.ones(len(frame), dtype=float)


def fake_strategy(decision_time):
    return {
        "decision_index": 0,
        "decision_timestamp": pd.Timestamp(decision_time),
        "training_period": {"start": pd.Timestamp("2025-12-27"), "end": pd.Timestamp("2025-12-30")},
        "validation_period": {"start": pd.Timestamp("2025-12-30"), "end": pd.Timestamp(decision_time)},
        "selected_model_family": "ridge",
        "allocation_parameters": {
            "rebalance_bars": 1,
            "long_count": 1,
            "short_count": 0,
            "long_threshold": 0.0,
            "short_threshold": -1.0,
            "long_leverage": 1.0,
            "short_leverage": 0.0,
        },
        "selector_policy": "full-validation",
        "selection_rule": "test",
        "feature_flags": {"include_futures_metrics": False, "include_premium_index": False},
        "selected_tickers": ("AAA",),
        "model": StaticModel(),
        "artifact_paths": {},
        "gate_outcome": {"passed": True, "reason": "selected"},
    }


def test_ledger_append_is_idempotent_and_reducer_reports_online_state(tmp_path):
    path = tmp_path / "paper_ledger.jsonl"
    key = idempotency_key("policy", "paper_forward_online", "2026-01-01", suffix="fill")
    records = [
        {
            "record_type": "online_paper_fill",
            "policy_id": "policy",
            "evidence_mode": "paper_forward_online",
            "idempotency_key": key,
            "signal_time": "2026-01-01T00:00:00",
            "fill_time": "2026-01-01T01:00:00",
            "cash": -0.001,
            "equity": 0.999,
            "turnover": 1.0,
            "fee": 0.001,
            "fill_prices": {"AAA": 100.0},
            "units": {"AAA": 0.01},
            "weights": {"AAA": 1.001001001001001},
        }
    ]

    assert append_ledger_records(str(path), records) == 1
    assert append_ledger_records(str(path), records) == 0

    loaded = read_ledger(str(path))
    state = reduce_paper_state(loaded, ("AAA",), 1.0)
    assert len(loaded) == 1
    assert state["cash"] == -0.001
    assert state["equity"] == 0.999
    assert state["units"] == {"AAA": 0.01}
    assert state["accumulated_fees"] == 0.001


def test_forward_dry_run_creates_pending_order_without_writing(tmp_path, monkeypatch):
    cfg = config()
    monkeypatch.setattr(run_paper_forward, "select_replay_strategy", lambda *call_args, **kwargs: fake_strategy(call_args[3]))

    result = run_paper_forward.build_forward_decision(
        args(tmp_path),
        cfg,
        raw_minutes(),
        dataset([100.0, 101.0, 102.0]),
        [],
        "policy",
        "paper_forward_dry_run",
        persist_artifacts=False,
    )

    assert result["evidence_mode"] == "paper_forward_dry_run"
    assert [record["record_type"] for record in result["records"]][-3:] == [
        "prediction",
        "target_weights",
        "pending_paper_order",
    ]
    assert result["records"][-1]["fill_time"] == "2026-01-01T01:00:00"
    assert not (tmp_path / "runs").exists()


def test_pending_order_fills_on_later_cycle_and_marks_state(tmp_path, monkeypatch):
    cfg = config()
    first_key = idempotency_key("policy", "paper_forward_online", "2026-01-01 00:00")
    records = [
        {
            "record_type": "pending_paper_order",
            "policy_id": "policy",
            "evidence_mode": "paper_forward_online",
            "idempotency_key": first_key,
            "pending_order_key": first_key,
            "signal_time": "2026-01-01T00:00:00",
            "fill_time": "2026-01-01T01:00:00",
            "target_weights": {"AAA": 1.0},
            "previous_weights": {"AAA": 0.0},
        }
    ]
    monkeypatch.setattr(run_paper_forward, "select_replay_strategy", lambda *call_args, **kwargs: fake_strategy(call_args[3]))

    result = run_paper_forward.build_forward_decision(
        args(tmp_path, decision_time="2026-01-01 03:00"),
        cfg,
        raw_minutes(),
        dataset([100.0, 100.0, 110.0]),
        records,
        "policy",
        "paper_forward_online",
        persist_artifacts=False,
    )

    record_types = [record["record_type"] for record in result["records"]]
    assert "online_paper_fill" in record_types
    assert "mark_to_market" in record_types
    state = result["state"]
    assert state["units"]["AAA"] == 0.01
    assert state["cash"] == -0.001
    assert state["equity"] > 1.0
    assert state["pending_order_count"] == 1


def test_no_new_closed_bar_appends_nothing(tmp_path):
    result = run_paper_forward.build_forward_decision(
        args(tmp_path, decision_time="2026-01-01 00:30"),
        config(),
        raw_minutes(),
        dataset([100.0]),
        [],
        "policy",
        "paper_forward_online",
        persist_artifacts=False,
    )

    assert result["records"] == []
    assert result["no_action"]["reason"] == "no_closed_signal_bar"


def test_forward_stale_data_blocks_before_selection(tmp_path, monkeypatch):
    cfg = config()
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(run_paper_forward, "select_replay_strategy", fail_if_called)

    result = run_paper_forward.build_forward_decision(
        args(tmp_path, decision_time="2026-01-03 02:00", max_staleness_hours=1.0),
        cfg,
        raw_minutes(),
        dataset([100.0, 100.0]),
        [],
        "policy",
        "paper_forward_online",
        persist_artifacts=False,
    )

    assert result["no_action"]["reason"] == "stale_data"
    assert called is False


def test_required_feature_staleness_blocks_when_enabled(tmp_path, monkeypatch):
    cfg = config(include_futures_metrics=True)
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(run_paper_forward, "select_replay_strategy", fail_if_called)

    result = run_paper_forward.build_forward_decision(
        args(tmp_path),
        cfg,
        raw_minutes(),
        dataset([100.0, 100.0, 101.0]),
        [],
        "policy",
        "paper_forward_online",
        persist_artifacts=False,
    )

    assert result["no_action"]["reason"] == "stale_feature_data"
    assert called is False


def test_forward_missing_model_artifact_records_no_action(tmp_path, monkeypatch):
    cfg = config()
    selection_key = idempotency_key("policy", "paper_forward_online", "2026-01-01 00:00")
    records = [
        {
            "record_type": "strategy_selection",
            "policy_id": "policy",
            "evidence_mode": "paper_forward_online",
            "idempotency_key": selection_key,
            "selection_time": "2026-01-01T00:00:00",
            "signal_time": "2026-01-01T00:00:00",
            "selected_model_family": "ridge",
            "allocation_parameters": fake_strategy("2026-01-01")["allocation_parameters"],
            "artifact_paths": {"model_ridge": "/tmp/missing-model.pkl"},
        }
    ]
    monkeypatch.setattr(run_paper_forward, "select_replay_strategy", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("reselected too early")))

    result = run_paper_forward.build_forward_decision(
        args(tmp_path, decision_time="2026-01-01 02:00"),
        cfg,
        raw_minutes(),
        dataset([100.0, 100.0, 101.0]),
        records,
        "policy",
        "paper_forward_online",
        persist_artifacts=False,
    )

    assert result["records"][-1]["record_type"] == "no_action"
    assert result["records"][-1]["reason"] == "no_selected_model"


def test_target_unchanged_records_no_action_without_pending_order(tmp_path, monkeypatch):
    cfg = config()
    monkeypatch.setattr(run_paper_forward, "select_replay_strategy", lambda *call_args, **kwargs: fake_strategy(call_args[3]))
    records = [
        {
            "record_type": "online_paper_fill",
            "policy_id": "policy",
            "evidence_mode": "paper_forward_online",
            "idempotency_key": idempotency_key("policy", "paper_forward_online", "2026-01-01 00:00", suffix="fill"),
            "signal_time": "2026-01-01T00:00:00",
            "fill_time": "2026-01-01T01:00:00",
            "cash": 0.0,
            "equity": 1.0,
            "turnover": 1.0,
            "fee": 0.0,
            "fill_prices": {"AAA": 100.0},
            "units": {"AAA": 0.01},
            "weights": {"AAA": 1.0},
        }
    ]

    result = run_paper_forward.build_forward_decision(
        args(tmp_path),
        cfg,
        raw_minutes(),
        dataset([100.0, 100.0, 100.0]),
        records,
        "policy",
        "paper_forward_online",
        persist_artifacts=False,
    )

    record_types = [record["record_type"] for record in result["records"]]
    assert "target_weights" in record_types
    assert "pending_paper_order" not in record_types
    assert result["records"][-1]["reason"] == "target_unchanged"


def test_append_duplicate_cycle_writes_zero_records(tmp_path, monkeypatch):
    cfg = config()
    monkeypatch.setattr(run_paper_forward, "select_replay_strategy", lambda *call_args, **kwargs: fake_strategy(call_args[3]))
    policy_id = paper_policy_id(cfg)
    path = tmp_path / "runs" / policy_id / "paper_ledger.jsonl"

    first = run_paper_forward.build_forward_decision(
        args(tmp_path),
        cfg,
        raw_minutes(),
        dataset([100.0, 101.0, 102.0]),
        [],
        policy_id,
        "paper_forward_online",
        persist_artifacts=False,
    )
    assert append_ledger_records(str(path), first["records"]) > 0
    records = read_ledger(str(path))
    second = run_paper_forward.build_forward_decision(
        args(tmp_path),
        cfg,
        raw_minutes(),
        dataset([100.0, 101.0, 102.0]),
        records,
        policy_id,
        "paper_forward_online",
        persist_artifacts=False,
    )
    assert append_ledger_records(str(path), second["records"]) == 0

    later = run_paper_forward.build_forward_decision(
        args(tmp_path, decision_time="2026-01-01 02:00"),
        cfg,
        raw_minutes(),
        dataset([100.0, 101.0, 102.0]),
        records,
        policy_id,
        "paper_forward_online",
        persist_artifacts=False,
    )
    assert any(record["record_type"] == "online_paper_fill" for record in later["records"])


def test_daemon_max_cycles_exits(tmp_path, monkeypatch, capsys):
    cfg = config()
    policy_id = paper_policy_id(cfg)
    path = tmp_path / "runs" / policy_id / "paper_ledger.jsonl"
    monkeypatch.setattr(run_paper_forward, "build_config", lambda _args: cfg)
    monkeypatch.setattr(run_paper_forward, "_run_one_cycle", lambda *_args, **_kwargs: {
        "policy_id": policy_id,
        "signal_time": None,
        "appended_records": 0,
        "no_action": None,
        "state": reduce_paper_state(read_ledger(str(path)), cfg.tickers, cfg.cash),
    })

    run_paper_forward.main([
        "--daemon",
        "--computed-data-dir",
        str(tmp_path),
        "--poll-seconds",
        "0",
        "--max-cycles",
        "2",
        "--no-include-futures-metrics",
        "--no-include-premium-index",
    ])

    assert len(capsys.readouterr().out.strip().splitlines()) == 2
