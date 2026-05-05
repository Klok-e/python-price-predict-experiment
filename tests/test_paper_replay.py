import numpy as np
import pandas as pd
import pytest

from run_rank_signal_experiment import _market_state
from utils.paper_replay import (
    PaperReplayConfig,
    _train_frame_before_decision,
    buy_and_hold_baseline,
    build_replay_report,
    decision_timestamps,
    guarded_market_regime_params,
    simulate_replay_positions,
    validate_replay_bar_series,
)


def config(**overrides):
    values = {
        "tickers": ("AAA",),
        "start_date": "2026-01-01",
        "end_date": "2026-01-10",
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


def make_dataset(prices, index=None):
    if index is None:
        index = pd.date_range("2026-01-01", periods=len(prices), freq="h")
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
    features = pd.DataFrame({"feature": np.arange(len(prices), dtype=float)}, index=index)
    return {"bars": bars, "features": features, "future_return": pd.Series(0.0, index=index)}


class StaticRankModel:
    def __init__(self, value):
        self.value = value
        self.seen_frames = []

    def predict(self, frame):
        self.seen_frames.append(frame.copy())
        return np.full(len(frame), self.value, dtype=float)


def test_decision_timestamps_respect_warmup_and_selection_cadence():
    index = pd.date_range("2026-01-01", periods=10, freq="D")

    timestamps = decision_timestamps(index, config())

    assert timestamps == [
        pd.Timestamp("2026-01-06"),
        pd.Timestamp("2026-01-08"),
    ]


def test_training_frame_excludes_validation_start_boundary():
    index = pd.date_range("2026-01-01", periods=5, freq="h")
    datasets = {"AAA": make_dataset([100, 101, 102, 103, 104], index=index)}

    x_train, y_train = _train_frame_before_decision(datasets, ["AAA"], pd.Timestamp("2026-01-01 03:00"))

    assert list(x_train.index) == list(index[:3])
    assert list(y_train.index) == list(index[:3])


def test_replay_fills_next_open_and_charges_turnover_fee():
    index = pd.date_range("2026-01-01", periods=3, freq="h")
    datasets = {"AAA": make_dataset([100.0, 100.0, 110.0], index=index)}
    replay_config = config(end_date="2026-01-01 02:00", cash=1.0, commission=0.001)
    decisions = [
        {
            "decision_index": 0,
            "decision_timestamp": index[0],
            "allocation_parameters": {
                "rebalance_bars": 1,
                "long_count": 1,
                "short_count": 0,
                "long_threshold": 0.0,
                "short_threshold": -1.0,
                "long_leverage": 1.0,
                "short_leverage": 0.0,
            },
            "model": StaticRankModel(1.0),
        }
    ]

    result = simulate_replay_positions(datasets, ["AAA"], decisions, replay_config, index)

    first = result["positions"].iloc[0]
    assert first["entry_time"] == index[1]
    assert first["exit_time"] == index[2]
    assert first["turnover"] == pytest.approx(1.0)
    assert first["turnover_cost"] == pytest.approx(0.001)
    assert result["portfolio"]["cumulative_return"] == pytest.approx(0.099)


def test_replay_respects_rebalance_cadence():
    index = pd.date_range("2026-01-01", periods=5, freq="h")
    datasets = {"AAA": make_dataset([100.0, 100.0, 101.0, 102.0, 103.0], index=index)}
    replay_config = config(end_date="2026-01-01 04:00", rebalance_cadence_bars=1, cash=1.0, commission=0.0)
    decisions = [
        {
            "decision_index": 0,
            "decision_timestamp": index[0],
            "allocation_parameters": {
                "rebalance_bars": 2,
                "long_count": 1,
                "short_count": 0,
                "long_threshold": 0.0,
                "short_threshold": -1.0,
                "long_leverage": 1.0,
                "short_leverage": 0.0,
            },
            "model": StaticRankModel(1.0),
        }
    ]

    result = simulate_replay_positions(datasets, ["AAA"], decisions, replay_config, index)

    assert result["positions"]["signal_time"].tolist() == [index[0], index[2]]


def test_replay_uses_selected_rebalance_bars_for_exit_time():
    index = pd.date_range("2026-01-01", periods=5, freq="h")
    datasets = {"AAA": make_dataset([100.0, 100.0, 101.0, 102.0, 104.0], index=index)}
    replay_config = config(end_date="2026-01-01 04:00", rebalance_cadence_bars=1, cash=1.0, commission=0.0)
    decisions = [
        {
            "decision_index": 0,
            "decision_timestamp": index[0],
            "allocation_parameters": {
                "rebalance_bars": 3,
                "long_count": 1,
                "short_count": 0,
                "long_threshold": 0.0,
                "short_threshold": -1.0,
                "long_leverage": 1.0,
                "short_leverage": 0.0,
            },
            "model": StaticRankModel(1.0),
        }
    ]

    result = simulate_replay_positions(datasets, ["AAA"], decisions, replay_config, index)

    first = result["positions"].iloc[0]
    assert first["entry_time"] == index[1]
    assert first["exit_time"] == index[4]
    assert result["portfolio"]["cumulative_return"] == pytest.approx(0.04)


def test_market_regime_inputs_use_validation_period_boundary():
    index = pd.date_range("2026-01-01", periods=160, freq="h")
    prices = np.linspace(100.0, 260.0, len(index))
    datasets = {"AAA": make_dataset(prices, index=index)}

    state = _market_state(
        datasets,
        ["AAA"],
        validation_start=pd.Timestamp("2026-01-02"),
        test_start=pd.Timestamp("2026-01-07"),
    )

    assert state["validation_market_return"] > 0
    assert state["market_momentum_96"] > 0


def test_buy_and_hold_baseline_is_equal_capital_across_tickers():
    index = pd.date_range("2026-01-01", periods=3, freq="h")
    datasets = {
        "AAA": make_dataset([100.0, 100.0, 110.0], index=index),
        "BBB": make_dataset([100.0, 100.0, 90.0], index=index),
    }

    result = buy_and_hold_baseline(datasets, ["AAA", "BBB"], index, cash=1.0, commission=0.0)

    assert result["trades"] == 2
    assert result["cumulative_return"] == pytest.approx(0.0)
    assert result["end_equity"] == pytest.approx(1.0)


def test_replay_report_marks_historical_evidence_and_includes_baselines():
    replay_config = config()
    positions = pd.DataFrame(
        {
            "decision_index": [0],
            "signal_time": [pd.Timestamp("2026-01-06")],
            "turnover": [1.0],
            "equity": [1.1],
        }
    )
    report = build_replay_report(
        replay_config,
        "run123",
        {"AAA": {"bar_rows": 10}},
        [{"decision_index": 0, "model": object(), "gate_outcome": {"passed": True}}],
        {
            "portfolio": {"cumulative_return": 0.1, "trades": 1, "start_cash": 1.0, "end_equity": 1.1},
            "buy_and_hold": {"cumulative_return": 0.05, "trades": 1, "start_cash": 1.0, "end_equity": 1.05},
            "positions": positions,
            "artifact_paths": {"report": "paper_replay_report.json"},
        },
    )

    assert report["report_type"] == "historical_paper_replay_report"
    assert report["evidence_mode"] == "historical_replay"
    assert set(report["replay"]["aggregate"]) == {"model", "buy_and_hold", "no_trade", "model_minus_buy_and_hold", "model_minus_no_trade"}
    assert report["replay"]["summary_by_calendar_month"][0]["month"] == "2026-01"


def test_data_gate_rejects_duplicate_raw_bars_before_scoring():
    index = [pd.Timestamp("2026-01-01 00:00"), pd.Timestamp("2026-01-01 00:00")]
    raw = pd.DataFrame(
        {
            "Open": [100.0, 100.0],
            "High": [100.0, 100.0],
            "Low": [100.0, 100.0],
            "Close": [100.0, 100.0],
        },
        index=index,
    )
    datasets = {"AAA": make_dataset([100.0, 101.0])}

    with pytest.raises(ValueError, match="duplicate bars"):
        validate_replay_bar_series([(raw, "AAA")], datasets, ("AAA",), "1h")


def test_guarded_market_regime_uses_only_pre_decision_evidence():
    selected = {
        "rebalance_bars": 3,
        "long_count": 1,
        "short_count": 0,
        "long_threshold": 0.0,
        "short_threshold": -1.0,
        "long_leverage": 1.0,
        "short_leverage": 0.0,
        "model_minus_buy_and_hold": -0.3,
    }

    params, reason = guarded_market_regime_params(
        {"validation_market_return": 0.7, "market_momentum_96": 0.2},
        selected,
        ticker_count=4,
    )

    assert reason == "strong_bull_broad_long"
    assert params["long_count"] == 4
    assert params["long_leverage"] == pytest.approx(1.5)

    params, reason = guarded_market_regime_params(
        {"validation_market_return": 0.1, "market_momentum_96": 0.2},
        selected,
        ticker_count=4,
    )

    assert reason == "negative_validation_edge_no_trade"
    assert params["long_count"] == 0
    assert params["short_count"] == 0
