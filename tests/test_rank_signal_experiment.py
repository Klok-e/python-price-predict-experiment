from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from run_rank_signal_experiment import (
    _fixed_market_regime_choice,
    _holdout_objective,
    _rolling_summary,
    _score_split,
    _success_gate,
    _train_frame,
)
from utils.rank_data import build_rank_datasets, load_premium_index_features


def make_bars(prices, index=None):
    if index is None:
        index = pd.date_range("2026-01-01", periods=len(prices), freq="h")
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": [1.0] * len(prices),
        },
        index=index,
    )


def make_dataset(prices, feature_offset=0.0, index=None):
    bars = make_bars(prices, index=index)
    features = pd.DataFrame(
        {"feature": [float(value) + feature_offset for value in range(len(prices))]},
        index=bars.index,
    )
    labels = pd.Series(np.resize([0, 1], len(prices)), index=bars.index)
    future_return = pd.Series([0.0] * len(prices), index=bars.index)
    return {
        "bars": bars,
        "features": features,
        "labels": labels,
        "future_return": future_return,
    }


def args(**overrides):
    values = {
        "tickers": "AAA,BBB",
        "selection_mode": "market-regime",
        "long_count_grid": "1",
        "short_count_grid": "0",
        "rebalance_bars_grid": "1",
        "long_threshold_grid": "0.0",
        "short_threshold_grid": "-1.0",
        "long_leverage_grid": "1.0",
        "short_leverage_grid": "0.0",
        "min_validation_trades": 1,
        "min_evaluation_trades": 1,
        "cash": 1000.0,
        "commission": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_rank_train_frame_excludes_validation_start_boundary():
    datasets = {
        "AAA": make_dataset([100.0, 101.0, 102.0, 103.0, 104.0]),
        "BBB": make_dataset([200.0, 201.0, 202.0, 203.0, 204.0], feature_offset=10.0),
    }
    validation_start = pd.Timestamp("2026-01-01 02:00")

    x_train, y_train = _train_frame(datasets, ["AAA", "BBB"], validation_start)

    assert len(x_train) == 4
    assert len(y_train) == 4
    assert x_train.index.max() < validation_start
    assert set(x_train["ticker_id"]) == {0, 1}


def test_premium_index_features_resample_negative_values_without_log_math(tmp_path):
    premium_dir = tmp_path / "futures" / "um" / "daily" / "premiumIndexKlines" / "AAAUSDT" / "1m"
    premium_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "open_time": [1767225600000, 1767225660000, 1767225720000, 1767225780000],
            "open": [-0.0010, -0.0008, -0.0005, -0.0002],
            "high": [-0.0007, -0.0004, -0.0001, 0.0001],
            "low": [-0.0012, -0.0010, -0.0007, -0.0004],
            "close": [-0.0008, -0.0005, -0.0002, 0.0],
            "volume": [0, 0, 0, 0],
            "close_time": [1767225659999, 1767225719999, 1767225779999, 1767225839999],
            "quote_volume": [0, 0, 0, 0],
            "count": [10, 11, 12, 13],
            "taker_buy_volume": [0, 0, 0, 0],
            "taker_buy_quote_volume": [0, 0, 0, 0],
            "ignore": [0, 0, 0, 0],
        }
    ).to_csv(premium_dir / "AAAUSDT-1m-2026-01-01.csv", index=False)

    features = load_premium_index_features(str(tmp_path), "AAAUSDT", "2min")

    assert features["premium_close"].tolist() == pytest.approx([-0.0005, 0.0])
    assert features["premium_count"].tolist() == [21.0, 25.0]
    assert "premium_close_diff_1" in features.columns
    assert np.isfinite(features[["premium_close", "premium_range", "premium_position_in_range"]]).all().all()


def test_rank_dataset_adds_market_relative_futures_and_premium_features(tmp_path):
    index = pd.date_range("2026-01-01", periods=120, freq="h")
    raw_tickers = [
        (make_bars(np.linspace(100.0, 140.0, len(index)), index=index).reset_index(names="Open time"), "AAAUSDT"),
        (make_bars(np.linspace(90.0, 130.0, len(index)), index=index).reset_index(names="Open time"), "BBBUSDT"),
    ]
    metrics_dir = tmp_path / "futures" / "um" / "daily" / "metrics" / "AAAUSDT"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "create_time": index[:12],
            "symbol": ["AAAUSDT"] * 12,
            "sum_open_interest": np.linspace(1.0, 2.0, 12),
        }
    ).to_csv(metrics_dir / "AAAUSDT-metrics-2026-01-01.csv", index=False)
    premium_dir = tmp_path / "futures" / "um" / "daily" / "premiumIndexKlines" / "AAAUSDT" / "1m"
    premium_dir.mkdir(parents=True)
    minute_index = pd.date_range("2026-01-01", periods=240, freq="min")
    pd.DataFrame(
        {
            "open_time": (minute_index.view("int64") // 1_000_000),
            "open": np.linspace(-0.001, 0.001, len(minute_index)),
            "high": np.linspace(-0.0005, 0.0015, len(minute_index)),
            "low": np.linspace(-0.0015, 0.0005, len(minute_index)),
            "close": np.linspace(-0.001, 0.001, len(minute_index)),
            "count": np.ones(len(minute_index)),
        }
    ).to_csv(premium_dir / "AAAUSDT-1m-2026-01-01.csv", index=False)

    datasets = build_rank_datasets(
        raw_tickers,
        "1h",
        prediction_horizon_bars=1,
        data_dir=str(tmp_path),
        include_futures_metrics=True,
        include_premium_index=True,
    )

    features = datasets["AAAUSDT"]["features"]
    assert not features.empty
    assert "market_log_return_1" in features.columns
    assert "relative_log_return_1" in features.columns
    assert "futures_sum_open_interest" in features.columns
    assert "premium_close" in features.columns
    assert datasets["AAAUSDT"]["future_return"].index.difference(features.index).empty
    assert features.index.max() > datasets["AAAUSDT"]["future_return"].index.max()


def test_rank_score_split_uses_end_exclusive_window():
    index = pd.date_range("2026-01-01", periods=5, freq="h")
    datasets = {
        "AAA": make_dataset([100.0, 100.0, 110.0, 121.0, 121.0], index=index),
        "BBB": make_dataset([100.0, 100.0, 90.0, 81.0, 81.0], index=index),
    }
    predictions = pd.DataFrame({"AAA": [1.0, 1.0, 1.0], "BBB": [-1.0, -1.0, -1.0]}, index=index[1:4])
    params = {
        "rebalance_bars": 1,
        "long_count": 1,
        "short_count": 0,
        "long_threshold": 0.0,
        "short_threshold": -1.0,
        "long_leverage": 1.0,
        "short_leverage": 0.0,
    }

    report = _score_split(
        datasets,
        ["AAA", "BBB"],
        predictions,
        params,
        start=index[1],
        end=index[4],
        cash=1000.0,
        commission=0.0,
    )

    assert report["_positions"]["signal_time"].tolist() == [index[1], index[2]]
    assert report["aggregate"]["model"]["trades"] == 1
    assert report["aggregate"]["buy_and_hold"]["trades"] == 2


def test_market_regime_selector_uses_pre_evaluation_market_evidence():
    index = pd.date_range("2026-01-01", periods=160, freq="h")
    datasets = {
        "AAA": make_dataset(np.linspace(100.0, 260.0, len(index)), index=index),
        "BBB": make_dataset(np.linspace(90.0, 250.0, len(index)), index=index),
    }
    predictions = pd.DataFrame({"AAA": 1.0, "BBB": -1.0}, index=index)
    report = {
        "model_family": "hist_gradient_boosting",
        "predictions": predictions,
        "parameter_sweep": {"selected": None},
    }
    market_state = {"validation_market_return": 0.1, "market_momentum_96": 0.2, "market_momentum_144": 0.1}

    selected_report, selected = _fixed_market_regime_choice(
        [report],
        market_state,
        datasets,
        ["AAA", "BBB"],
        validation_start=index[20],
        test_start=index[120],
        args=args(),
    )

    assert selected_report["model_family"] == "hist_gradient_boosting"
    assert selected["model_family"] == "hist_gradient_boosting"
    assert selected["rebalance_bars"] == 6
    assert selected["long_leverage"] == pytest.approx(1.5)


def test_rank_success_gate_requires_all_strict_conditions():
    evaluation = {
        "aggregate": {
            "model_minus_buy_and_hold": 0.01,
            "model_minus_no_trade": 0.01,
            "model": {"trades": 29},
        }
    }

    assert _success_gate(evaluation, min_evaluation_trades=30)["passed"] is False

    evaluation["aggregate"]["model"]["trades"] = 30

    assert _success_gate(evaluation, min_evaluation_trades=30)["passed"] is True


def test_rank_rolling_summary_requires_every_window_to_pass():
    windows = [
        {
            "evaluation": {
                "success_gate": {"passed": True},
                "aggregate": {
                    "model": {"cumulative_return": 0.02, "trades": 30},
                    "buy_and_hold": {"cumulative_return": 0.01},
                    "model_minus_buy_and_hold": 0.01,
                },
            }
        },
        {"evaluation": None},
    ]

    summary = _rolling_summary(windows)

    assert summary["evaluated_window_count"] == 1
    assert summary["passed_window_count"] == 1
    assert summary["passed_all_windows"] is False


def test_holdout_objective_reports_final_window_as_primary_goal():
    window = {
        "split_dates": {
            "validation_start": "2026-01-01T00:00:00",
            "test_start": "2026-04-01T00:00:00",
            "max_end": "2026-06-30T00:00:00",
        },
        "evaluation": {
            "aggregate": {
                "model": {"cumulative_return": 0.12, "trades": 40},
                "buy_and_hold": {"cumulative_return": 0.05},
                "model_minus_buy_and_hold": 0.07,
                "model_minus_no_trade": 0.12,
            },
            "success_gate": {
                "beats_buy_and_hold": True,
                "beats_no_trade": True,
                "has_minimum_evaluation_trades": True,
                "passed": True,
            },
        },
    }

    objective = _holdout_objective(window, test_days=90)

    assert objective["objective"] == "beat_buy_and_hold_on_final_holdout"
    assert objective["holdout_days"] == 90
    assert objective["passed"] is True
    assert objective["training_data_ends_before"] == "2026-01-01T00:00:00"
    assert objective["validation_data_ends_before"] == "2026-04-01T00:00:00"
    assert objective["evaluation_data_starts_at"] == "2026-04-01T00:00:00"
