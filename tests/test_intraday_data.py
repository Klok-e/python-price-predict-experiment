import numpy as np
import pandas as pd
import pytest

from utils.intraday_data import (
    build_intraday_datasets,
    load_futures_metrics_intraday_features,
    load_premium_intraday_features,
)


def bars(index, prices):
    return pd.DataFrame(
        {
            "Open time": index,
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": np.ones(len(index)),
        }
    )


def test_futures_metrics_forward_fill_from_native_five_minute_cadence(tmp_path):
    metrics_dir = tmp_path / "futures" / "um" / "daily" / "metrics" / "AAAUSDT"
    metrics_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "create_time": [pd.Timestamp("2025-01-01 00:00"), pd.Timestamp("2025-01-01 00:05")],
            "symbol": ["AAAUSDT", "AAAUSDT"],
            "sum_open_interest": [1.0, 2.0],
        }
    ).to_csv(metrics_dir / "AAAUSDT-metrics-2025-01-01.csv", index=False)

    features = load_futures_metrics_intraday_features(str(tmp_path), "AAAUSDT")

    assert features.loc[pd.Timestamp("2025-01-01 00:04"), "futures_sum_open_interest"] == pytest.approx(1.0)
    assert features.loc[pd.Timestamp("2025-01-01 00:05"), "futures_sum_open_interest"] == pytest.approx(2.0)


def test_premium_features_preserve_one_minute_precision(tmp_path):
    premium_dir = tmp_path / "futures" / "um" / "daily" / "premiumIndexKlines" / "AAAUSDT" / "1m"
    premium_dir.mkdir(parents=True)
    index = pd.date_range("2025-01-01", periods=3, freq="min")
    pd.DataFrame(
        {
            "open_time": index.view("int64") // 1_000_000,
            "open": [-0.001, -0.0005, 0.0],
            "high": [-0.0005, 0.0, 0.0005],
            "low": [-0.0015, -0.001, -0.0005],
            "close": [-0.0008, -0.0002, 0.0002],
            "count": [10, 11, 12],
        }
    ).to_csv(premium_dir / "AAAUSDT-1m-2025-01-01.csv", index=False)

    features = load_premium_intraday_features(str(tmp_path), "AAAUSDT")

    assert features.index.tolist() == list(index)
    assert features["premium_close"].tolist() == pytest.approx([-0.0008, -0.0002, 0.0002])


def test_intraday_dataset_adds_market_relative_features_without_future_rows():
    index = pd.date_range("2025-01-01", periods=8, freq="min")
    raw_tickers = [
        (bars(index, np.linspace(100.0, 108.0, len(index))), "AAAUSDT"),
        (bars(index, np.linspace(200.0, 204.0, len(index))), "BBBUSDT"),
    ]

    datasets = build_intraday_datasets(
        raw_tickers,
        start_date="2025-01-01",
        include_futures_metrics=False,
        include_premium_index=False,
        lags=(1, 2),
        windows=(2,),
    )

    features = datasets["AAAUSDT"]["features"]
    assert not features.empty
    assert "minute_of_day_sin" in features.columns
    assert "day_of_week_cos" in features.columns
    assert "market_log_return_1m" in features.columns
    assert "relative_log_return_1m" in features.columns
    assert features.index.max() <= index.max()


def test_intraday_dataset_validates_only_requested_bar_slice():
    early = pd.date_range("2025-01-01", periods=2, freq="min")
    usable = pd.date_range("2025-01-01 00:03", periods=8, freq="min")
    index = early.append(usable)
    raw_tickers = [
        (bars(index, np.linspace(100.0, 110.0, len(index))), "AAAUSDT"),
        (bars(index, np.linspace(200.0, 210.0, len(index))), "BBBUSDT"),
    ]

    datasets = build_intraday_datasets(
        raw_tickers,
        start_date="2025-01-01 00:03",
        include_futures_metrics=False,
        include_premium_index=False,
        lags=(1, 2),
        windows=(2,),
    )

    assert datasets["AAAUSDT"]["bars"].index.min() == pd.Timestamp("2025-01-01 00:03")
    assert not datasets["AAAUSDT"]["features"].empty
