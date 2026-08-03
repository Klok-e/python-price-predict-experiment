from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from utils.util import open_time_to_datetime, validate_ohlc_bars


DEFAULT_INTRADAY_TICKERS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
DEFAULT_INTRADAY_START_DATE = "2025-01-01"
INTRADAY_LAGS = (1, 2, 3, 5, 10, 15, 30, 60, 120, 240, 480, 720, 1440)
INTRADAY_WINDOWS = (5, 15, 30, 60, 120, 240, 720, 1440)


def raw_to_1m_bars(
    raw_df: pd.DataFrame,
    ticker: str,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    df = raw_df.copy()
    if "Open time" in df.columns:
        df.index = open_time_to_datetime(df["Open time"])
    bars = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()
    if start_date is not None:
        bars = bars.loc[bars.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        bars = bars.loc[bars.index <= pd.Timestamp(end_date)]
    return validate_ohlc_bars(bars, ticker_name=ticker, frequency="1min")


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    return (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-12)


def spot_intraday_features(
    bars: pd.DataFrame,
    lags: tuple[int, ...] = INTRADAY_LAGS,
    windows: tuple[int, ...] = INTRADAY_WINDOWS,
) -> pd.DataFrame:
    close = bars["Close"]
    high = bars["High"]
    low = bars["Low"]
    volume = bars["Volume"]
    one_min_return = np.log(close / close.shift(1))
    features = pd.DataFrame(index=bars.index)
    minute_of_day = features.index.hour * 60 + features.index.minute
    day_of_week = features.index.dayofweek
    features["minute_of_day_sin"] = np.sin(2.0 * np.pi * minute_of_day / 1440.0)
    features["minute_of_day_cos"] = np.cos(2.0 * np.pi * minute_of_day / 1440.0)
    features["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    features["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0)

    for lag in lags:
        features[f"log_return_{lag}m"] = np.log(close / close.shift(lag))
        features[f"volume_change_{lag}m"] = volume.pct_change(lag)

    for window in windows:
        rolling_close = close.rolling(window)
        rolling_volume = volume.rolling(window)
        features[f"volatility_{window}m"] = one_min_return.rolling(window).std()
        features[f"sma_distance_{window}m"] = np.log(close / rolling_close.mean())
        features[f"range_position_{window}m"] = (
            (close - rolling_close.min()) / (rolling_close.max() - rolling_close.min() + 1e-12)
        )
        features[f"volume_zscore_{window}m"] = (volume - rolling_volume.mean()) / (
            rolling_volume.std() + 1e-12
        )
        features[f"hl_range_{window}m"] = ((high - low) / close).rolling(window).mean()

    return features.replace([np.inf, -np.inf], np.nan)


def load_futures_metrics_intraday_features(data_dir: str, ticker: str) -> pd.DataFrame:
    paths = sorted(glob.glob(f"{data_dir}/futures/um/daily/metrics/{ticker}/{ticker}-metrics-*.csv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "create_time" not in frame.columns:
            continue
        frame["create_time"] = pd.to_datetime(frame["create_time"], errors="coerce")
        frame = frame.dropna(subset=["create_time"]).set_index("create_time").sort_index()
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    metrics = pd.concat(frames).sort_index()
    metrics = metrics[~metrics.index.duplicated(keep="last")]
    numeric = metrics.drop(columns=["symbol"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    one_minute = numeric.resample("1min").mean().ffill()
    features = pd.DataFrame(index=one_minute.index)
    for column in one_minute.columns:
        series = one_minute[column].astype(float)
        features[f"futures_{column}"] = series
        features[f"futures_{column}_change_5m"] = series.pct_change(5)
        for window in (15, 60, 240):
            features[f"futures_{column}_zscore_{window}m"] = _rolling_zscore(series, window)
    return features.replace([np.inf, -np.inf], np.nan)


def load_premium_intraday_features(data_dir: str, ticker: str) -> pd.DataFrame:
    paths = sorted(
        glob.glob(f"{data_dir}/futures/um/daily/premiumIndexKlines/{ticker}/1m/{ticker}-1m-*.csv")
    )
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "open_time" not in frame.columns:
            continue
        frame["open_time"] = open_time_to_datetime(frame["open_time"])
        frame = frame.dropna(subset=["open_time"]).set_index("open_time").sort_index()
        frames.append(frame)
    if not frames:
        return pd.DataFrame()

    premium = pd.concat(frames).sort_index()
    premium = premium[~premium.index.duplicated(keep="last")]
    numeric = premium[["open", "high", "low", "close", "count"]].apply(pd.to_numeric, errors="coerce")
    one_minute = (
        numeric.resample("1min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "count": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
    )
    close = one_minute["close"].astype(float)
    range_width = (one_minute["high"] - one_minute["low"]).astype(float)
    features = pd.DataFrame(index=one_minute.index)
    features["premium_close"] = close
    features["premium_range"] = range_width
    features["premium_count"] = one_minute["count"].astype(float)
    features["premium_position_in_range"] = (close - one_minute["low"]) / (range_width + 1e-12)
    for lag in (1, 5, 15, 60):
        features[f"premium_close_diff_{lag}m"] = close - close.shift(lag)
        features[f"premium_range_diff_{lag}m"] = range_width - range_width.shift(lag)
    for window in (15, 60, 240):
        features[f"premium_close_zscore_{window}m"] = _rolling_zscore(close, window)
        features[f"premium_range_zscore_{window}m"] = _rolling_zscore(range_width, window)
    return features.replace([np.inf, -np.inf], np.nan)


def build_intraday_datasets(
    raw_tickers,
    data_dir: str = "computed-data/dataset",
    start_date: str = DEFAULT_INTRADAY_START_DATE,
    end_date: str | None = None,
    include_futures_metrics: bool = True,
    include_premium_index: bool = True,
    lags: tuple[int, ...] = INTRADAY_LAGS,
    windows: tuple[int, ...] = INTRADAY_WINDOWS,
):
    datasets = {}
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else None
    for raw_df, ticker in raw_tickers:
        bars = raw_to_1m_bars(raw_df, ticker, start_date=start, end_date=end)
        features = spot_intraday_features(bars, lags=lags, windows=windows)
        if include_futures_metrics:
            futures = load_futures_metrics_intraday_features(data_dir, ticker)
            if not futures.empty:
                features = features.join(futures, how="left")
                features.loc[:, futures.columns] = features.loc[:, futures.columns].ffill()
        if include_premium_index:
            premium = load_premium_intraday_features(data_dir, ticker)
            if not premium.empty:
                features = features.join(premium, how="left")
                features.loc[:, premium.columns] = features.loc[:, premium.columns].ffill()
        datasets[ticker] = {"bars": bars, "features": features}

    close_frame = pd.DataFrame({ticker: data["bars"]["Close"] for ticker, data in datasets.items()})
    market_return = np.log(close_frame / close_frame.shift(1)).mean(axis=1)
    market_features = pd.DataFrame(index=close_frame.index)
    for lag in lags:
        market_features[f"market_log_return_{lag}m"] = market_return.rolling(lag).sum()
    for window in windows:
        market_features[f"market_volatility_{window}m"] = market_return.rolling(window).std()

    for ticker, data in datasets.items():
        ticker_return = np.log(data["bars"]["Close"] / data["bars"]["Close"].shift(1))
        relative = pd.DataFrame(index=data["features"].index)
        for lag in lags:
            relative[f"relative_log_return_{lag}m"] = (
                ticker_return.rolling(lag).sum() - market_return.rolling(lag).sum()
            )
        features = data["features"].join(market_features, how="left").join(relative, how="left")
        data["features"] = features.replace([np.inf, -np.inf], np.nan).dropna()

    return datasets
