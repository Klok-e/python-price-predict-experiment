from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from utils.util import open_time_to_datetime


FEATURE_LAGS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 72, 96)
ROLLING_WINDOWS = (4, 8, 16, 24, 32, 48, 72, 96)


def raw_to_ohlcv(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    if "Open time" in df.columns:
        df.index = open_time_to_datetime(df["Open time"])
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float).sort_index()


def base_features(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["Close"]
    volume = bars["Volume"]
    one_bar_return = np.log(close / close.shift(1))
    features = pd.DataFrame(index=bars.index)

    for lag in FEATURE_LAGS:
        features[f"log_return_{lag}"] = np.log(close / close.shift(lag))

    for window in ROLLING_WINDOWS:
        rolling_close = close.rolling(window)
        rolling_volume = volume.rolling(window)
        features[f"volatility_{window}"] = one_bar_return.rolling(window).std()
        features[f"sma_distance_{window}"] = np.log(close / rolling_close.mean())
        features[f"range_position_{window}"] = (
            (close - rolling_close.min()) / (rolling_close.max() - rolling_close.min() + 1e-12)
        )
        features[f"volume_zscore_{window}"] = (
            (volume - rolling_volume.mean()) / (rolling_volume.std() + 1e-12)
        )

    return features.replace([np.inf, -np.inf], np.nan)


def load_futures_metrics_features(data_dir: str, ticker: str, bar_size: str) -> pd.DataFrame:
    paths = sorted(glob.glob(f"{data_dir}/futures/um/daily/metrics/{ticker}/{ticker}-metrics-*.csv"))
    if not paths:
        return pd.DataFrame()
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
    resampled = numeric.resample(bar_size).mean().ffill()
    features = pd.DataFrame(index=resampled.index)
    for column in resampled.columns:
        series = resampled[column].astype(float)
        features[f"futures_{column}"] = series
        features[f"futures_{column}_change_1"] = series.pct_change(1).replace([np.inf, -np.inf], np.nan)
        for window in (6, 24, 72):
            features[f"futures_{column}_zscore_{window}"] = (
                (series - series.rolling(window).mean()) / (series.rolling(window).std() + 1e-12)
            )
    return features.replace([np.inf, -np.inf], np.nan)


def load_premium_index_features(data_dir: str, ticker: str, bar_size: str) -> pd.DataFrame:
    paths = sorted(
        glob.glob(
            f"{data_dir}/futures/um/daily/premiumIndexKlines/{ticker}/1m/{ticker}-1m-*.csv"
        )
    )
    if not paths:
        return pd.DataFrame()
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
    resampled = (
        numeric.resample(bar_size)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "count": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
    )
    if resampled.empty:
        return pd.DataFrame()

    close = resampled["close"].astype(float)
    high = resampled["high"].astype(float)
    low = resampled["low"].astype(float)
    range_width = high - low
    features = pd.DataFrame(index=resampled.index)
    features["premium_close"] = close
    features["premium_range"] = range_width
    features["premium_count"] = resampled["count"].astype(float)
    features["premium_position_in_range"] = (close - low) / (range_width + 1e-12)

    for lag in FEATURE_LAGS:
        features[f"premium_close_diff_{lag}"] = close - close.shift(lag)
        features[f"premium_range_diff_{lag}"] = range_width - range_width.shift(lag)

    for window in (6, 24, 72):
        close_mean = close.rolling(window).mean()
        close_std = close.rolling(window).std()
        range_mean = range_width.rolling(window).mean()
        range_std = range_width.rolling(window).std()
        features[f"premium_close_zscore_{window}"] = (close - close_mean) / (close_std + 1e-12)
        features[f"premium_range_zscore_{window}"] = (range_width - range_mean) / (range_std + 1e-12)
        features[f"premium_close_mean_{window}"] = close_mean

    return features.replace([np.inf, -np.inf], np.nan)


def build_rank_datasets(
    raw_tickers,
    bar_size: str,
    prediction_horizon_bars: int,
    data_dir: str = "computed-data/dataset",
    include_futures_metrics: bool = False,
    include_premium_index: bool = False,
) -> dict[str, dict[str, pd.DataFrame | pd.Series]]:
    datasets = {}
    for raw_df, ticker in raw_tickers:
        ohlcv = raw_to_ohlcv(raw_df)
        bars = (
            ohlcv.resample(bar_size)
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            .dropna()
        )
        future_return = np.log(bars["Close"].shift(-prediction_horizon_bars) / bars["Open"].shift(-1))
        labels = (future_return > 0).astype(int)
        features = base_features(bars)
        if include_futures_metrics:
            futures_features = load_futures_metrics_features(data_dir, ticker, bar_size)
            if not futures_features.empty:
                features = features.join(futures_features, how="left")
                features.loc[:, futures_features.columns] = features.loc[:, futures_features.columns].fillna(0.0)
        if include_premium_index:
            premium_features = load_premium_index_features(data_dir, ticker, bar_size)
            if not premium_features.empty:
                features = features.join(premium_features, how="left")
                features.loc[:, premium_features.columns] = features.loc[:, premium_features.columns].fillna(0.0)
        datasets[ticker] = {
            "bars": bars,
            "features": features,
            "labels": labels,
            "future_return": future_return,
        }

    close_frame = pd.DataFrame({ticker: data["bars"]["Close"] for ticker, data in datasets.items()}).sort_index()
    volume_frame = pd.DataFrame({ticker: data["bars"]["Volume"] for ticker, data in datasets.items()}).sort_index()
    market_return = np.log(close_frame / close_frame.shift(1)).mean(axis=1)
    market_close = close_frame.mean(axis=1)
    market_volume = volume_frame.mean(axis=1)
    market_features = pd.DataFrame(index=close_frame.index)
    for lag in FEATURE_LAGS:
        market_features[f"market_log_return_{lag}"] = market_return.rolling(lag).sum()
    for window in ROLLING_WINDOWS:
        market_features[f"market_volatility_{window}"] = market_return.rolling(window).std()
        market_features[f"market_sma_distance_{window}"] = np.log(market_close / market_close.rolling(window).mean())
        market_features[f"market_volume_zscore_{window}"] = (
            (market_volume - market_volume.rolling(window).mean()) / (market_volume.rolling(window).std() + 1e-12)
        )

    for ticker, data in datasets.items():
        relative = pd.DataFrame(index=data["features"].index)
        ticker_return = np.log(data["bars"]["Close"] / data["bars"]["Close"].shift(1))
        for lag in FEATURE_LAGS:
            relative[f"relative_log_return_{lag}"] = ticker_return.rolling(lag).sum() - market_return.rolling(lag).sum()
        features = data["features"].join(market_features, how="left").join(relative, how="left")
        feature_index = features.dropna().index
        target_index = feature_index.intersection(data["future_return"].dropna().index)
        data["features"] = features.loc[feature_index]
        data["labels"] = data["labels"].loc[target_index]
        data["future_return"] = data["future_return"].loc[target_index]

    return datasets
