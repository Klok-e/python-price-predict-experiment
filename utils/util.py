from __future__ import annotations

import datetime
import os

import pandas as pd

try:
    from binance_historical_data import BinanceDataDumper
except ModuleNotFoundError:
    BinanceDataDumper = None


OHLC_COLUMNS = ["Open", "High", "Low", "Close"]
DEFAULT_TICKERS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
]
DEFAULT_EXPERIMENT_START_DATE = "2025-01-01"
BINANCE_DATA_START_DATE = datetime.date(2023, 1, 1)


def parse_tickers(value: str) -> tuple[str, ...]:
    return tuple(ticker.strip().upper() for ticker in value.split(",") if ticker.strip())


def open_time_to_datetime(open_time) -> pd.DatetimeIndex:
    if pd.api.types.is_datetime64_any_dtype(open_time):
        return pd.DatetimeIndex(pd.to_datetime(open_time))

    numeric_open_time = pd.Series(pd.to_numeric(open_time, errors="coerce"))
    result = pd.Series(pd.NaT, index=numeric_open_time.index, dtype="datetime64[ns]")
    unit_masks = [
        (numeric_open_time >= 1e17, "ns"),
        ((numeric_open_time >= 1e14) & (numeric_open_time < 1e17), "us"),
        ((numeric_open_time >= 1e11) & (numeric_open_time < 1e14), "ms"),
        (numeric_open_time < 1e11, "s"),
    ]

    for mask, unit in unit_masks:
        if mask.any():
            result.loc[mask] = pd.to_datetime(numeric_open_time.loc[mask], unit=unit, errors="coerce")

    return pd.DatetimeIndex(result)


def validate_ohlc_bars(df: pd.DataFrame, ticker_name: str | None = None, frequency="1min") -> pd.DataFrame:
    missing_columns = [column for column in OHLC_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{ticker_name or 'ticker'} is missing OHLC columns: {missing_columns}")

    result = df.copy()
    if not isinstance(result.index, pd.DatetimeIndex):
        if "Open time" not in result.columns:
            raise ValueError(f"{ticker_name or 'ticker'} needs a DatetimeIndex or an Open time column")
        result.index = open_time_to_datetime(result["Open time"])
        if result.index.isna().any():
            raise ValueError(f"{ticker_name or 'ticker'} has invalid Open time values")

    result[OHLC_COLUMNS] = result[OHLC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    if result.index.has_duplicates:
        duplicates = result.index[result.index.duplicated()].unique()
        raise ValueError(f"{ticker_name or 'ticker'} has duplicate bars, first duplicate: {duplicates[0]}")

    if not result.index.is_monotonic_increasing:
        raise ValueError(f"{ticker_name or 'ticker'} has a non-monotonic index")

    if result[OHLC_COLUMNS].isna().any().any():
        raise ValueError(f"{ticker_name or 'ticker'} has NaN OHLC values")

    invalid_ohlc = (
        (result["High"] < result[["Open", "Close", "Low"]].max(axis=1))
        | (result["Low"] > result[["Open", "Close", "High"]].min(axis=1))
    )
    if invalid_ohlc.any():
        raise ValueError(f"{ticker_name or 'ticker'} has invalid OHLC values at {invalid_ohlc.idxmax()}")

    if frequency is not None and len(result.index) > 1:
        expected_delta = pd.Timedelta(frequency)
        deltas = result.index.to_series().diff().dropna()
        missing_or_irregular = deltas[deltas != expected_delta]
        if not missing_or_irregular.empty:
            raise ValueError(
                f"{ticker_name or 'ticker'} has missing or irregular bars before {missing_or_irregular.index[0]}"
            )

    return result


def download_ohlc_data(data_dir: str, tickers=None, start_date=BINANCE_DATA_START_DATE):
    if tickers is None:
        tickers = DEFAULT_TICKERS
    if BinanceDataDumper is None:
        raise RuntimeError("binance_historical_data is required for downloads")

    data_dumper = BinanceDataDumper(
        path_dir_where_to_dump=f"{data_dir}/",
        asset_class="spot",
        data_type="klines",
        data_frequency="1m",
    )
    data_dumper.dump_data(
        tickers=tickers,
        date_start=start_date,
        is_to_update_existing=True,
    )
    return load_cached_ohlc_data(data_dir, tickers)


def load_cached_ohlc_data(data_dir: str, tickers=None):
    if tickers is None:
        tickers = DEFAULT_TICKERS
    df_tickers = list(zip((get_df_for_ticker(data_dir, ticker) for ticker in tickers), tickers))
    missing = [ticker for df, ticker in df_tickers if df.empty]
    if missing:
        raise FileNotFoundError(
            f"No cached 1m Binance klines found for {missing} under {data_dir}. "
            "Run download_data.py first."
        )
    return df_tickers


def filter_tickers_by_start_date(df_tickers, start_date: str):
    start = pd.Timestamp(start_date)
    filtered = []
    for df, ticker in df_tickers:
        result = df.copy()
        if not isinstance(result.index, pd.DatetimeIndex):
            if "Open time" not in result.columns:
                raise ValueError(f"{ticker} needs a DatetimeIndex or an Open time column")
            result.index = open_time_to_datetime(result["Open time"])
        result = result.sort_index().loc[start:]
        if result.empty:
            raise ValueError(f"{ticker} has no cached rows at or after {start}")
        filtered.append((result, ticker))
    return filtered


def get_df_for_ticker(data_dir: str, ticker: str) -> pd.DataFrame:
    columns = [
        "Open time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close time",
        "Quote asset volume",
        "Number of trades",
        "Taker buy base asset volume",
        "Taker buy quote asset volume",
        "Ignore",
    ]

    frames = []
    kline_dirs = [
        f"{data_dir}/spot/monthly/klines/{ticker}/1m",
        f"{data_dir}/spot/daily/klines/{ticker}/1m",
    ]

    for kline_dir in kline_dirs:
        filenames = sorted(next(os.walk(kline_dir), (None, None, []))[2])
        for filename in filenames:
            frames.append(pd.read_csv(f"{kline_dir}/{filename}", header=None, names=columns))

    if not frames:
        return pd.DataFrame(columns=columns)

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(by="Open time")
    df = df.drop_duplicates(subset="Open time", keep="first")
    return df
