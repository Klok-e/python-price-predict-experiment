"""Public Binance archive synchronization and canonical adapters."""

from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd

from .config import PolicyConfig
from .market_data import CanonicalDataset, InstrumentData

ARCHIVE = "https://data.binance.vision/data"
KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
)


def _utc_index(raw: pd.Series) -> pd.DatetimeIndex:
    numeric = pd.to_numeric(raw, errors="coerce")
    sample = numeric.dropna().iloc[0]
    if sample > 100_000_000_000_000:
        converted = pd.to_datetime(numeric.to_numpy(), unit="us", utc=True)
    else:
        converted = pd.to_datetime(numeric.to_numpy(), unit="ms", utc=True)
    return pd.DatetimeIndex(converted)


def _read_klines(files: list[Path]) -> pd.DataFrame:
    if not files:
        raise ValueError("missing required Binance kline archive; run data-sync")
    frames: list[pd.DataFrame] = []
    for path in files:
        raw = pd.read_csv(path, header=None, low_memory=False)
        if pd.isna(pd.to_numeric(raw.iloc[0, 0], errors="coerce")):
            raw = raw.iloc[1:]
        raw = raw.iloc[:, : len(KLINE_COLUMNS)]
        raw.columns = KLINE_COLUMNS
        raw.index = _utc_index(raw["open_time"])
        numeric_columns = ("open", "high", "low", "close", "volume", "taker_buy_volume", "trades")
        frames.append(raw.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce"))
    return pd.concat(frames).sort_index().loc[lambda value: ~value.index.duplicated(keep="last")]


def _read_timestamped_series(
    files: list[Path], timestamp_columns: tuple[str, ...], value_columns: tuple[str, ...], name: str
) -> pd.Series:
    if not files:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name=name)
    frames = [pd.read_csv(path, low_memory=False) for path in files]
    raw = pd.concat(frames, ignore_index=True)
    timestamp_column = next((column for column in timestamp_columns if column in raw), None)
    value_column = next((column for column in value_columns if column in raw), None)
    if timestamp_column is None or value_column is None:
        raise ValueError(f"cannot identify {name} columns in Binance archive")
    timestamps = raw[timestamp_column]
    if pd.api.types.is_numeric_dtype(timestamps):
        index = _utc_index(timestamps)
    else:
        index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    values = pd.Series(pd.to_numeric(raw[value_column], errors="coerce").to_numpy(), index=index, name=name)
    return values.sort_index().loc[lambda value: ~value.index.duplicated(keep="last")]


@dataclass(frozen=True)
class HistoricalArchiveAdapter:
    root: Path
    tickers: tuple[str, ...]
    mode: Literal["historical", "live"] = "historical"

    def _files(self, pattern: str) -> list[Path]:
        return sorted(self.root.glob(pattern))

    def load(self) -> CanonicalDataset:
        instruments: dict[str, InstrumentData] = {}
        for ticker in self.tickers:
            spot = _read_klines(self._files(f"spot/*/klines/{ticker}/1m/{ticker}-1m-*.csv"))
            perpetual = _read_klines(self._files(f"futures/um/*/klines/{ticker}/1m/{ticker}-1m-*.csv"))
            funding = _read_timestamped_series(
                self._files(f"futures/um/*/fundingRate/{ticker}/{ticker}-fundingRate-*.csv"),
                ("calc_time", "fundingTime", "funding_time"),
                ("last_funding_rate", "fundingRate", "funding_rate"),
                "funding_rate",
            )
            metrics_files = self._files(f"futures/um/*/metrics/{ticker}/{ticker}-metrics-*.csv")
            interest = _read_timestamped_series(
                metrics_files,
                ("create_time", "timestamp"),
                ("sum_open_interest", "open_interest"),
                "open_interest",
            )
            premium_files = self._files(f"futures/um/*/premiumIndexKlines/{ticker}/1m/{ticker}-1m-*.csv")
            if premium_files:
                premium_frame = _read_klines(premium_files)
                premium = premium_frame["close"].rename("premium")
            else:
                premium = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name="premium")
            instruments[ticker] = InstrumentData(
                perpetual=perpetual,
                spot=spot,
                funding=funding,
                open_interest=interest,
                premium=premium,
            )
        dataset = CanonicalDataset(instruments=instruments, tickers=self.tickers)
        dataset.validate()
        return dataset


@dataclass(frozen=True)
class PublicPaperAdapter(HistoricalArchiveAdapter):
    """Public-data-only adapter; data-sync appends revealed archive/API observations."""

    mode: Literal["historical", "live"] = "live"


def _months(start: date, end: date) -> list[date]:
    result = []
    current = start.replace(day=1)
    while current <= end.replace(day=1):
        result.append(current)
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
    return result


@dataclass
class BinanceDataSync:
    root: Path
    config: PolicyConfig
    retries: int = 3

    def _download_zip(self, url: str, destination: Path) -> str:
        if destination.exists():
            return "cached"
        destination.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    payload = response.read()
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    csv_name = next(name for name in archive.namelist() if name.endswith(".csv"))
                    temporary = destination.with_suffix(".tmp")
                    with archive.open(csv_name) as source, temporary.open("wb") as target:
                        target.write(source.read())
                    temporary.replace(destination)
                return "downloaded"
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return "not-yet-listed"
            except (OSError, TimeoutError, urllib.error.URLError, zipfile.BadZipFile):
                pass
            time.sleep(0.5 * (attempt + 1))
        return "failed"

    def sync(self) -> str:
        start = date.fromisoformat(self.config.history_start)
        today = datetime.now(UTC).date()
        completed_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        counts: dict[str, int] = {}
        monthly_sources = (
            ("spot", "klines", "1m"),
            ("futures/um", "klines", "1m"),
            ("futures/um", "fundingRate", None),
        )
        for ticker in self.config.tickers:
            for market, source, interval in monthly_sources:
                for month in _months(start, completed_month):
                    suffix = f"-{interval}" if interval else ""
                    filename = f"{ticker}{suffix}-{month:%Y-%m}"
                    parts = [ARCHIVE, market, "monthly", source, ticker]
                    local_parts = [self.root, market, "monthly", source, ticker]
                    if interval:
                        parts.append(interval)
                        local_parts.append(interval)
                    status = self._download_zip(
                        "/".join(str(part) for part in parts) + f"/{filename}.zip",
                        Path(*(str(part) for part in local_parts)) / f"{filename}.csv",
                    )
                    counts[status] = counts.get(status, 0) + 1
            current = start
            daily_sources = (
                ("spot", "klines", "1m"),
                ("futures/um", "klines", "1m"),
                ("futures/um", "fundingRate", None),
                ("futures/um", "metrics", None),
                ("futures/um", "premiumIndexKlines", "1m"),
            )
            while current < today:
                for market, source, interval in daily_sources:
                    suffix = f"-{interval}" if interval else ""
                    source_name = "metrics" if source == "metrics" else source
                    filename = (
                        f"{ticker}-metrics-{current.isoformat()}"
                        if source == "metrics"
                        else f"{ticker}{suffix}-{current.isoformat()}"
                    )
                    parts = [ARCHIVE, market, "daily", source_name, ticker]
                    local_parts = [self.root, market, "daily", source_name, ticker]
                    if interval:
                        parts.append(interval)
                        local_parts.append(interval)
                    status = self._download_zip(
                        "/".join(str(part) for part in parts) + f"/{filename}.zip",
                        Path(*(str(part) for part in local_parts)) / f"{filename}.csv",
                    )
                    counts[status] = counts.get(status, 0) + 1
                current += timedelta(days=1)
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        return f"Binance data-sync complete: {rendered}"
