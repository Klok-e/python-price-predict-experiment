"""Public Binance archive synchronization and canonical adapters."""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from .config import PolicyConfig
from .market_data import CanonicalDataset, InstrumentData, PublicDataUnavailable

ARCHIVE = "https://data.binance.vision/data"
FUTURES_API = "https://fapi.binance.com"
SPOT_API = "https://api.binance.com"
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
ARCHIVE_SOURCES: tuple[tuple[str, str, str | None], ...] = (
    ("spot", "klines", "1m"),
    ("futures/um", "klines", "1m"),
    ("futures/um", "fundingRate", None),
    ("futures/um", "metrics", None),
    ("futures/um", "premiumIndexKlines", "1m"),
)


def _archive_request(
    root: Path,
    *,
    ticker: str,
    frequency: Literal["monthly", "daily"],
    market: str,
    source: str,
    interval: str | None,
    period: str,
) -> tuple[str, Path]:
    descriptor = interval or source
    filename = f"{ticker}-{descriptor}-{period}"
    remote = [ARCHIVE, market, frequency, source, ticker]
    local = [root, market, frequency, source, ticker]
    if interval:
        remote.append(interval)
        local.append(interval)
    return (
        "/".join(str(part) for part in remote) + f"/{filename}.zip",
        Path(*(str(part) for part in local)) / f"{filename}.csv",
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
    values_by_file: list[pd.Series] = []
    for path in files:
        raw = pd.read_csv(path, low_memory=False)
        timestamp_column = next((column for column in timestamp_columns if column in raw), None)
        value_column = next((column for column in value_columns if column in raw), None)
        if timestamp_column is None or value_column is None:
            raise ValueError(f"cannot identify {name} columns in Binance archive")
        timestamps = raw[timestamp_column]
        if pd.api.types.is_numeric_dtype(timestamps):
            index = _utc_index(timestamps)
        else:
            index = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
        values_by_file.append(
            pd.Series(pd.to_numeric(raw[value_column], errors="coerce").to_numpy(), index=index, name=name)
        )
    values = pd.concat(values_by_file)
    ordered = values.sort_index()
    return cast(pd.Series, ordered.loc[~ordered.index.duplicated(keep="last")])


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
class BookTicker:
    ticker: str
    bid: float
    ask: float
    exchange_time: pd.Timestamp
    observed_at: pd.Timestamp

    @property
    def midpoint(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True)
class SettledFunding:
    event_time: pd.Timestamp
    rate: float
    mark_price: float


@dataclass(frozen=True)
class PaperInstrumentCatchup:
    perpetual: pd.DataFrame
    spot: pd.DataFrame
    premium: pd.Series
    settled_funding: tuple[SettledFunding, ...]
    open_interest: pd.Series


@dataclass(frozen=True)
class PaperFeedObservation:
    server_time: pd.Timestamp
    observed_at: pd.Timestamp
    quotes: dict[str, BookTicker]
    instruments: dict[str, PaperInstrumentCatchup]
    tickers: tuple[str, ...]

    def canonical_dataset(self) -> CanonicalDataset:
        instruments = {
            ticker: InstrumentData(
                perpetual=value.perpetual,
                spot=value.spot,
                funding=pd.Series(
                    [event.rate for event in value.settled_funding],
                    index=pd.DatetimeIndex([event.event_time for event in value.settled_funding]),
                    dtype=float,
                    name="funding_rate",
                ),
                open_interest=value.open_interest,
                premium=value.premium,
            )
            for ticker, value in self.instruments.items()
        }
        return CanonicalDataset(instruments=instruments, tickers=self.tickers)

    def append_to(self, history: CanonicalDataset) -> CanonicalDataset:
        """Append revealed public rows without replacing the fitting history."""
        if history.tickers != self.tickers:
            raise ValueError("paper catch-up must match the fixed Trading Universe")

        def frame(previous: pd.DataFrame, revealed: pd.DataFrame) -> pd.DataFrame:
            return pd.concat((previous, revealed)).sort_index().loc[lambda value: ~value.index.duplicated(keep="last")]

        def series(previous: pd.Series, revealed: pd.Series) -> pd.Series:
            result = pd.concat((previous, revealed)).sort_index()
            return cast(pd.Series, result.loc[~result.index.duplicated(keep="last")])

        instruments: dict[str, InstrumentData] = {}
        for ticker in self.tickers:
            previous = history.instruments[ticker]
            revealed = self.instruments[ticker]
            funding = pd.Series(
                [event.rate for event in revealed.settled_funding],
                index=pd.DatetimeIndex([event.event_time for event in revealed.settled_funding]),
                dtype=float,
                name="funding_rate",
            )
            instruments[ticker] = InstrumentData(
                perpetual=frame(previous.perpetual, revealed.perpetual),
                spot=frame(previous.spot, revealed.spot),
                funding=series(previous.funding, funding),
                open_interest=series(previous.open_interest, revealed.open_interest),
                premium=series(previous.premium, revealed.premium),
            )
        result = CanonicalDataset(instruments=instruments, tickers=self.tickers)
        result.validate()
        return result


@dataclass
class PublicPaperAdapter:
    """Contemporaneous, unauthenticated Binance feed for the Paper Account."""

    root: Path
    tickers: tuple[str, ...]
    mode: Literal["historical", "live"] = "live"
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    timeout: float = 10.0
    quote_max_age: timedelta = timedelta(seconds=5)
    bootstrap_minutes: int = 7 * 24 * 60
    _history: CanonicalDataset | None = field(default=None, init=False, repr=False)

    def _json(self, base: str, path: str, parameters: dict[str, object] | None = None) -> object:
        query = "" if not parameters else "?" + urllib.parse.urlencode(parameters)
        url = f"{base}{path}{query}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                payload = json.loads(response.read())
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as error:
            raise PublicDataUnavailable(f"Binance public endpoint unavailable: {path}") from error
        if isinstance(payload, dict) and "code" in payload:
            raise PublicDataUnavailable(f"Binance public endpoint rejected request: {path}")
        return payload

    @staticmethod
    def _timestamp(milliseconds: object) -> pd.Timestamp:
        try:
            if not isinstance(milliseconds, (int, float, str)):
                raise TypeError
            return pd.Timestamp(int(milliseconds), unit="ms", tz="UTC")
        except (TypeError, ValueError, OverflowError) as error:
            raise PublicDataUnavailable("Binance returned an invalid UTC timestamp") from error

    def _server_time(self) -> pd.Timestamp:
        payload = self._json(FUTURES_API, "/fapi/v1/time")
        if not isinstance(payload, dict) or "serverTime" not in payload:
            raise PublicDataUnavailable("Binance server time is missing")
        return self._timestamp(payload["serverTime"])

    def _quotes(self, server_time: pd.Timestamp, observed_at: pd.Timestamp) -> dict[str, BookTicker]:
        payload = self._json(FUTURES_API, "/fapi/v1/ticker/bookTicker")
        if not isinstance(payload, list):
            raise PublicDataUnavailable("Binance all-ticker book snapshot is malformed")
        by_ticker = {
            str(row.get("symbol")): row
            for row in payload
            if isinstance(row, dict) and str(row.get("symbol")) in self.tickers
        }
        if set(by_ticker) != set(self.tickers):
            raise PublicDataUnavailable("Binance book snapshot is missing a Trading Universe ticker")
        quotes: dict[str, BookTicker] = {}
        for ticker in self.tickers:
            row = by_ticker[ticker]
            exchange_time = self._timestamp(row.get("time"))
            try:
                bid = float(row["bidPrice"])
                ask = float(row["askPrice"])
            except (KeyError, TypeError, ValueError) as error:
                raise PublicDataUnavailable(f"{ticker} book snapshot has an invalid price") from error
            if bid <= 0.0 or ask < bid:
                raise PublicDataUnavailable(f"{ticker} book snapshot has an invalid spread")
            age = server_time - exchange_time
            if age < -pd.Timedelta(seconds=2) or age > self.quote_max_age:
                raise PublicDataUnavailable(f"{ticker} book snapshot is stale")
            quotes[ticker] = BookTicker(ticker, bid, ask, exchange_time, observed_at)
        return quotes

    @staticmethod
    def _first_open(after: pd.Timestamp | None, server_time: pd.Timestamp, bootstrap_minutes: int) -> pd.Timestamp:
        latest = server_time.floor("min") - pd.Timedelta(minutes=1)
        if after is None:
            return latest - pd.Timedelta(minutes=bootstrap_minutes - 1)
        return (after + pd.Timedelta(milliseconds=1)).floor("min")

    def _klines(
        self,
        *,
        base: str,
        path: str,
        ticker: str,
        after: pd.Timestamp | None,
        server_time: pd.Timestamp,
    ) -> pd.DataFrame:
        first_open = self._first_open(after, server_time, self.bootstrap_minutes)
        latest_open = server_time.floor("min") - pd.Timedelta(minutes=1)
        if first_open > latest_open:
            return pd.DataFrame(
                columns=("open", "high", "low", "close", "volume", "taker_buy_volume", "trades"),
                index=pd.DatetimeIndex([], tz="UTC"),
            )
        cursor = first_open
        rows: list[list[object]] = []
        irreconstructible_gap = after is not None and latest_open > first_open
        page_limit = 1000 if base == SPOT_API else 1500
        while cursor <= latest_open:
            payload = self._json(
                base,
                path,
                {
                    "symbol": ticker,
                    "interval": "1m",
                    "startTime": int(cursor.timestamp() * 1_000),
                    "endTime": int(server_time.timestamp() * 1_000) - 1,
                    "limit": page_limit,
                },
            )
            if not isinstance(payload, list) or any(not isinstance(row, list) for row in payload):
                raise PublicDataUnavailable(f"{ticker} public one-minute rows are malformed")
            page = [row for row in payload if len(row) >= len(KLINE_COLUMNS)]
            rows.extend(page)
            if len(page) < page_limit:
                break
            next_cursor = self._timestamp(page[-1][0]) + pd.Timedelta(minutes=1)
            if next_cursor <= cursor:
                raise PublicDataUnavailable(f"{ticker} public one-minute pagination did not advance")
            cursor = next_cursor
        if not rows:
            if irreconstructible_gap:
                raise PublicDataUnavailable(f"{ticker} required paper minute range is missing")
            raise PublicDataUnavailable(f"{ticker} required public one-minute rows are missing")
        raw = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        raw.index = pd.DatetimeIndex(pd.to_datetime(pd.to_numeric(raw["open_time"]), unit="ms", utc=True))
        close_times = pd.DatetimeIndex(pd.to_datetime(pd.to_numeric(raw["close_time"]), unit="ms", utc=True))
        closed = close_times < server_time
        if after is not None:
            closed &= close_times > after
        raw = raw.loc[closed]
        raw = raw.loc[(raw.index >= first_open) & (raw.index <= latest_open)]
        raw = raw.loc[~raw.index.duplicated(keep="last")].sort_index()
        expected = pd.date_range(first_open, latest_open, freq="min", tz="UTC")
        if not raw.index.equals(expected):
            if irreconstructible_gap:
                raise PublicDataUnavailable(f"{ticker} required paper minute range has a gap")
            raise PublicDataUnavailable(f"{ticker} required closed one-minute rows are stale or missing")
        columns = ("open", "high", "low", "close", "volume", "taker_buy_volume", "trades")
        frame = raw.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
        if frame.isna().any().any():
            raise PublicDataUnavailable(f"{ticker} required public one-minute values are invalid")
        return cast(pd.DataFrame, frame)

    @staticmethod
    def _empty_klines() -> pd.DataFrame:
        return pd.DataFrame(
            columns=("open", "high", "low", "close", "volume", "taker_buy_volume", "trades"),
            index=pd.DatetimeIndex([], tz="UTC"),
        )

    def _funding(
        self, ticker: str, after: pd.Timestamp | None, server_time: pd.Timestamp
    ) -> tuple[SettledFunding, ...]:
        first = self._first_open(after, server_time, self.bootstrap_minutes)
        payload = self._json(
            FUTURES_API,
            "/fapi/v1/fundingRate",
            {
                "symbol": ticker,
                "startTime": int(first.timestamp() * 1_000),
                "endTime": int(server_time.timestamp() * 1_000),
                "limit": 1000,
            },
        )
        if not isinstance(payload, list):
            raise PublicDataUnavailable(f"{ticker} settled funding rows are malformed")
        result: list[SettledFunding] = []
        for row in payload:
            if not isinstance(row, dict):
                raise PublicDataUnavailable(f"{ticker} settled funding row is malformed")
            event_time = self._timestamp(row.get("fundingTime"))
            if event_time > server_time or (after is not None and event_time <= after):
                continue
            try:
                rate = float(row["fundingRate"])
                mark_price = float(row["markPrice"])
            except (KeyError, TypeError, ValueError) as error:
                raise PublicDataUnavailable(f"{ticker} settled funding row is incomplete") from error
            if mark_price <= 0.0:
                raise PublicDataUnavailable(f"{ticker} settled funding mark price is invalid")
            result.append(SettledFunding(event_time, rate, mark_price))
        if not result and server_time - first >= pd.Timedelta(hours=8):
            raise PublicDataUnavailable(f"{ticker} required settled funding is missing")
        return tuple(sorted(result, key=lambda event: event.event_time))

    def _open_interest(self, ticker: str, after: pd.Timestamp | None, server_time: pd.Timestamp) -> pd.Series:
        first = self._first_open(after, server_time, self.bootstrap_minutes)
        try:
            payload = self._json(
                FUTURES_API,
                "/futures/data/openInterestHist",
                {
                    "symbol": ticker,
                    "period": "5m",
                    "startTime": int(first.timestamp() * 1_000),
                    "endTime": int(server_time.timestamp() * 1_000),
                    "limit": 500,
                },
            )
            if not isinstance(payload, list):
                raise PublicDataUnavailable("open-interest response is malformed")
            values = {
                self._timestamp(row.get("timestamp")): float(row["sumOpenInterest"])
                for row in payload
                if isinstance(row, dict)
            }
        except (PublicDataUnavailable, KeyError, TypeError, ValueError):
            values = {}
        return pd.Series(values, dtype=float, name="open_interest").sort_index()

    def observe(self, after: datetime | None) -> PaperFeedObservation:
        """Fetch one current midpoint snapshot and REST catch-up; quote history is never synthesized."""
        after_time = pd.Timestamp(after).tz_convert("UTC") if after is not None else None
        server_time = self._server_time()
        observed_at = pd.Timestamp(self.clock()).tz_convert("UTC")
        quotes = self._quotes(server_time, observed_at)
        instruments: dict[str, PaperInstrumentCatchup] = {}
        for ticker in self.tickers:
            perpetual = self._klines(
                base=FUTURES_API,
                path="/fapi/v1/klines",
                ticker=ticker,
                after=after_time,
                server_time=server_time,
            )
            try:
                spot = self._klines(
                    base=SPOT_API,
                    path="/api/v3/klines",
                    ticker=ticker,
                    after=after_time,
                    server_time=server_time,
                )
            except PublicDataUnavailable:
                spot = self._empty_klines()
            try:
                premium_frame = self._klines(
                    base=FUTURES_API,
                    path="/fapi/v1/premiumIndexKlines",
                    ticker=ticker,
                    after=after_time,
                    server_time=server_time,
                )
            except PublicDataUnavailable:
                premium_frame = self._empty_klines()
            instruments[ticker] = PaperInstrumentCatchup(
                perpetual=perpetual,
                spot=spot,
                premium=premium_frame["close"].rename("premium"),
                settled_funding=self._funding(ticker, after_time, server_time),
                open_interest=self._open_interest(ticker, after_time, server_time),
            )
        return PaperFeedObservation(server_time, observed_at, quotes, instruments, self.tickers)

    def mark(self, after: datetime | None) -> PaperFeedObservation:
        """Fetch only the contemporaneous inputs needed for one-minute execution and marking."""
        after_time = pd.Timestamp(after).tz_convert("UTC") if after is not None else None
        server_time = self._server_time()
        observed_at = pd.Timestamp(self.clock()).tz_convert("UTC")
        quotes = self._quotes(server_time, observed_at)
        query_after = after_time
        if query_after is None:
            query_after = server_time.floor("min") - pd.Timedelta(minutes=1, milliseconds=1)

        def instrument(ticker: str) -> PaperInstrumentCatchup:
            perpetual = self._klines(
                base=FUTURES_API,
                path="/fapi/v1/klines",
                ticker=ticker,
                after=query_after,
                server_time=server_time,
            )
            return PaperInstrumentCatchup(
                perpetual=perpetual,
                spot=self._empty_klines(),
                premium=pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"), name="premium"),
                settled_funding=self._funding(ticker, query_after, server_time),
                open_interest=pd.Series(
                    dtype=float,
                    index=pd.DatetimeIndex([], tz="UTC"),
                    name="open_interest",
                ),
            )

        with ThreadPoolExecutor(max_workers=len(self.tickers)) as executor:
            futures = {ticker: executor.submit(instrument, ticker) for ticker in self.tickers}
            instruments = {ticker: futures[ticker].result() for ticker in self.tickers}
        return PaperFeedObservation(server_time, observed_at, quotes, instruments, self.tickers)

    def load(self) -> CanonicalDataset:
        if self._history is None:
            self._history = HistoricalArchiveAdapter(self.root, self.tickers).load()
        last_close = min(value.perpetual.index.max() for value in self._history.instruments.values()) + pd.Timedelta(
            minutes=1, milliseconds=-1
        )
        self._history = self.observe(last_close.to_pydatetime()).append_to(self._history)
        return self._history


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
    today: Callable[[], date] = lambda: datetime.now(UTC).date()
    workers: int = 8

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

    def _perpetual_gap_requests(self, ticker: str, through: date) -> list[tuple[str, Path]]:
        files = sorted(self.root.glob(f"futures/um/*/klines/{ticker}/1m/{ticker}-1m-*.csv"))
        observed: list[pd.DatetimeIndex] = []
        for path in files:
            raw = pd.read_csv(path, header=None, usecols=[0]).iloc[:, 0]
            numeric = pd.to_numeric(raw, errors="coerce").dropna()
            if not numeric.empty:
                observed.append(_utc_index(numeric))
        if not observed:
            return []
        index = observed[0]
        for addition in observed[1:]:
            index = pd.DatetimeIndex(index.append(addition))
        index = pd.DatetimeIndex(index.drop_duplicates().sort_values())
        end = pd.Timestamp(through, tz="UTC") + pd.Timedelta(days=1, minutes=-1)
        expected = pd.date_range(index.min(), end, freq="min", tz="UTC")
        missing_days = pd.DatetimeIndex(expected.difference(index).normalize().drop_duplicates())
        requests = []
        for timestamp in missing_days:
            day = timestamp.date().isoformat()
            filename = f"{ticker}-1m-{day}"
            requests.append(
                (
                    f"{ARCHIVE}/futures/um/daily/klines/{ticker}/1m/{filename}.zip",
                    self.root / "futures/um/daily/klines" / ticker / "1m" / f"{filename}.csv",
                )
            )
        return requests

    def _sync_recent_funding(self, ticker: str) -> int:
        files = sorted(self.root.glob(f"futures/um/*/fundingRate/{ticker}/{ticker}-fundingRate-*.csv"))
        if not files:
            return 0
        existing = _read_timestamped_series(
            files,
            ("calc_time", "fundingTime", "funding_time"),
            ("last_funding_rate", "fundingRate", "funding_rate"),
            "funding_rate",
        )
        start = int(existing.index.max().timestamp() * 1_000) + 1
        url = f"{FUTURES_API}/fapi/v1/fundingRate?" + urllib.parse.urlencode(
            {"symbol": ticker, "startTime": start, "limit": 1000}
        )
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.loads(response.read())
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError):
            return 0
        if not isinstance(payload, list):
            return 0
        rows = [
            {
                "fundingTime": int(row["fundingTime"]),
                "fundingRate": float(row["fundingRate"]),
            }
            for row in payload
            if isinstance(row, dict) and "fundingTime" in row and "fundingRate" in row
        ]
        if not rows:
            return 0
        destination = self.root / "futures/um/live/fundingRate" / ticker / f"{ticker}-fundingRate-rest.csv"
        destination.parent.mkdir(parents=True, exist_ok=True)
        prior = (
            pd.read_csv(destination) if destination.exists() else pd.DataFrame(columns=("fundingTime", "fundingRate"))
        )
        combined = pd.concat((prior, pd.DataFrame(rows)), ignore_index=True)
        combined = combined.drop_duplicates(subset="fundingTime", keep="last").sort_values("fundingTime")
        temporary = destination.with_suffix(".tmp")
        combined.to_csv(temporary, index=False)
        temporary.replace(destination)
        return len(rows)

    def sync(self) -> str:
        start = date.fromisoformat(self.config.history_start)
        today = self.today()
        completed_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        requests: list[tuple[str, Path]] = []
        for ticker in self.config.tickers:
            for market, source, interval in ARCHIVE_SOURCES:
                for month in _months(start, completed_month):
                    requests.append(
                        _archive_request(
                            self.root,
                            ticker=ticker,
                            frequency="monthly",
                            market=market,
                            source=source,
                            interval=interval,
                            period=f"{month:%Y-%m}",
                        )
                    )
            current = today.replace(day=1)
            while current < today:
                for market, source, interval in ARCHIVE_SOURCES:
                    requests.append(
                        _archive_request(
                            self.root,
                            ticker=ticker,
                            frequency="daily",
                            market=market,
                            source=source,
                            interval=interval,
                            period=current.isoformat(),
                        )
                    )
                current += timedelta(days=1)
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            statuses = executor.map(lambda request: self._download_zip(*request), requests)
            counts: dict[str, int] = {}
            for status in statuses:
                counts[status] = counts.get(status, 0) + 1
        gap_requests = [
            request
            for ticker in self.config.tickers
            for request in self._perpetual_gap_requests(ticker, today - timedelta(days=1))
        ]
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            for status in executor.map(lambda request: self._download_zip(*request), gap_requests):
                counts[status] = counts.get(status, 0) + 1
        recent_funding = sum(self._sync_recent_funding(ticker) for ticker in self.config.tickers)
        if recent_funding:
            counts["public-funding-events"] = recent_funding
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
        return f"Binance data-sync complete: {rendered}"
