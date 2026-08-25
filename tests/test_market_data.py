from __future__ import annotations

import json
import urllib.parse
from datetime import UTC, datetime
from threading import Barrier

import numpy as np
import pandas as pd
import pytest

from netgrowth.binance import BinanceDataSync, PublicDataUnavailable, PublicPaperAdapter
from netgrowth.config import load_config
from netgrowth.market_data import (
    BAR_COLUMNS,
    CanonicalDataset,
    InMemoryMarketData,
    InstrumentData,
    build_market_state,
)

TICKERS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")


class JsonResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _kline(open_time: int, close_time: int, close: str = "100.5") -> list[object]:
    return [
        open_time,
        "100.0",
        "101.0",
        "99.0",
        close,
        "10.0",
        close_time,
        "1005.0",
        20,
        "6.0",
        "603.0",
        "0",
    ]


def frame(index: pd.DatetimeIndex, start: float) -> pd.DataFrame:
    close = start + np.arange(len(index), dtype=float) * 0.1
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 10.0,
            "taker_buy_volume": 6.0,
            "trades": 20,
        },
        index=index,
    )


def dataset(*, missing_optional: bool = False) -> CanonicalDataset:
    index = pd.date_range("2026-01-01", periods=60 * 26, freq="min", tz="UTC")
    instruments = {}
    for number, ticker in enumerate(TICKERS):
        optional_index = index[:0] if missing_optional else index[::5]
        instruments[ticker] = InstrumentData(
            perpetual=frame(index, 100.0 + number * 10),
            spot=frame(index, 99.0 + number * 10),
            funding=pd.Series(0.0001, index=index[::480], name="funding_rate"),
            open_interest=pd.Series(1_000.0, index=optional_index, name="open_interest"),
            premium=pd.Series(0.001, index=optional_index, name="premium"),
        )
    return CanonicalDataset(instruments=instruments, tickers=TICKERS)


def test_historical_and_live_adapters_share_one_canonical_contract() -> None:
    expected = dataset()
    historical = InMemoryMarketData(expected, mode="historical").load()
    live = InMemoryMarketData(expected, mode="live").load()

    assert historical.identity_hash == live.identity_hash
    assert historical.tickers == live.tickers == TICKERS


def test_public_paper_feed_emits_current_quotes_and_only_newly_closed_rows(monkeypatch, tmp_path) -> None:
    server_time = pd.Timestamp("2026-08-03 12:03:30", tz="UTC")
    server_ms = int(server_time.timestamp() * 1_000)
    minute_1 = int(pd.Timestamp("2026-08-03 12:01:00", tz="UTC").timestamp() * 1_000)
    minute_2 = int(pd.Timestamp("2026-08-03 12:02:00", tz="UTC").timestamp() * 1_000)
    minute_3 = int(pd.Timestamp("2026-08-03 12:03:00", tz="UTC").timestamp() * 1_000)
    requested: list[str] = []

    def public_api(url: str, timeout: float):
        assert timeout == 10.0
        requested.append(url)
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path == "/fapi/v1/time":
            return JsonResponse({"serverTime": server_ms})
        if parsed.path == "/fapi/v1/ticker/bookTicker":
            assert not query
            return JsonResponse(
                [
                    {
                        "symbol": ticker,
                        "bidPrice": str(100 + number),
                        "askPrice": str(102 + number),
                        "time": server_ms - 100,
                    }
                    for number, ticker in enumerate(TICKERS)
                ]
            )
        if parsed.path in ("/fapi/v1/klines", "/api/v3/klines", "/fapi/v1/premiumIndexKlines"):
            return JsonResponse(
                [
                    _kline(minute_1, minute_2 - 1, "100.5"),
                    _kline(minute_2, minute_3 - 1, "101.5"),
                    _kline(minute_3, minute_3 + 59_999, "999.0"),
                ]
            )
        if parsed.path == "/fapi/v1/fundingRate":
            return JsonResponse(
                [
                    {
                        "symbol": query["symbol"][0],
                        "fundingTime": minute_2,
                        "fundingRate": "0.0001",
                        "markPrice": "101.25",
                    }
                ]
            )
        if parsed.path == "/futures/data/openInterestHist":
            return JsonResponse([{"timestamp": minute_2, "sumOpenInterest": "1234.5"}])
        raise AssertionError(f"unexpected public endpoint: {url}")

    monkeypatch.setattr("urllib.request.urlopen", public_api)
    adapter = PublicPaperAdapter(
        tmp_path,
        TICKERS,
        clock=lambda: datetime(2026, 8, 3, 12, 3, 31, tzinfo=UTC),
    )

    observed = adapter.observe(datetime(2026, 8, 3, 12, 0, 59, 999000, tzinfo=UTC))

    assert observed.server_time == server_time
    assert observed.observed_at == pd.Timestamp("2026-08-03 12:03:31", tz="UTC")
    assert observed.quotes["BTCUSDT"].midpoint == pytest.approx(101.0)
    assert observed.quotes["BTCUSDT"].exchange_time == pd.Timestamp("2026-08-03 12:03:29.900", tz="UTC")
    assert observed.quotes["BTCUSDT"].observed_at == observed.observed_at
    assert list(observed.instruments["BTCUSDT"].perpetual.index) == [
        pd.Timestamp("2026-08-03 12:01:00", tz="UTC"),
        pd.Timestamp("2026-08-03 12:02:00", tz="UTC"),
    ]
    assert observed.instruments["BTCUSDT"].spot.iloc[-1]["close"] == pytest.approx(101.5)
    assert observed.instruments["BTCUSDT"].premium.iloc[-1] == pytest.approx(101.5)
    assert observed.instruments["BTCUSDT"].settled_funding[0].mark_price == pytest.approx(101.25)
    assert observed.instruments["BTCUSDT"].open_interest.iloc[-1] == pytest.approx(1234.5)
    history = dataset()
    shift = pd.Timestamp("2026-08-03 12:00:00", tz="UTC") - history.instruments["BTCUSDT"].perpetual.index[-1]
    for instrument in history.instruments.values():
        for values in (
            instrument.perpetual,
            instrument.spot,
            instrument.funding,
            instrument.open_interest,
            instrument.premium,
        ):
            values.index = values.index + shift
    previous_rows = len(history.instruments["BTCUSDT"].perpetual)
    merged = observed.append_to(history)
    assert len(merged.instruments["BTCUSDT"].perpetual) == previous_rows + 2
    assert merged.instruments["BTCUSDT"].perpetual.index[-1] == pd.Timestamp("2026-08-03 12:02:00", tz="UTC")
    assert sum("/ticker/bookTicker" in url for url in requested) == 1
    assert all(url.startswith(("https://fapi.binance.com/", "https://api.binance.com/")) for url in requested)
    assert not any("signature=" in url or "apiKey=" in url for url in requested)


def test_public_paper_mark_fetches_only_execution_complete_inputs(monkeypatch, tmp_path) -> None:
    server_time = pd.Timestamp("2026-08-03 12:03:30", tz="UTC")
    server_ms = int(server_time.timestamp() * 1_000)
    open_time = int(pd.Timestamp("2026-08-03 12:02:00", tz="UTC").timestamp() * 1_000)
    requested: list[str] = []
    concurrent_klines = Barrier(len(TICKERS), timeout=1.0)

    def public_api(url: str, timeout: float):
        del timeout
        requested.append(url)
        parsed = urllib.parse.urlparse(url)
        if parsed.path == "/fapi/v1/time":
            return JsonResponse({"serverTime": server_ms})
        if parsed.path == "/fapi/v1/ticker/bookTicker":
            return JsonResponse(
                [
                    {
                        "symbol": ticker,
                        "bidPrice": "100",
                        "askPrice": "102",
                        "time": server_ms,
                    }
                    for ticker in TICKERS
                ]
            )
        if parsed.path == "/fapi/v1/klines":
            concurrent_klines.wait()
            return JsonResponse([_kline(open_time, open_time + 59_999)])
        if parsed.path == "/fapi/v1/fundingRate":
            return JsonResponse([])
        raise AssertionError(f"paper mark requested a model-only input: {url}")

    monkeypatch.setattr("urllib.request.urlopen", public_api)

    PublicPaperAdapter(tmp_path, TICKERS).mark(datetime(2026, 8, 3, 12, 1, 59, 999000, tzinfo=UTC))
    assert sum("/fapi/v1/klines" in url for url in requested) == len(TICKERS)
    assert sum("/fapi/v1/fundingRate" in url for url in requested) == len(TICKERS)
    assert not any("/api/v3/klines" in url for url in requested)
    assert not any("premiumIndexKlines" in url for url in requested)
    assert not any("openInterestHist" in url for url in requested)


def test_public_paper_mark_reports_a_missing_required_range_as_unavailable(monkeypatch, tmp_path) -> None:
    ticker = "BTCUSDT"
    server_time = pd.Timestamp("2026-08-03 12:03:30", tz="UTC")
    server_ms = int(server_time.timestamp() * 1_000)
    latest_open = int(pd.Timestamp("2026-08-03 12:02:00", tz="UTC").timestamp() * 1_000)

    def public_api(url: str, timeout: float):
        del timeout
        path = urllib.parse.urlparse(url).path
        if path == "/fapi/v1/time":
            return JsonResponse({"serverTime": server_ms})
        if path == "/fapi/v1/ticker/bookTicker":
            return JsonResponse([{"symbol": ticker, "bidPrice": "100", "askPrice": "102", "time": server_ms}])
        if path == "/fapi/v1/klines":
            return JsonResponse([_kline(latest_open, latest_open + 59_999)])
        raise AssertionError(f"gap must be rejected before optional requests: {url}")

    monkeypatch.setattr("urllib.request.urlopen", public_api)

    with pytest.raises(PublicDataUnavailable, match="range has a gap"):
        PublicPaperAdapter(tmp_path, (ticker,)).mark(datetime(2026, 8, 3, 12, 0, 59, 999000, tzinfo=UTC))


def test_public_paper_mark_retries_one_just_closed_candle_during_publication_lag(monkeypatch, tmp_path) -> None:
    ticker = "BTCUSDT"
    server_time = pd.Timestamp("2026-08-03 12:03:30", tz="UTC")
    server_ms = int(server_time.timestamp() * 1_000)

    def public_api(url: str, timeout: float):
        del timeout
        path = urllib.parse.urlparse(url).path
        if path == "/fapi/v1/time":
            return JsonResponse({"serverTime": server_ms})
        if path == "/fapi/v1/ticker/bookTicker":
            return JsonResponse([{"symbol": ticker, "bidPrice": "100", "askPrice": "102", "time": server_ms}])
        if path == "/fapi/v1/klines":
            return JsonResponse([])
        raise AssertionError(f"publication lag must fail before optional requests: {url}")

    monkeypatch.setattr("urllib.request.urlopen", public_api)

    with pytest.raises(PublicDataUnavailable):
        PublicPaperAdapter(tmp_path, (ticker,)).mark(datetime(2026, 8, 3, 12, 1, 59, 999000, tzinfo=UTC))


def test_public_paper_feed_does_not_invent_a_row_before_the_next_minute_closes(monkeypatch, tmp_path) -> None:
    server_ms = int(pd.Timestamp("2026-08-03 12:03:30", tz="UTC").timestamp() * 1_000)

    def public_api(url: str, timeout: float):
        del timeout
        parsed = urllib.parse.urlparse(url)
        if parsed.path == "/fapi/v1/time":
            return JsonResponse({"serverTime": server_ms})
        if parsed.path == "/fapi/v1/ticker/bookTicker":
            return JsonResponse(
                [
                    {
                        "symbol": ticker,
                        "bidPrice": "100",
                        "askPrice": "102",
                        "time": server_ms,
                    }
                    for ticker in TICKERS
                ]
            )
        if parsed.path in ("/fapi/v1/fundingRate", "/futures/data/openInterestHist"):
            return JsonResponse([])
        raise AssertionError(f"a no-op catch-up must not request candles: {url}")

    monkeypatch.setattr("urllib.request.urlopen", public_api)

    observed = PublicPaperAdapter(tmp_path, TICKERS).observe(datetime(2026, 8, 3, 12, 2, 59, 999000, tzinfo=UTC))

    assert observed.instruments["BTCUSDT"].perpetual.empty
    assert tuple(observed.instruments["BTCUSDT"].perpetual.columns) == BAR_COLUMNS
    assert observed.instruments["BTCUSDT"].spot.empty
    assert observed.instruments["BTCUSDT"].premium.empty


def test_public_paper_feed_rejects_a_long_catchup_without_settled_funding(monkeypatch, tmp_path) -> None:
    ticker = "BTCUSDT"
    server_time = pd.Timestamp("2026-08-03 12:03:30", tz="UTC")
    server_ms = int(server_time.timestamp() * 1_000)
    opens = pd.date_range("2026-08-03 03:00", "2026-08-03 12:02", freq="min", tz="UTC")
    klines = [
        _kline(int(open_time.timestamp() * 1_000), int(open_time.timestamp() * 1_000) + 59_999) for open_time in opens
    ]

    def public_api(url: str, timeout: float):
        del timeout
        path = urllib.parse.urlparse(url).path
        if path == "/fapi/v1/time":
            return JsonResponse({"serverTime": server_ms})
        if path == "/fapi/v1/ticker/bookTicker":
            return JsonResponse([{"symbol": ticker, "bidPrice": "100", "askPrice": "102", "time": server_ms}])
        if path in ("/fapi/v1/klines", "/api/v3/klines", "/fapi/v1/premiumIndexKlines"):
            return JsonResponse(klines)
        if path in ("/fapi/v1/fundingRate", "/futures/data/openInterestHist"):
            return JsonResponse([])
        raise AssertionError(f"unexpected endpoint: {url}")

    monkeypatch.setattr("urllib.request.urlopen", public_api)

    with pytest.raises(PublicDataUnavailable, match="funding"):
        PublicPaperAdapter(tmp_path, (ticker,)).observe(datetime(2026, 8, 3, 3, 0, tzinfo=UTC))


def test_public_paper_feed_rejects_a_stale_midpoint_snapshot(monkeypatch, tmp_path) -> None:
    server_ms = int(pd.Timestamp("2026-08-03 12:03:30", tz="UTC").timestamp() * 1_000)

    def public_api(url: str, timeout: float):
        del timeout
        path = urllib.parse.urlparse(url).path
        if path == "/fapi/v1/time":
            return JsonResponse({"serverTime": server_ms})
        if path == "/fapi/v1/ticker/bookTicker":
            return JsonResponse(
                [
                    {
                        "symbol": ticker,
                        "bidPrice": "100",
                        "askPrice": "102",
                        "time": server_ms - 6_000,
                    }
                    for ticker in TICKERS
                ]
            )
        raise AssertionError("stale quotes must fail before any catch-up requests")

    monkeypatch.setattr("urllib.request.urlopen", public_api)

    with pytest.raises(PublicDataUnavailable, match="stale"):
        PublicPaperAdapter(tmp_path, TICKERS).observe(None)


def test_public_paper_feed_keeps_open_interest_optional_during_an_outage(monkeypatch, tmp_path) -> None:
    server_ms = int(pd.Timestamp("2026-08-03 12:03:30", tz="UTC").timestamp() * 1_000)

    def public_api(url: str, timeout: float):
        del timeout
        path = urllib.parse.urlparse(url).path
        if path == "/fapi/v1/time":
            return JsonResponse({"serverTime": server_ms})
        if path == "/fapi/v1/ticker/bookTicker":
            return JsonResponse(
                [
                    {
                        "symbol": ticker,
                        "bidPrice": "100",
                        "askPrice": "102",
                        "time": server_ms,
                    }
                    for ticker in TICKERS
                ]
            )
        if path == "/fapi/v1/fundingRate":
            return JsonResponse([])
        if path == "/futures/data/openInterestHist":
            raise urllib.error.URLError("optional endpoint unavailable")
        raise AssertionError(f"unexpected endpoint: {url}")

    monkeypatch.setattr("urllib.request.urlopen", public_api)

    observed = PublicPaperAdapter(tmp_path, TICKERS).observe(datetime(2026, 8, 3, 12, 2, 59, 999000, tzinfo=UTC))

    assert observed.instruments["BTCUSDT"].open_interest.empty


def test_optional_inputs_have_masks_but_required_prices_cannot_be_missing() -> None:
    optional = dataset(missing_optional=True)
    optional.instruments["BTCUSDT"].spot = optional.instruments["BTCUSDT"].spot.iloc[:0]

    state = build_market_state(optional)

    assert state.filter(like="open_interest_available").to_numpy().sum() == 0
    assert state.filter(like="premium_available").to_numpy().sum() == 0
    assert state["BTCUSDT_spot_available"].eq(0.0).all()

    invalid = dataset()
    invalid.instruments["BTCUSDT"].perpetual.iloc[10, 0] = np.nan
    with pytest.raises(ValueError, match="execution price"):
        invalid.validate()

    invalid_funding = dataset()
    invalid_funding.instruments["BTCUSDT"].funding.iloc[0] = np.nan
    with pytest.raises(ValueError, match="funding"):
        invalid_funding.validate()

    missing_minute = dataset()
    instrument = missing_minute.instruments["SOLUSDT"]
    instrument.perpetual = instrument.perpetual.drop(instrument.perpetual.index[100])
    with pytest.raises(ValueError, match="Execution-Complete Interval"):
        missing_minute.validate()

    missing_funding = dataset()
    funding = missing_funding.instruments["ETHUSDT"].funding
    missing_funding.instruments["ETHUSDT"].funding = funding.drop(funding.index[1])
    with pytest.raises(ValueError, match="funding cashflow"):
        missing_funding.validate()


def test_common_trading_start_excludes_a_partial_universe() -> None:
    raw = dataset()
    late_start = raw.instruments["SOLUSDT"].perpetual.index[30]
    raw.instruments["SOLUSDT"].perpetual = raw.instruments["SOLUSDT"].perpetual.loc[late_start:]

    raw.validate()

    assert raw.common_trading_start == late_start
    assert all(data.perpetual.index.min() >= late_start for data in raw.execution_complete().values())


def test_decision_state_uses_only_fully_closed_bars_and_is_future_invariant() -> None:
    original = dataset()
    changed = dataset()
    signal_time = pd.Timestamp("2026-01-02 00:00", tz="UTC")
    for instrument in changed.instruments.values():
        instrument.perpetual.loc[signal_time:, "close"] *= 10.0
        instrument.spot.loc[signal_time:, "close"] *= 10.0

    before = build_market_state(original).loc[:signal_time]
    after = build_market_state(changed).loc[:signal_time]

    pd.testing.assert_frame_equal(before, after)
    assert before.index.minute.isin((0, 15, 30, 45)).all()
    assert any(column.endswith("return_1h") for column in before.columns)
    assert any(column.endswith("return_4h") for column in before.columns)
    assert any(column.endswith("return_1d") for column in before.columns)


def test_data_sync_uses_monthly_history_and_only_open_month_daily(tmp_path) -> None:
    requested: list[str] = []

    class RecordingSync(BinanceDataSync):
        def _download_zip(self, url, destination):
            del destination
            requested.append(url)
            return "cached"

    synchronizer = RecordingSync(
        tmp_path,
        load_config("policy.toml"),
        today=lambda: pd.Timestamp("2026-08-03", tz="UTC").date(),
    )

    synchronizer.sync()

    assert any("/monthly/metrics/BTCUSDT/BTCUSDT-metrics-2026-07.zip" in url for url in requested)
    assert any("/monthly/premiumIndexKlines/SOLUSDT/1m/SOLUSDT-1m-2026-07.zip" in url for url in requested)
    assert any("BTCUSDT-1m-2026-08-02.zip" in url for url in requested)
    assert not any("/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2025" in url for url in requested)
