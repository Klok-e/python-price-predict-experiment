from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netgrowth.market_data import (
    CanonicalDataset,
    InMemoryMarketData,
    InstrumentData,
    build_market_state,
)

TICKERS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")


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


def test_optional_inputs_have_masks_but_required_prices_cannot_be_missing() -> None:
    optional = dataset(missing_optional=True)

    state = build_market_state(optional)

    assert state.filter(like="open_interest_available").to_numpy().sum() == 0
    assert state.filter(like="premium_available").to_numpy().sum() == 0

    invalid = dataset()
    invalid.instruments["BTCUSDT"].perpetual.iloc[10, 0] = np.nan
    with pytest.raises(ValueError, match="execution price"):
        invalid.validate()

    invalid_funding = dataset()
    invalid_funding.instruments["BTCUSDT"].funding.iloc[0] = np.nan
    with pytest.raises(ValueError, match="funding"):
        invalid_funding.validate()


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
