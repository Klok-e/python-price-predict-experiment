from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from netgrowth.simulation import MarketMinute, SimulationConfig, simulate

TICKERS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
FLAT = dict.fromkeys(TICKERS, 0.0)


def minute(
    offset: int,
    prices: dict[str, float],
    *,
    target: dict[str, float] | None = None,
    funding: dict[str, float] | None = None,
) -> MarketMinute:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=offset)
    return MarketMinute(
        timestamp=timestamp,
        mark_prices=prices,
        historical_open=prices,
        target_weights=target,
        funding_rates=funding or {},
    )


def test_historical_fill_waits_sixty_seconds_and_charges_adverse_cost_once() -> None:
    prices = {ticker: 100.0 for ticker in TICKERS}
    target = {**FLAT, "BTCUSDT": 0.5, "ETHUSDT": -0.5}

    result = simulate(
        [minute(0, prices, target=target), minute(1, prices), minute(2, prices)],
        SimulationConfig(tickers=TICKERS),
    )

    assert {trade.timestamp for trade in result.trades} == {datetime(2026, 1, 1, 0, 1, tzinfo=UTC)}
    assert [trade.effective_fill for trade in result.trades] == pytest.approx([100.07, 99.93])
    assert result.transaction_cost == pytest.approx(7.0)
    assert result.final_equity == pytest.approx(9_993.0)
    assert result.qualifying_portfolio_changes == 1


def test_sub_one_percent_changes_accumulate_until_the_full_change_executes() -> None:
    prices = {ticker: 100.0 for ticker in TICKERS}

    result = simulate(
        [
            minute(0, prices, target={**FLAT, "BTCUSDT": 0.005}),
            minute(1, prices),
            minute(15, prices, target={**FLAT, "BTCUSDT": 0.011}),
            minute(16, prices),
        ],
        SimulationConfig(tickers=TICKERS),
    )

    assert len(result.trades) == 1
    assert result.trades[0].target_weight == pytest.approx(0.011)
    assert result.qualifying_portfolio_changes == 1


def test_funding_is_applied_to_the_signed_position_at_its_event_time() -> None:
    prices = {ticker: 100.0 for ticker in TICKERS}
    target = {**FLAT, "BTCUSDT": 0.5, "ETHUSDT": -0.5}

    result = simulate(
        [
            minute(0, prices, target=target),
            minute(1, prices),
            minute(2, prices, funding={"BTCUSDT": 0.001, "ETHUSDT": 0.001}),
        ],
        SimulationConfig(tickers=TICKERS),
    )

    assert result.funding_cashflow == pytest.approx(0.0)


def test_paper_fill_uses_midpoint_and_records_spread_without_double_charging() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    target = {**FLAT, "BTCUSDT": 0.5}
    prices = {ticker: 100.0 for ticker in TICKERS}
    quotes_bid = {ticker: 99.0 for ticker in TICKERS}
    quotes_ask = {ticker: 101.0 for ticker in TICKERS}
    rows = [
        MarketMinute(
            timestamp=timestamp,
            mark_prices=prices,
            bid=quotes_bid,
            ask=quotes_ask,
            target_weights=target,
        ),
        MarketMinute(
            timestamp=timestamp + timedelta(minutes=1),
            mark_prices=prices,
            bid=quotes_bid,
            ask=quotes_ask,
        ),
    ]

    result = simulate(rows, SimulationConfig(tickers=TICKERS, mode="paper"))

    assert result.trades[0].reference_price == 100.0
    assert result.trades[0].bid == 99.0
    assert result.trades[0].ask == 101.0
    assert result.transaction_cost == pytest.approx(3.5)


def test_drawdown_breach_flattens_after_latency_and_stops_later_decisions() -> None:
    initial = {ticker: 100.0 for ticker in TICKERS}
    target = {**FLAT, "BTCUSDT": 0.5}
    collapsed = {**initial, "BTCUSDT": 50.0}

    result = simulate(
        [
            minute(0, initial, target=target),
            minute(1, initial),
            minute(2, collapsed),
            minute(3, collapsed),
            minute(15, collapsed, target={**FLAT, "ETHUSDT": 0.5}),
            minute(16, collapsed),
        ],
        SimulationConfig(tickers=TICKERS, drawdown_limit=0.20),
    )

    assert result.risk_stop_triggered is True
    assert result.risk_stop_time == datetime(2026, 1, 1, 0, 2, tzinfo=UTC)
    assert result.trades[-1].timestamp == datetime(2026, 1, 1, 0, 3, tzinfo=UTC)
    assert result.trades[-1].ticker == "BTCUSDT"
    assert result.trades[-1].target_weight == 0.0
    assert all(trade.ticker != "ETHUSDT" for trade in result.trades)
    assert result.max_drawdown > 0.20
