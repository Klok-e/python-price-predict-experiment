from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from netgrowth.simulation import (
    MarketMinute,
    SimulationConfig,
    SimulationState,
    advance_simulation,
    simulate,
)

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
    post_cost_equity = 10_000.0 / 1.0007
    assert result.transaction_cost == pytest.approx(10_000.0 - post_cost_equity)
    assert result.final_equity == pytest.approx(post_cost_equity)
    assert result.executable_portfolio_changes == 1
    assert result.max_gross_exposure == pytest.approx(1.0)
    assert result.max_instrument_exposure == pytest.approx(
        {"BTCUSDT": 0.5, "ETHUSDT": 0.5, "BNBUSDT": 0.0, "SOLUSDT": 0.0}
    )
    assert result.equity[-1].gross_exposure == pytest.approx(result.max_gross_exposure)
    assert all(abs(trade.quantity * 100.0 / result.final_equity) <= 0.5 + 1e-12 for trade in result.trades)


def test_historical_fill_uses_reference_as_the_new_position_basis() -> None:
    prices = {ticker: 100.0 for ticker in TICKERS}
    fill_prices = {**prices, "BTCUSDT": 200.0}
    target = {**FLAT, "BTCUSDT": 0.5}
    start = datetime(2026, 1, 1, tzinfo=UTC)

    result = simulate(
        [
            MarketMinute(start, prices, historical_open=prices, target_weights=target),
            MarketMinute(start + timedelta(minutes=1), prices, historical_open=fill_prices),
            MarketMinute(start + timedelta(minutes=2), fill_prices, historical_open=fill_prices),
        ],
        SimulationConfig(tickers=TICKERS),
    )

    assert result.trades[0].reference_price == 200.0
    post_cost_equity = 10_000.0 / (1.0 + 0.5 * 0.0007)
    assert result.final_equity == pytest.approx(post_cost_equity)
    assert result.max_drawdown == pytest.approx(1.0 - post_cost_equity / 10_000.0)


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
    assert result.executable_portfolio_changes == 1


def test_signal_time_qualified_change_fills_after_price_drift_reduces_fill_turnover() -> None:
    prices = {ticker: 100.0 for ticker in TICKERS}
    doubled = {**prices, "BTCUSDT": 200.0}

    result = simulate(
        [
            minute(0, prices, target={**FLAT, "BTCUSDT": 0.02}),
            minute(1, prices),
            minute(2, prices, target={**FLAT, "BTCUSDT": 0.031}),
            minute(3, doubled),
        ],
        SimulationConfig(tickers=TICKERS),
    )

    assert len(result.trades) == 2
    assert result.trades[-1].target_weight == pytest.approx(0.031)
    assert result.executable_portfolio_changes == 2


def test_signal_time_below_threshold_change_does_not_create_pending_fill() -> None:
    prices = {ticker: 100.0 for ticker in TICKERS}
    _, state = advance_simulation(
        [minute(0, prices, target={**FLAT, "BTCUSDT": 0.005})],
        SimulationConfig(tickers=TICKERS),
    )

    assert state.pending is None


def test_legacy_pending_payload_uses_fill_time_threshold_only_while_draining() -> None:
    prices = {ticker: 100.0 for ticker in TICKERS}
    legacy_state = SimulationState.from_payload(
        {
            "quantities": FLAT,
            "equity": 10_000.0,
            "high_water": 10_000.0,
            "max_drawdown": 0.0,
            "turnover_notional": 0.0,
            "transaction_cost": 0.0,
            "funding_cashflow": 0.0,
            "executable_portfolio_changes": 0,
            "previous_timestamp": "2026-01-01T00:00:00+00:00",
            "previous_marks": prices,
            "pending": {
                "eligible_at": "2026-01-01T00:01:00+00:00",
                "target_weights": {**FLAT, "BTCUSDT": 0.005},
                "forced": False,
            },
            "risk_stop_time": None,
            "flattened": False,
        }
    )

    result, state = advance_simulation([minute(1, prices)], SimulationConfig(tickers=TICKERS), legacy_state)

    assert result.trades == ()
    assert state.pending is None


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


def test_settled_funding_uses_its_event_mark_price_exactly_once() -> None:
    prices = {ticker: 100.0 for ticker in TICKERS}
    target = {**FLAT, "BTCUSDT": 0.5}
    event_prices = {**prices, "BTCUSDT": 200.0}

    result = simulate(
        [
            minute(0, prices, target=target),
            minute(1, prices),
            MarketMinute(
                timestamp=datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
                mark_prices=event_prices,
                historical_open=event_prices,
                funding_rates={"BTCUSDT": 0.001},
                funding_mark_prices={"BTCUSDT": 150.0},
            ),
        ],
        SimulationConfig(tickers=TICKERS),
    )

    post_cost_equity = 10_000.0 / (1.0 + 0.5 * 0.0007)
    assert result.funding_cashflow == pytest.approx(-(0.5 * post_cost_equity / 100.0) * 150.0 * 0.001)

    with pytest.raises(ValueError, match="funding mark"):
        simulate(
            [
                MarketMinute(
                    timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                    mark_prices=prices,
                    bid=prices,
                    ask=prices,
                    funding_rates={"BTCUSDT": 0.001},
                )
            ],
            SimulationConfig(tickers=TICKERS, mode="paper"),
        )


def test_paper_risk_stop_waits_for_its_own_delayed_fill() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    prices = {ticker: 100.0 for ticker in TICKERS}
    collapsed = {**prices, "BTCUSDT": 40.0}
    target = {**FLAT, "BTCUSDT": 0.5}

    def paper_row(offset: int, marks: dict[str, float], *, desired=None) -> MarketMinute:
        timestamp = start + timedelta(minutes=offset)
        return MarketMinute(
            timestamp=timestamp,
            mark_prices=marks,
            bid=marks,
            ask=marks,
            quote_exchange_times=dict.fromkeys(TICKERS, timestamp),
            target_weights=desired,
        )

    result = simulate(
        [
            paper_row(0, prices, desired=target),
            paper_row(1, prices, desired={**FLAT, "BTCUSDT": 0.4}),
            paper_row(2, collapsed),
            paper_row(3, collapsed),
        ],
        SimulationConfig(tickers=TICKERS, mode="paper"),
    )

    assert result.risk_stop_time == start + timedelta(minutes=2)
    assert result.trades[-1].quantity < 0.0
    assert result.trades[-1].timestamp == start + timedelta(minutes=3)


def test_paper_fill_uses_midpoint_and_records_spread_without_double_charging() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    target = {**FLAT, "BTCUSDT": 0.5}
    prices = {ticker: 100.0 for ticker in TICKERS}
    quotes_bid = {ticker: 99.0 for ticker in TICKERS}
    quotes_ask = {ticker: 101.0 for ticker in TICKERS}
    quote_time = timestamp + timedelta(seconds=59)
    rows = [
        MarketMinute(
            timestamp=timestamp,
            mark_prices=prices,
            bid=quotes_bid,
            ask=quotes_ask,
            quote_exchange_times=dict.fromkeys(TICKERS, quote_time),
            quote_observed_at=quote_time,
            target_weights=target,
        ),
        MarketMinute(
            timestamp=timestamp + timedelta(minutes=1),
            mark_prices=prices,
            bid=quotes_bid,
            ask=quotes_ask,
            quote_exchange_times=dict.fromkeys(TICKERS, quote_time + timedelta(minutes=1)),
            quote_observed_at=quote_time + timedelta(minutes=1),
        ),
    ]

    result = simulate(rows, SimulationConfig(tickers=TICKERS, mode="paper"))

    assert result.trades[0].reference_price == 100.0
    assert result.trades[0].bid == 99.0
    assert result.trades[0].ask == 101.0
    assert result.trades[0].quote_exchange_time == quote_time + timedelta(minutes=1)
    assert result.trades[0].quote_observed_at == quote_time + timedelta(minutes=1)
    assert result.trades[0].timestamp == quote_time + timedelta(minutes=1)
    assert result.transaction_cost == pytest.approx(10_000.0 - 10_000.0 / (1.0 + 0.5 * 0.0007))


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


def test_execution_boundary_rejects_weights_outside_hard_exposure_limits() -> None:
    prices = {ticker: 100.0 for ticker in TICKERS}

    with pytest.raises(ValueError, match="concentration"):
        simulate(
            [minute(0, prices, target={**FLAT, "BTCUSDT": 0.5001})],
            SimulationConfig(tickers=TICKERS),
        )
    with pytest.raises(ValueError, match="Gross Exposure"):
        simulate(
            [minute(0, prices, target={"BTCUSDT": 0.4, "ETHUSDT": 0.4, "BNBUSDT": 0.3, "SOLUSDT": 0.0})],
            SimulationConfig(tickers=TICKERS),
        )


def test_incremental_paper_session_preserves_portfolio_and_pending_fill() -> None:
    prices = {ticker: 100.0 for ticker in TICKERS}
    target = {**FLAT, "BTCUSDT": 0.5}
    config = SimulationConfig(tickers=TICKERS, mode="paper")

    first, state = advance_simulation(
        [
            MarketMinute(
                timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                mark_prices=prices,
                bid=prices,
                ask=prices,
                target_weights=target,
            )
        ],
        config,
    )
    second, state = advance_simulation(
        [
            MarketMinute(
                timestamp=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
                mark_prices=prices,
                bid=prices,
                ask=prices,
                quote_exchange_times=dict.fromkeys(TICKERS, datetime(2026, 1, 1, 0, 1, tzinfo=UTC)),
                quote_observed_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
            ),
            MarketMinute(
                timestamp=datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
                mark_prices={**prices, "BTCUSDT": 101.0},
                bid=prices,
                ask=prices,
            ),
        ],
        config,
        state,
    )

    assert first.trades == ()
    assert len(second.trades) == 1
    post_cost_equity = 10_000.0 / (1.0 + 0.5 * 0.0007)
    quantity = 0.5 * post_cost_equity / 100.0
    assert state.quantities["BTCUSDT"] == pytest.approx(quantity)
    assert second.final_equity == pytest.approx(post_cost_equity + quantity)
    assert second.executable_portfolio_changes == 1
