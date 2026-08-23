from __future__ import annotations

import pytest

from netgrowth.accounting import (
    AccountAccounting,
    ActivityOutcome,
    AverageCostPosition,
    PassiveBenchmarks,
    activity_metrics,
)


def test_average_cost_add_reduction_and_close_keep_gross_pnl_separate() -> None:
    position = AverageCostPosition()

    position.apply_fill(quantity_delta=10.0, reference_price=100.0)
    position.apply_fill(quantity_delta=5.0, reference_price=110.0)

    assert position.quantity == pytest.approx(15.0)
    assert position.average_entry == pytest.approx(103.33333333333333)
    assert position.realized_pnl == pytest.approx(0.0)

    reduction = position.apply_fill(quantity_delta=-6.0, reference_price=120.0)
    assert reduction.realized_pnl == pytest.approx(100.0)
    assert position.quantity == pytest.approx(9.0)
    assert position.average_entry == pytest.approx(103.33333333333333)

    close = position.apply_fill(quantity_delta=-9.0, reference_price=90.0)
    assert close.realized_pnl == pytest.approx(-120.0)
    assert position.quantity == pytest.approx(0.0)
    assert position.average_entry == pytest.approx(0.0)
    assert position.realized_pnl == pytest.approx(-20.0)


def test_average_cost_reversals_close_old_side_then_open_residual_at_fill() -> None:
    position = AverageCostPosition()
    position.apply_fill(quantity_delta=10.0, reference_price=100.0)

    long_to_short = position.apply_fill(quantity_delta=-15.0, reference_price=120.0)

    assert long_to_short.closed_quantity == pytest.approx(10.0)
    assert long_to_short.opened_quantity == pytest.approx(5.0)
    assert long_to_short.realized_pnl == pytest.approx(200.0)
    assert position.quantity == pytest.approx(-5.0)
    assert position.average_entry == pytest.approx(120.0)

    short_to_long = position.apply_fill(quantity_delta=8.0, reference_price=100.0)

    assert short_to_long.closed_quantity == pytest.approx(5.0)
    assert short_to_long.opened_quantity == pytest.approx(3.0)
    assert short_to_long.realized_pnl == pytest.approx(100.0)
    assert position.quantity == pytest.approx(3.0)
    assert position.average_entry == pytest.approx(100.0)
    assert position.realized_pnl == pytest.approx(300.0)


def test_account_metrics_reconcile_gross_pnl_cost_and_funding_to_marked_equity() -> None:
    account = AccountAccounting.flat_start(("BTCUSDT",), starting_equity=10_000.0)
    account.apply_fill("BTCUSDT", quantity_delta=10.0, reference_price=100.0, transaction_cost=2.0)
    account.apply_funding(-5.0)

    snapshot = account.mark({"BTCUSDT": 110.0}, authoritative_equity=10_090.0)

    assert snapshot.positions["BTCUSDT"].unrealized_pnl == pytest.approx(100.0)
    assert snapshot.account.gross_trading_pnl == pytest.approx(100.0)
    assert snapshot.account.transaction_cost == pytest.approx(2.0)
    assert snapshot.account.funding_cashflow == pytest.approx(-5.0)
    assert snapshot.account.composed_equity == pytest.approx(10_093.0)
    assert snapshot.account.current_equity == pytest.approx(10_090.0)
    assert snapshot.account.reconciliation_difference == pytest.approx(-3.0)
    assert snapshot.account.net_pnl == pytest.approx(90.0)
    assert snapshot.account.compounded_net_return == pytest.approx(0.009)


def test_operator_forced_close_realizes_average_cost_and_charges_transaction_cost() -> None:
    account = AccountAccounting.flat_start(("BTCUSDT",), starting_equity=10_000.0)
    account.apply_fill("BTCUSDT", quantity_delta=10.0, reference_price=100.0, transaction_cost=0.70)
    account.apply_fill("BTCUSDT", quantity_delta=5.0, reference_price=110.0, transaction_cost=0.385)

    assert account.positions["BTCUSDT"].average_entry == pytest.approx(103.33333333333333)

    operator_close = account.apply_fill(
        "BTCUSDT",
        quantity_delta=-15.0,
        reference_price=120.0,
        transaction_cost=1.26,
    )
    snapshot = account.mark({"BTCUSDT": 120.0})

    assert operator_close.realized_pnl == pytest.approx(250.0)
    assert snapshot.positions["BTCUSDT"].quantity == pytest.approx(0.0)
    assert snapshot.positions["BTCUSDT"].average_entry == pytest.approx(0.0)
    assert snapshot.account.gross_trading_pnl == pytest.approx(250.0)
    assert snapshot.account.transaction_cost == pytest.approx(2.345)
    assert snapshot.account.current_equity == pytest.approx(10_247.655)


def test_risk_metrics_follow_authoritative_minute_marks() -> None:
    account = AccountAccounting.flat_start(("BTCUSDT",), starting_equity=1_000.0)
    account.apply_fill("BTCUSDT", quantity_delta=10.0, reference_price=100.0, transaction_cost=0.0)

    first = account.mark({"BTCUSDT": 100.0})
    second = account.mark({"BTCUSDT": 80.0})

    assert first.risk.gross_exposure == pytest.approx(1.0)
    assert first.risk.net_exposure == pytest.approx(1.0)
    assert first.risk.cash_weight == pytest.approx(0.0)
    assert first.risk.concentration == pytest.approx(1.0)
    assert second.account.current_equity == pytest.approx(800.0)
    assert second.risk.current_drawdown == pytest.approx(0.20)
    assert second.risk.maximum_drawdown == pytest.approx(0.20)
    assert second.risk.high_water_equity == pytest.approx(1_000.0)


def test_cash_and_equal_weight_benchmarks_include_initial_cost_funding_and_downtime_marks() -> None:
    tickers = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
    initial_marks = dict.fromkeys(tickers, 100.0)
    benchmarks = PassiveBenchmarks.flat_start(
        tickers,
        initial_marks,
        starting_equity=10_000.0,
        transaction_cost_rate=0.0007,
    )

    opening = benchmarks.mark(initial_marks)
    post_cost_equity = 10_000.0 / 1.0007
    assert opening.cash_equity == pytest.approx(10_000.0)
    assert opening.equal_weight_equity == pytest.approx(post_cost_equity)
    assert opening.equal_weight_transaction_cost == pytest.approx(10_000.0 - post_cost_equity)
    assert all(
        quantity == pytest.approx(post_cost_equity * 0.25 / 100.0)
        for quantity in benchmarks.equal_weight_quantities.values()
    )

    downtime_marks = {**initial_marks, "BTCUSDT": 200.0}
    funding = benchmarks.apply_funding({"BTCUSDT": 0.001}, downtime_marks)
    reconstructed = benchmarks.mark(downtime_marks, reconstructed=True)

    btc_quantity = post_cost_equity * 0.25 / 100.0
    assert funding == pytest.approx(-btc_quantity * 200.0 * 0.001)
    assert reconstructed.equal_weight_equity == pytest.approx(post_cost_equity + btc_quantity * 100.0 + funding)
    assert reconstructed.equal_weight_funding_cashflow == pytest.approx(funding)
    assert reconstructed.reconstructed is True


def test_activity_metrics_distinguish_decisions_execution_and_interventions() -> None:
    metrics = activity_metrics(
        [
            ActivityOutcome.EXECUTABLE_PORTFOLIO_CHANGE,
            ActivityOutcome.EXECUTABLE_PORTFOLIO_CHANGE,
            ActivityOutcome.BELOW_THRESHOLD_DECISION,
            ActivityOutcome.UNCHANGED_TARGET_DECISION,
            ActivityOutcome.INSTRUMENT_FILL,
            ActivityOutcome.INSTRUMENT_FILL,
            ActivityOutcome.MISSED_EXECUTION,
            ActivityOutcome.OPERATOR_INTERVENTION,
        ]
    )

    assert metrics.decisions == 4
    assert metrics.executable_portfolio_changes == 2
    assert metrics.instrument_fills == 2
    assert metrics.below_threshold_decisions == 1
    assert metrics.unchanged_target_decisions == 1
    assert metrics.missed_executions == 1
    assert metrics.operator_interventions == 1
