"""Pure average-cost accounting and passive Paper Account benchmarks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import Literal

_ZERO_TOLERANCE = 1e-12


def _require_finite(name: str, value: float) -> float:
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _require_price(name: str, value: float) -> float:
    converted = _require_finite(name, value)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted


@dataclass(frozen=True)
class PositionFillAccounting:
    """Gross position effect of one fill, excluding Transaction Cost."""

    closed_quantity: float
    opened_quantity: float
    realized_pnl: float


@dataclass
class AverageCostPosition:
    """One signed perpetual position using an average Reference Price basis."""

    quantity: float = 0.0
    average_entry: float = 0.0
    realized_pnl: float = 0.0

    def apply_fill(self, *, quantity_delta: float, reference_price: float) -> PositionFillAccounting:
        """Apply an add, reduction, close, or reversal at the Reference Price."""
        delta = _require_finite("fill quantity", quantity_delta)
        price = _require_price("Reference Price", reference_price)
        if abs(delta) <= _ZERO_TOLERANCE:
            return PositionFillAccounting(0.0, 0.0, 0.0)

        prior = self.quantity
        if abs(prior) <= _ZERO_TOLERANCE or prior * delta > 0.0:
            prior_notional = abs(prior) * self.average_entry
            opened = abs(delta)
            self.quantity = prior + delta
            self.average_entry = (prior_notional + opened * price) / abs(self.quantity)
            return PositionFillAccounting(0.0, opened, 0.0)

        closed = min(abs(prior), abs(delta))
        direction = 1.0 if prior > 0.0 else -1.0
        realized = closed * (price - self.average_entry) * direction
        remaining = prior + delta
        if abs(remaining) <= _ZERO_TOLERANCE:
            self.quantity = 0.0
            self.average_entry = 0.0
            opened = 0.0
        elif prior * remaining > 0.0:
            self.quantity = remaining
            opened = 0.0
        else:
            self.quantity = remaining
            self.average_entry = price
            opened = abs(remaining)
        self.realized_pnl += realized
        return PositionFillAccounting(closed, opened, realized)

    def unrealized_pnl(self, mark_price: float) -> float:
        mark = _require_price("mark price", mark_price)
        return self.quantity * (mark - self.average_entry)


@dataclass(frozen=True)
class PositionMetrics:
    side: Literal["long", "short", "flat"]
    quantity: float
    mark: float
    notional: float
    current_weight: float
    target_weight: float | None
    average_entry: float
    realized_pnl: float
    unrealized_pnl: float


@dataclass(frozen=True)
class AccountMetrics:
    starting_equity: float
    current_equity: float
    net_pnl: float
    compounded_net_return: float
    gross_trading_pnl: float
    realized_pnl: float
    unrealized_pnl: float
    transaction_cost: float
    funding_cashflow: float
    turnover_notional: float
    cash_balance: float
    composed_equity: float
    reconciliation_difference: float


@dataclass(frozen=True)
class RiskMetrics:
    current_drawdown: float
    maximum_drawdown: float
    high_water_equity: float
    drawdown_limit: float
    gross_exposure: float
    net_exposure: float
    cash_weight: float
    concentration: float
    instrument_weights: dict[str, float]


@dataclass(frozen=True)
class AccountSnapshot:
    account: AccountMetrics
    risk: RiskMetrics
    positions: dict[str, PositionMetrics]


@dataclass
class AccountAccounting:
    """Presentation accounting reconciled against authoritative Marked Equity."""

    starting_equity: float
    positions: dict[str, AverageCostPosition]
    drawdown_limit: float = 0.20
    transaction_cost: float = 0.0
    funding_cashflow: float = 0.0
    turnover_notional: float = 0.0
    high_water_equity: float = 0.0
    maximum_drawdown: float = 0.0

    @classmethod
    def flat_start(
        cls,
        tickers: Iterable[str],
        *,
        starting_equity: float = 10_000.0,
        drawdown_limit: float = 0.20,
    ) -> AccountAccounting:
        names = tuple(tickers)
        if not names or len(names) != len(set(names)):
            raise ValueError("Trading Universe must contain unique tickers")
        initial = _require_price("starting equity", starting_equity)
        limit = _require_finite("Drawdown Limit", drawdown_limit)
        if not 0.0 < limit < 1.0:
            raise ValueError("Drawdown Limit must be between zero and one")
        return cls(
            starting_equity=initial,
            positions={ticker: AverageCostPosition() for ticker in names},
            drawdown_limit=limit,
            high_water_equity=initial,
        )

    def apply_fill(
        self,
        ticker: str,
        *,
        quantity_delta: float,
        reference_price: float,
        transaction_cost: float,
    ) -> PositionFillAccounting:
        if ticker not in self.positions:
            raise ValueError(f"ticker {ticker!r} is outside the Trading Universe")
        cost = _require_finite("Transaction Cost", transaction_cost)
        if cost < 0.0:
            raise ValueError("Transaction Cost cannot be negative")
        result = self.positions[ticker].apply_fill(
            quantity_delta=quantity_delta,
            reference_price=reference_price,
        )
        self.transaction_cost += cost
        self.turnover_notional += abs(float(quantity_delta) * float(reference_price))
        return result

    def apply_funding(self, cashflow: float) -> None:
        self.funding_cashflow += _require_finite("Funding cashflow", cashflow)

    def mark(
        self,
        mark_prices: Mapping[str, float],
        *,
        target_weights: Mapping[str, float] | None = None,
        authoritative_equity: float | None = None,
    ) -> AccountSnapshot:
        if set(mark_prices) != set(self.positions):
            raise ValueError("mark prices must cover exactly the Trading Universe")
        if target_weights is not None and set(target_weights) != set(self.positions):
            raise ValueError("Target Weights must cover exactly the Trading Universe")
        marks = {ticker: _require_price("mark price", mark_prices[ticker]) for ticker in self.positions}
        realized = sum(position.realized_pnl for position in self.positions.values())
        unrealized = sum(position.unrealized_pnl(marks[ticker]) for ticker, position in self.positions.items())
        gross_trading_pnl = realized + unrealized
        cash_balance = self.starting_equity + realized - self.transaction_cost + self.funding_cashflow
        composed_equity = cash_balance + unrealized
        current_equity = (
            composed_equity
            if authoritative_equity is None
            else _require_finite("authoritative Marked Equity", authoritative_equity)
        )
        reconciliation = current_equity - composed_equity

        self.high_water_equity = max(self.high_water_equity, current_equity)
        drawdown = max(0.0, 1.0 - current_equity / self.high_water_equity) if self.high_water_equity > 0.0 else 0.0
        self.maximum_drawdown = max(self.maximum_drawdown, drawdown)
        if current_equity > 0.0:
            weights = {
                ticker: position.quantity * marks[ticker] / current_equity
                for ticker, position in self.positions.items()
            }
        else:
            weights = dict.fromkeys(self.positions, 0.0)
        gross_exposure = sum(abs(weight) for weight in weights.values())
        net_exposure = sum(weights.values())

        position_metrics: dict[str, PositionMetrics] = {}
        for ticker, position in self.positions.items():
            if position.quantity > _ZERO_TOLERANCE:
                side: Literal["long", "short", "flat"] = "long"
            elif position.quantity < -_ZERO_TOLERANCE:
                side = "short"
            else:
                side = "flat"
            position_metrics[ticker] = PositionMetrics(
                side=side,
                quantity=position.quantity,
                mark=marks[ticker],
                notional=abs(position.quantity * marks[ticker]),
                current_weight=weights[ticker],
                target_weight=float(target_weights[ticker]) if target_weights is not None else None,
                average_entry=position.average_entry,
                realized_pnl=position.realized_pnl,
                unrealized_pnl=position.unrealized_pnl(marks[ticker]),
            )

        return AccountSnapshot(
            account=AccountMetrics(
                starting_equity=self.starting_equity,
                current_equity=current_equity,
                net_pnl=current_equity - self.starting_equity,
                compounded_net_return=current_equity / self.starting_equity - 1.0,
                gross_trading_pnl=gross_trading_pnl,
                realized_pnl=realized,
                unrealized_pnl=unrealized,
                transaction_cost=self.transaction_cost,
                funding_cashflow=self.funding_cashflow,
                turnover_notional=self.turnover_notional,
                cash_balance=cash_balance,
                composed_equity=composed_equity,
                reconciliation_difference=reconciliation,
            ),
            risk=RiskMetrics(
                current_drawdown=drawdown,
                maximum_drawdown=self.maximum_drawdown,
                high_water_equity=self.high_water_equity,
                drawdown_limit=self.drawdown_limit,
                gross_exposure=gross_exposure,
                net_exposure=net_exposure,
                cash_weight=1.0 - gross_exposure,
                concentration=max((abs(weight) for weight in weights.values()), default=0.0),
                instrument_weights=weights,
            ),
            positions=position_metrics,
        )


@dataclass(frozen=True)
class BenchmarkMark:
    cash_equity: float
    equal_weight_equity: float
    equal_weight_transaction_cost: float
    equal_weight_funding_cashflow: float
    reconstructed: bool


@dataclass
class PassiveBenchmarks:
    """Cash and passive equal-weight-long projections from one Flat Start."""

    starting_equity: float
    equal_weight_quantities: dict[str, float]
    entry_prices: dict[str, float]
    equal_weight_transaction_cost: float
    equal_weight_funding_cashflow: float = 0.0

    @classmethod
    def flat_start(
        cls,
        tickers: Iterable[str],
        mark_prices: Mapping[str, float],
        *,
        starting_equity: float = 10_000.0,
        transaction_cost_rate: float = 0.0007,
    ) -> PassiveBenchmarks:
        names = tuple(tickers)
        if not names or len(names) != len(set(names)):
            raise ValueError("Trading Universe must contain unique tickers")
        if set(mark_prices) != set(names):
            raise ValueError("initial marks must cover exactly the Trading Universe")
        initial = _require_price("starting equity", starting_equity)
        rate = _require_finite("Transaction Cost rate", transaction_cost_rate)
        if rate < 0.0:
            raise ValueError("Transaction Cost rate cannot be negative")
        entries = {ticker: _require_price("initial mark", mark_prices[ticker]) for ticker in names}
        post_cost_equity = initial / (1.0 + rate)
        allocation = post_cost_equity / len(names)
        return cls(
            starting_equity=initial,
            equal_weight_quantities={ticker: allocation / entries[ticker] for ticker in names},
            entry_prices=entries,
            equal_weight_transaction_cost=initial - post_cost_equity,
        )

    def apply_funding(self, funding_rates: Mapping[str, float], funding_mark_prices: Mapping[str, float]) -> float:
        unknown = set(funding_rates) - set(self.equal_weight_quantities)
        if unknown:
            raise ValueError(f"funding contains tickers outside the Trading Universe: {sorted(unknown)}")
        missing_marks = set(funding_rates) - set(funding_mark_prices)
        if missing_marks:
            raise ValueError(f"funding mark prices are missing for {sorted(missing_marks)}")
        cashflow = -sum(
            self.equal_weight_quantities[ticker]
            * _require_price("funding mark price", funding_mark_prices[ticker])
            * _require_finite("Funding rate", rate)
            for ticker, rate in funding_rates.items()
        )
        self.equal_weight_funding_cashflow += cashflow
        return cashflow

    def mark(self, mark_prices: Mapping[str, float], *, reconstructed: bool = False) -> BenchmarkMark:
        if set(mark_prices) != set(self.equal_weight_quantities):
            raise ValueError("benchmark marks must cover exactly the Trading Universe")
        gross_pnl = sum(
            quantity * (_require_price("benchmark mark", mark_prices[ticker]) - self.entry_prices[ticker])
            for ticker, quantity in self.equal_weight_quantities.items()
        )
        return BenchmarkMark(
            cash_equity=self.starting_equity,
            equal_weight_equity=(
                self.starting_equity
                - self.equal_weight_transaction_cost
                + gross_pnl
                + self.equal_weight_funding_cashflow
            ),
            equal_weight_transaction_cost=self.equal_weight_transaction_cost,
            equal_weight_funding_cashflow=self.equal_weight_funding_cashflow,
            reconstructed=reconstructed,
        )


class ActivityOutcome(StrEnum):
    EXECUTABLE_PORTFOLIO_CHANGE = "executable_portfolio_change"
    BELOW_THRESHOLD_DECISION = "below_threshold_decision"
    UNCHANGED_TARGET_DECISION = "unchanged_target_decision"
    INSTRUMENT_FILL = "instrument_fill"
    MISSED_EXECUTION = "missed_execution"
    OPERATOR_INTERVENTION = "operator_intervention"


@dataclass(frozen=True)
class ActivityMetrics:
    decisions: int
    executable_portfolio_changes: int
    instrument_fills: int
    below_threshold_decisions: int
    unchanged_target_decisions: int
    missed_executions: int
    operator_interventions: int


def activity_metrics(outcomes: Iterable[ActivityOutcome]) -> ActivityMetrics:
    """Count mutually exclusive decision outcomes and material account events."""
    counts = Counter(outcomes)
    executable = counts[ActivityOutcome.EXECUTABLE_PORTFOLIO_CHANGE]
    below_threshold = counts[ActivityOutcome.BELOW_THRESHOLD_DECISION]
    unchanged = counts[ActivityOutcome.UNCHANGED_TARGET_DECISION]
    return ActivityMetrics(
        decisions=executable + below_threshold + unchanged,
        executable_portfolio_changes=executable,
        instrument_fills=counts[ActivityOutcome.INSTRUMENT_FILL],
        below_threshold_decisions=below_threshold,
        unchanged_target_decisions=unchanged,
        missed_executions=counts[ActivityOutcome.MISSED_EXECUTION],
        operator_interventions=counts[ActivityOutcome.OPERATOR_INTERVENTION],
    )


@dataclass
class HoldBenchmark:
    starting_equity: float
    quantities: dict[str, float]
    entry_marks: dict[str, float]
    funding_cashflow: float = 0.0
    high_water: float = 0.0
    maximum_drawdown: float = 0.0

    @classmethod
    def start(cls, equity: float, quantities: Mapping[str, float], marks: Mapping[str, float]) -> HoldBenchmark:
        return cls(equity, dict(quantities), dict(marks), high_water=equity)

    def apply_funding(self, rates: Mapping[str, float], marks: Mapping[str, float]) -> None:
        self.funding_cashflow -= sum(self.quantities[t] * marks[t] * rate for t, rate in rates.items())

    def mark(self, marks: Mapping[str, float]) -> dict[str, float]:
        equity = (
            self.starting_equity
            + self.funding_cashflow
            + sum(q * (marks[t] - self.entry_marks[t]) for t, q in self.quantities.items())
        )
        self.high_water = max(self.high_water, equity)
        drawdown = 1.0 - equity / self.high_water if self.high_water > 0 else 0.0
        self.maximum_drawdown = max(self.maximum_drawdown, drawdown)
        return {
            "equity": equity,
            "starting_equity": self.starting_equity,
            "net_pnl": equity - self.starting_equity,
            "compounded_net_return": equity / self.starting_equity - 1.0,
            "funding": self.funding_cashflow,
            "maximum_drawdown": self.maximum_drawdown,
            "gross_exposure": sum(abs(q * marks[t]) for t, q in self.quantities.items()) / equity
            if equity > 0
            else 0.0,
        }
