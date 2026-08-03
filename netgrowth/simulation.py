"""Pure Causal Replay of portfolio execution, funding, and the Risk Stop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal


@dataclass(frozen=True)
class MarketMinute:
    timestamp: datetime
    mark_prices: dict[str, float]
    historical_open: dict[str, float] | None = None
    bid: dict[str, float] | None = None
    ask: dict[str, float] | None = None
    funding_rates: dict[str, float] = field(default_factory=dict)
    target_weights: dict[str, float] | None = None


@dataclass(frozen=True)
class SimulationConfig:
    tickers: tuple[str, ...]
    initial_equity: float = 10_000.0
    decision_latency_seconds: int = 60
    transaction_cost_rate: float = 0.0007
    minimum_turnover: float = 0.01
    drawdown_limit: float = 0.20
    mode: Literal["historical", "paper"] = "historical"


@dataclass(frozen=True)
class Trade:
    timestamp: datetime
    ticker: str
    quantity: float
    reference_price: float
    effective_fill: float
    target_weight: float
    bid: float | None = None
    ask: float | None = None


@dataclass(frozen=True)
class EquityPoint:
    timestamp: datetime
    equity: float
    drawdown: float


@dataclass(frozen=True)
class ReplayResult:
    initial_equity: float
    final_equity: float
    max_drawdown: float
    transaction_cost: float
    funding_cashflow: float
    qualifying_portfolio_changes: int
    risk_stop_triggered: bool
    risk_stop_time: datetime | None
    trades: tuple[Trade, ...]
    equity: tuple[EquityPoint, ...]

    @property
    def compounded_net_return(self) -> float:
        return self.final_equity / self.initial_equity - 1.0


def _validate_minute(row: MarketMinute, config: SimulationConfig) -> None:
    if row.timestamp.tzinfo is None or row.timestamp.utcoffset() is None:
        raise ValueError("UTC Market Time must be timezone-aware")
    missing_marks = set(config.tickers) - row.mark_prices.keys()
    if missing_marks:
        raise ValueError(f"missing mark prices for {sorted(missing_marks)}")
    if config.mode == "historical":
        missing = set(config.tickers) - (row.historical_open or {}).keys()
        if missing:
            raise ValueError(f"missing historical Reference Price for {sorted(missing)}")
    elif set(config.tickers) - (row.bid or {}).keys() or set(config.tickers) - (row.ask or {}).keys():
        raise ValueError("paper Reference Price requires bid and ask for every ticker")


def _reference_price(row: MarketMinute, ticker: str, mode: str) -> float:
    if mode == "historical":
        return (row.historical_open or {})[ticker]
    return ((row.bid or {})[ticker] + (row.ask or {})[ticker]) / 2.0


def simulate(rows: list[MarketMinute], config: SimulationConfig) -> ReplayResult:
    """Replay ordered minute marks; all mutable state is local to this call."""
    if not rows:
        raise ValueError("Causal Replay requires at least one minute")
    ordered = sorted(rows, key=lambda row: row.timestamp)
    if ordered != rows or len({row.timestamp for row in rows}) != len(rows):
        raise ValueError("market minutes must be unique and ordered")

    quantities = dict.fromkeys(config.tickers, 0.0)
    previous_marks: dict[str, float] | None = None
    equity = config.initial_equity
    high_water = equity
    max_drawdown = 0.0
    total_cost = 0.0
    total_funding = 0.0
    changes = 0
    trades: list[Trade] = []
    curve: list[EquityPoint] = []
    pending: tuple[datetime, dict[str, float], bool] | None = None
    risk_stop_time: datetime | None = None
    flattened = False

    for row in ordered:
        _validate_minute(row, config)

        if previous_marks is not None:
            equity += sum(
                quantities[ticker] * (row.mark_prices[ticker] - previous_marks[ticker]) for ticker in config.tickers
            )
        funding = -sum(
            quantities[ticker] * row.mark_prices[ticker] * row.funding_rates.get(ticker, 0.0)
            for ticker in config.tickers
        )
        equity += funding
        total_funding += funding

        high_water = max(high_water, equity)
        drawdown = 1.0 - equity / high_water
        if drawdown > config.drawdown_limit and risk_stop_time is None:
            risk_stop_time = row.timestamp
            pending = (
                row.timestamp + timedelta(seconds=config.decision_latency_seconds),
                dict.fromkeys(config.tickers, 0.0),
                True,
            )

        if pending is not None and row.timestamp >= pending[0]:
            target, forced = pending[1], pending[2]
            references = {ticker: _reference_price(row, ticker, config.mode) for ticker in config.tickers}
            deltas = {
                ticker: target[ticker] * equity / references[ticker] - quantities[ticker] for ticker in config.tickers
            }
            turnover = sum(abs(deltas[ticker] * references[ticker]) for ticker in config.tickers)
            turnover_fraction = turnover / equity if equity > 0.0 else float("inf")
            if forced or turnover_fraction >= config.minimum_turnover:
                for ticker in config.tickers:
                    delta = deltas[ticker]
                    if abs(delta) <= 1e-12:
                        continue
                    reference = references[ticker]
                    direction = 1.0 if delta > 0.0 else -1.0
                    effective = reference * (1.0 + direction * config.transaction_cost_rate)
                    cost = abs(delta) * abs(effective - reference)
                    equity -= cost
                    total_cost += cost
                    quantities[ticker] += delta
                    trades.append(
                        Trade(
                            timestamp=row.timestamp,
                            ticker=ticker,
                            quantity=delta,
                            reference_price=reference,
                            effective_fill=effective,
                            target_weight=target[ticker],
                            bid=(row.bid or {}).get(ticker),
                            ask=(row.ask or {}).get(ticker),
                        )
                    )
                if not forced:
                    changes += 1
                else:
                    flattened = True
            pending = None

        high_water = max(high_water, equity)
        drawdown = max(0.0, 1.0 - equity / high_water)
        max_drawdown = max(max_drawdown, drawdown)
        curve.append(EquityPoint(timestamp=row.timestamp, equity=equity, drawdown=drawdown))

        if row.target_weights is not None and risk_stop_time is None:
            if set(row.target_weights) != set(config.tickers):
                raise ValueError("Target Weights must cover the fixed Trading Universe")
            pending = (
                row.timestamp + timedelta(seconds=config.decision_latency_seconds),
                row.target_weights,
                False,
            )

        previous_marks = row.mark_prices
        if flattened:
            break

    return ReplayResult(
        initial_equity=config.initial_equity,
        final_equity=equity,
        max_drawdown=max_drawdown,
        transaction_cost=total_cost,
        funding_cashflow=total_funding,
        qualifying_portfolio_changes=changes,
        risk_stop_triggered=risk_stop_time is not None,
        risk_stop_time=risk_stop_time,
        trades=tuple(trades),
        equity=tuple(curve),
    )
