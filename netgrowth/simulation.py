"""Pure Causal Replay of portfolio execution, funding, and the Risk Stop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .config import PolicyConfig


@dataclass(frozen=True)
class MarketMinute:
    timestamp: datetime
    mark_prices: dict[str, float]
    historical_open: dict[str, float] | None = None
    bid: dict[str, float] | None = None
    ask: dict[str, float] | None = None
    quote_exchange_times: dict[str, datetime] | None = None
    quote_observed_at: datetime | None = None
    funding_rates: dict[str, float] = field(default_factory=dict)
    funding_mark_prices: dict[str, float] = field(default_factory=dict)
    target_weights: dict[str, float] | None = None


@dataclass(frozen=True)
class SimulationConfig:
    tickers: tuple[str, ...]
    initial_equity: float = 10_000.0
    decision_latency_seconds: int = 60
    transaction_cost_rate: float = 0.0007
    minimum_turnover: float = 0.01
    drawdown_limit: float = 0.20
    max_gross_exposure: float = 1.0
    max_instrument_weight: float = 0.5
    mode: Literal["historical", "paper"] = "historical"


def simulation_config_for_policy(
    config: PolicyConfig,
    *,
    mode: Literal["historical", "paper"],
) -> SimulationConfig:
    """Create the one execution contract shared by every evidence mode."""
    return SimulationConfig(
        tickers=config.tickers,
        initial_equity=config.initial_equity,
        decision_latency_seconds=config.decision_latency_seconds,
        transaction_cost_rate=config.transaction_cost_rate,
        minimum_turnover=config.minimum_turnover,
        drawdown_limit=config.drawdown_limit,
        max_gross_exposure=config.max_gross_exposure,
        max_instrument_weight=config.max_instrument_weight,
        mode=mode,
    )


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
    quote_exchange_time: datetime | None = None
    quote_observed_at: datetime | None = None


@dataclass(frozen=True)
class EquityPoint:
    timestamp: datetime
    equity: float
    drawdown: float
    gross_exposure: float


@dataclass(frozen=True)
class PendingPortfolioChange:
    eligible_at: datetime
    target_weights: dict[str, float]
    forced: bool = False
    eligibility_at_signal: bool = True


@dataclass
class SimulationState:
    """Minimal serializable state for continuing a causal simulation."""

    quantities: dict[str, float]
    equity: float
    high_water: float
    max_drawdown: float = 0.0
    max_gross_exposure: float = 0.0
    max_instrument_exposure: dict[str, float] = field(default_factory=dict)
    turnover_notional: float = 0.0
    transaction_cost: float = 0.0
    funding_cashflow: float = 0.0
    executable_portfolio_changes: int = 0
    previous_timestamp: datetime | None = None
    previous_marks: dict[str, float] | None = None
    pending: PendingPortfolioChange | None = None
    risk_stop_time: datetime | None = None
    flattened: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "quantities": self.quantities,
            "equity": self.equity,
            "high_water": self.high_water,
            "max_drawdown": self.max_drawdown,
            "max_gross_exposure": self.max_gross_exposure,
            "max_instrument_exposure": self.max_instrument_exposure,
            "turnover_notional": self.turnover_notional,
            "transaction_cost": self.transaction_cost,
            "funding_cashflow": self.funding_cashflow,
            "executable_portfolio_changes": self.executable_portfolio_changes,
            "previous_timestamp": self.previous_timestamp,
            "previous_marks": self.previous_marks,
            "pending": (
                {
                    "eligible_at": self.pending.eligible_at,
                    "target_weights": self.pending.target_weights,
                    "forced": self.pending.forced,
                    "eligibility_at_signal": self.pending.eligibility_at_signal,
                }
                if self.pending
                else None
            ),
            "risk_stop_time": self.risk_stop_time,
            "flattened": self.flattened,
        }

    @classmethod
    def from_payload(cls, raw: dict[str, Any]) -> SimulationState:
        pending = raw.get("pending")
        return cls(
            quantities={key: float(value) for key, value in raw["quantities"].items()},
            equity=float(raw["equity"]),
            high_water=float(raw["high_water"]),
            max_drawdown=float(raw["max_drawdown"]),
            max_gross_exposure=float(raw.get("max_gross_exposure", 0.0)),
            max_instrument_exposure={
                ticker: float(value)
                for ticker, value in raw.get(
                    "max_instrument_exposure",
                    dict.fromkeys(raw["quantities"], 0.0),
                ).items()
            },
            turnover_notional=float(raw["turnover_notional"]),
            transaction_cost=float(raw["transaction_cost"]),
            funding_cashflow=float(raw["funding_cashflow"]),
            executable_portfolio_changes=int(raw["executable_portfolio_changes"]),
            previous_timestamp=datetime.fromisoformat(raw["previous_timestamp"]) if raw["previous_timestamp"] else None,
            previous_marks=(
                {key: float(value) for key, value in raw["previous_marks"].items()} if raw["previous_marks"] else None
            ),
            pending=(
                PendingPortfolioChange(
                    eligible_at=datetime.fromisoformat(pending["eligible_at"]),
                    target_weights={key: float(value) for key, value in pending["target_weights"].items()},
                    forced=bool(pending["forced"]),
                    eligibility_at_signal=bool(pending.get("eligibility_at_signal", False)),
                )
                if pending
                else None
            ),
            risk_stop_time=datetime.fromisoformat(raw["risk_stop_time"]) if raw["risk_stop_time"] else None,
            flattened=bool(raw["flattened"]),
        )


@dataclass(frozen=True)
class ReplayResult:
    initial_equity: float
    final_equity: float
    max_drawdown: float
    max_gross_exposure: float
    max_instrument_exposure: dict[str, float]
    turnover_notional: float
    transaction_cost: float
    funding_cashflow: float
    executable_portfolio_changes: int
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
    missing_funding_marks = set(row.funding_rates) - set(row.funding_mark_prices)
    if config.mode == "paper" and missing_funding_marks:
        raise ValueError(f"missing settled funding mark price for {sorted(missing_funding_marks)}")


def _reference_price(row: MarketMinute, ticker: str, mode: str) -> float:
    if mode == "historical":
        return (row.historical_open or {})[ticker]
    return ((row.bid or {})[ticker] + (row.ask or {})[ticker]) / 2.0


def _post_cost_deltas(
    state: SimulationState,
    target_weights: dict[str, float],
    references: dict[str, float],
    config: SimulationConfig,
) -> tuple[dict[str, float], float]:
    """Solve target quantities against equity after the one all-in Effective Fill cost."""
    post_cost_equity = state.equity
    deltas: dict[str, float] = {}
    for _ in range(32):
        deltas = {
            ticker: target_weights[ticker] * post_cost_equity / references[ticker] - state.quantities[ticker]
            for ticker in config.tickers
        }
        turnover = sum(abs(deltas[ticker] * references[ticker]) for ticker in config.tickers)
        updated_equity = state.equity - config.transaction_cost_rate * turnover
        if abs(updated_equity - post_cost_equity) <= max(1e-10, abs(state.equity) * 1e-12):
            break
        post_cost_equity = updated_equity
    else:
        raise RuntimeError("Effective Fill post-cost equity did not converge")
    return deltas, turnover


def marked_weights(
    state: SimulationState,
    mark_prices: dict[str, float],
    tickers: tuple[str, ...],
) -> dict[str, float]:
    if state.equity <= 0.0:
        return dict.fromkeys(tickers, 0.0)
    return {ticker: state.quantities[ticker] * mark_prices[ticker] / state.equity for ticker in tickers}


def schedule_portfolio_change(
    state: SimulationState,
    signal_time: datetime,
    target_weights: dict[str, float],
    config: SimulationConfig,
) -> bool:
    """Schedule a target after marking the signal minute in an incremental session."""
    if state.previous_timestamp != signal_time:
        raise ValueError("a paper target must follow its marked Signal Time")
    if state.risk_stop_time is not None:
        raise ValueError("Risk Stop prevents later policy decisions")
    if set(target_weights) != set(config.tickers):
        raise ValueError("Target Weights must cover the fixed Trading Universe")
    if any(abs(weight) > config.max_instrument_weight + 1e-6 for weight in target_weights.values()):
        raise ValueError("Target Weights exceed the per-instrument concentration limit")
    if sum(abs(weight) for weight in target_weights.values()) > config.max_gross_exposure + 1e-6:
        raise ValueError("Target Weights exceed the Gross Exposure limit")
    if state.previous_marks is None:
        raise ValueError("a paper target requires marked Signal-Time prices")
    current_weights = marked_weights(state, state.previous_marks, config.tickers)
    turnover = sum(abs(target_weights[ticker] - current_weights[ticker]) for ticker in config.tickers)
    if turnover < config.minimum_turnover:
        return False
    state.pending = PendingPortfolioChange(
        eligible_at=signal_time + timedelta(seconds=config.decision_latency_seconds),
        target_weights=target_weights,
    )
    return True


def _mark_drawdown(state: SimulationState, timestamp: datetime, config: SimulationConfig) -> float:
    state.high_water = max(state.high_water, state.equity)
    drawdown = max(0.0, 1.0 - state.equity / state.high_water)
    state.max_drawdown = max(state.max_drawdown, drawdown)
    if drawdown > config.drawdown_limit and state.risk_stop_time is None:
        state.risk_stop_time = timestamp
        state.pending = PendingPortfolioChange(
            eligible_at=timestamp + timedelta(seconds=config.decision_latency_seconds),
            target_weights=dict.fromkeys(config.tickers, 0.0),
            forced=True,
        )
    return drawdown


def _initial_state(config: SimulationConfig) -> SimulationState:
    return SimulationState(
        quantities=dict.fromkeys(config.tickers, 0.0),
        equity=config.initial_equity,
        high_water=config.initial_equity,
        max_instrument_exposure=dict.fromkeys(config.tickers, 0.0),
    )


def advance_simulation(
    rows: list[MarketMinute],
    config: SimulationConfig,
    state: SimulationState | None = None,
) -> tuple[ReplayResult, SimulationState]:
    """Advance a pure simulation while preserving only explicit serializable state."""
    if not rows:
        raise ValueError("Causal Replay requires at least one minute")
    ordered = sorted(rows, key=lambda row: row.timestamp)
    if ordered != rows or len({row.timestamp for row in rows}) != len(rows):
        raise ValueError("market minutes must be unique and ordered")
    state = state or _initial_state(config)
    if set(state.quantities) != set(config.tickers):
        raise ValueError("simulation state must cover the fixed Trading Universe")
    if state.previous_timestamp is not None and rows[0].timestamp <= state.previous_timestamp:
        raise ValueError("incremental market minutes must follow prior state")

    trades: list[Trade] = []
    curve: list[EquityPoint] = []

    for row in ordered:
        _validate_minute(row, config)

        fill_references: dict[str, float] | None = None
        if state.pending is not None and row.timestamp >= state.pending.eligible_at:
            if config.mode == "paper":
                quote_times = row.quote_exchange_times or {}
                if set(config.tickers) - set(quote_times):
                    raise ValueError("paper Effective Fill requires exchange-timestamped quotes")
                if any(quote_times[ticker] < state.pending.eligible_at for ticker in config.tickers):
                    raise ValueError("paper quote predates the delayed fill eligibility time")
            fill_references = {ticker: _reference_price(row, ticker, config.mode) for ticker in config.tickers}
        valuation_prices = fill_references or row.mark_prices
        if state.previous_marks is not None:
            state.equity += sum(
                state.quantities[ticker] * (valuation_prices[ticker] - state.previous_marks[ticker])
                for ticker in config.tickers
            )
        funding = -sum(
            state.quantities[ticker]
            * row.funding_mark_prices.get(ticker, row.mark_prices[ticker])
            * row.funding_rates.get(ticker, 0.0)
            for ticker in config.tickers
        )
        state.equity += funding
        state.funding_cashflow += funding

        if config.mode == "paper":
            _mark_drawdown(state, row.timestamp, config)

        executable_pending = (
            state.pending
            if fill_references is not None and state.pending is not None and row.timestamp >= state.pending.eligible_at
            else None
        )
        if executable_pending is not None:
            assert fill_references is not None
            target, forced = executable_pending.target_weights, executable_pending.forced
            deltas, turnover = _post_cost_deltas(state, target, fill_references, config)
            executes = forced or executable_pending.eligibility_at_signal
            if not executes:
                turnover_fraction = turnover / state.equity if state.equity > 0.0 else float("inf")
                executes = turnover_fraction >= config.minimum_turnover
            if executes:
                state.turnover_notional += turnover
                for ticker in config.tickers:
                    delta = deltas[ticker]
                    if abs(delta) <= 1e-12:
                        continue
                    reference = fill_references[ticker]
                    direction = 1.0 if delta > 0.0 else -1.0
                    effective = reference * (1.0 + direction * config.transaction_cost_rate)
                    cost = abs(delta) * abs(effective - reference)
                    state.equity -= cost
                    state.transaction_cost += cost
                    state.quantities[ticker] += delta
                    trades.append(
                        Trade(
                            timestamp=(row.quote_exchange_times or {}).get(ticker, row.timestamp),
                            ticker=ticker,
                            quantity=delta,
                            reference_price=reference,
                            effective_fill=effective,
                            target_weight=target[ticker],
                            bid=(row.bid or {}).get(ticker),
                            ask=(row.ask or {}).get(ticker),
                            quote_exchange_time=(row.quote_exchange_times or {}).get(ticker),
                            quote_observed_at=row.quote_observed_at,
                        )
                    )
                if not forced:
                    state.executable_portfolio_changes += 1
                else:
                    state.flattened = True
            state.pending = None

        drawdown = _mark_drawdown(state, row.timestamp, config)
        weights = marked_weights(state, valuation_prices, config.tickers)
        gross_exposure = sum(abs(weight) for weight in weights.values())
        state.max_gross_exposure = max(state.max_gross_exposure, gross_exposure)
        for ticker, weight in weights.items():
            state.max_instrument_exposure[ticker] = max(
                state.max_instrument_exposure.get(ticker, 0.0),
                abs(weight),
            )
        curve.append(
            EquityPoint(
                timestamp=row.timestamp,
                equity=state.equity,
                drawdown=drawdown,
                gross_exposure=gross_exposure,
            )
        )

        if row.target_weights is not None and state.risk_stop_time is None:
            state.previous_timestamp = row.timestamp
            state.previous_marks = valuation_prices
            schedule_portfolio_change(state, row.timestamp, row.target_weights, config)
        else:
            state.previous_timestamp = row.timestamp
            state.previous_marks = valuation_prices
        if state.flattened:
            break

    result = ReplayResult(
        initial_equity=config.initial_equity,
        final_equity=state.equity,
        max_drawdown=state.max_drawdown,
        max_gross_exposure=state.max_gross_exposure,
        max_instrument_exposure=state.max_instrument_exposure.copy(),
        turnover_notional=state.turnover_notional,
        transaction_cost=state.transaction_cost,
        funding_cashflow=state.funding_cashflow,
        executable_portfolio_changes=state.executable_portfolio_changes,
        risk_stop_triggered=state.risk_stop_time is not None,
        risk_stop_time=state.risk_stop_time,
        trades=tuple(trades),
        equity=tuple(curve),
    )
    return result, state


def simulate(rows: list[MarketMinute], config: SimulationConfig) -> ReplayResult:
    """Replay ordered minute marks from a Flat Start."""
    result, _ = advance_simulation(rows, config)
    return result
