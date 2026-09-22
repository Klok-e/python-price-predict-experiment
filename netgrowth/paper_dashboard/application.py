"""Single-owner Paper Account application and its in-process ASGI boundary."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from math import ceil
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, cast
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from netgrowth.accounting import HoldBenchmark, PassiveBenchmarks
from netgrowth.simulation import (
    MarketMinute,
    PendingPortfolioChange,
    SimulationConfig,
    advance_simulation,
    marked_weights,
    schedule_portfolio_change,
)

from .domain import (
    AccountEvent,
    Clock,
    DataStatus,
    LifecycleState,
    MarketObservation,
    PaperAccountState,
    PendingExecution,
    PolicyDecision,
    SystemClock,
    eligibility_revision_is_supported,
    policy_revision_is_compatible,
)
from .persistence import SQLitePaperStore, StateVersionConflict
from .revision_evidence import validate_revision_evidence

DEFAULT_TICKERS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
KYIV_TIME_ZONE = ZoneInfo("Europe/Kiev")
CHART_ONLY_EVENT_TYPES = frozenset({"AccountMarked", "AccountMarkReconstructed"})


def _is_material_event(event: AccountEvent) -> bool:
    return event.event_type not in CHART_ONLY_EVENT_TYPES


class MarketFeed(Protocol):
    def observe(self, after: datetime | None) -> MarketObservation | Awaitable[MarketObservation]: ...


class PolicyBackend(Protocol):
    def decide(
        self,
        observation: MarketObservation,
        current_weights: dict[str, float],
    ) -> PolicyDecision | Mapping[str, float] | Awaitable[PolicyDecision | Mapping[str, float]]: ...


class NotificationSink(Protocol):
    def notify(self, kind: str, payload: dict[str, Any]) -> object: ...


@dataclass(frozen=True, slots=True)
class FittedPolicyCandidate:
    model_id: str
    checkpoint: str
    payload: object | None = None


class PolicyFitter(Protocol):
    def is_due(self, now: datetime, fitting: Mapping[str, Any]) -> bool: ...

    def fit(
        self,
        observed_at: datetime,
        current_weights: dict[str, float],
    ) -> FittedPolicyCandidate: ...

    def activate(self, candidate: FittedPolicyCandidate) -> Callable[[], None]: ...


class AttributionBackend(Protocol):
    def attribute(self, input_id: str, decision: Mapping[str, Any]) -> dict[str, Any]: ...


class _UnavailableFeed:
    def observe(self, after: datetime | None) -> MarketObservation:
        del after
        raise RuntimeError("market feed is not configured")


class _FlatPolicy:
    def __init__(self, tickers: tuple[str, ...]) -> None:
        self._tickers = tickers

    def decide(self, observation: MarketObservation, current_weights: dict[str, float]) -> PolicyDecision:
        return PolicyDecision(
            raw_target_weights=current_weights,
            target_weights=current_weights,
            model_id="unconfigured",
            input_id=observation.input_id,
            protocol_id="unconfigured",
        )


class _NoNotifications:
    def notify(self, kind: str, payload: dict[str, Any]) -> None:
        del kind, payload


class ControlRequest(BaseModel):
    expected_version: int
    confirmation: str | bool | None = None


class PaperDashboardApplication:
    """Deep operational interface owning state, persistence, controls, and read models."""

    def __init__(
        self,
        *,
        database_path: Path | str,
        tickers: tuple[str, ...] = DEFAULT_TICKERS,
        clock: Clock | None = None,
        market_feed: MarketFeed | None = None,
        policy_backend: PolicyBackend | None = None,
        notifications: NotificationSink | None = None,
        policy_fitter: PolicyFitter | None = None,
        attribution_backend: AttributionBackend | None = None,
        simulation_config: SimulationConfig | None = None,
        starting_equity: float = 10_000.0,
        backup_directory: Path | None = None,
    ) -> None:
        self.clock = clock or SystemClock()
        self.market_feed = market_feed or _UnavailableFeed()
        self.policy_backend = policy_backend or _FlatPolicy(tickers)
        self.notifications = notifications or _NoNotifications()
        self.policy_fitter = policy_fitter
        self.attribution_backend = attribution_backend
        self.config = simulation_config or SimulationConfig(
            tickers=tickers,
            initial_equity=starting_equity,
            mode="paper",
        )
        if self.config.tickers != tickers or self.config.mode != "paper":
            raise ValueError("Paper Account requires its fixed paper-mode Trading Universe")
        self.store = SQLitePaperStore(database_path, backup_directory=backup_directory)
        self.csrf_token = secrets.token_urlsafe(32)
        self._owner_lock = RLock()
        self._closed = False
        self._observed_in_window = False
        self._policy_preparing = False
        self._fitting_task: asyncio.Task[None] | None = None
        self._notification_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="paper-notifications")
        self._notification_jobs: dict[str, Future[object]] = {}
        self._attribution_tasks: set[asyncio.Task[None]] = set()
        self._attribution_decisions: set[str] = set()
        self._pending_notifications: list[tuple[str, str, dict[str, Any], datetime]] = []
        self._fitting_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="paper-policy-fitting",
        )
        self._attribution_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="paper-attribution",
        )
        now = self._now()
        self.state = self.store.open_active_account(now, tickers, starting_equity=starting_equity)
        if self.state.model_fitted_at is None:
            handoffs = [e for e in self.store.events(self.state.account_id) if e.event_type == "PolicyHandoff"]
            if handoffs:
                self.state.model_fitted_at = (
                    handoffs[-1].payload.get("completed_at") or handoffs[-1].occurred_at.isoformat()
                )
        self._check_daily_backup()
        prior_windows = self.store.operating_windows(self.state.account_id)
        detected_open_window = next(
            (window for window in reversed(prior_windows) if window["ended_at"] is None),
            None,
        )
        if (
            self.state.lifecycle is LifecycleState.RESET_PENDING
            and self.state.pending_execution is None
            and self._is_flat()
        ):
            if detected_open_window is None:
                raise RuntimeError("flat Reset Pending account has no Operating Window to complete")
            self.state, self.window_id = self.store.reset_and_replace(
                self.state,
                now,
                window_id=str(detected_open_window["window_id"]),
                window_ended_at=self.state.last_observation_at,
                starting_equity=self.config.initial_equity,
            )
            self._deliver_notifications()
            return
        self.window_id = str(uuid4())
        opening_events: list[tuple[str, datetime, dict[str, Any], str | None, str | None]] = []
        if detected_open_window is not None:
            opening_events.append(
                (
                    "OperatingWindowClosed",
                    now,
                    {
                        "window_id": detected_open_window["window_id"],
                        "reason": "detected-restart",
                    },
                    None,
                    None,
                )
            )
        opening_events.append(("OperatingWindowOpened", now, {"window_id": self.window_id}, None, None))
        if self.state.last_observation_at is not None and now - self.state.last_observation_at > timedelta(minutes=1):
            opening_events.append(
                (
                    "OperatingGap",
                    now,
                    {"start": self.state.last_observation_at, "end": now, "reconstructed": False},
                    None,
                    None,
                )
            )
        detected_end = self.state.last_observation_at
        if detected_open_window is not None and detected_end is None:
            detected_end = datetime.fromisoformat(str(detected_open_window["started_at"]))
        self._commit(
            opening_events,
            operating_window=(self.window_id, now, detected_end),
        )

    def __enter__(self) -> PaperDashboardApplication:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        with self._owner_lock:
            if self._closed:
                return
            now = self._now()
            self._commit(
                [("OperatingWindowClosed", now, {"window_id": self.window_id, "reason": "graceful"}, None, None)],
                close_window=(self.window_id, now, "graceful"),
            )
            self._notification_executor.shutdown(wait=True, cancel_futures=True)
            self._collect_notifications()
            self._fitting_executor.shutdown(wait=True, cancel_futures=True)
            self._attribution_executor.shutdown(wait=True, cancel_futures=True)
            self.store.close()
            self._closed = True

    async def advance_once(self, observation: MarketObservation | None = None) -> dict[str, Any]:
        self._check_daily_backup()
        if observation is None:
            try:
                supplied = self.market_feed.observe(self.state.last_observation_at)
                observation = await supplied if inspect.isawaitable(supplied) else supplied
                recovered = self._recover_before(observation)
                recovered = await recovered if inspect.isawaitable(recovered) else recovered
            except Exception as error:
                return self.record_data_stale(error)
            for recovered_observation in recovered:
                self._advance(recovered_observation)
        return self._advance(observation)

    def advance_once_sync(self, observation: MarketObservation | None = None) -> dict[str, Any]:
        self._check_daily_backup()
        if observation is None:
            try:
                supplied = self.market_feed.observe(self.state.last_observation_at)
                if inspect.isawaitable(supplied):
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        observation = asyncio.run(_await_observation(supplied))
                    else:
                        raise RuntimeError("use await advance_once() with an asynchronous market feed")
                else:
                    observation = supplied
                recovered = self._recover_before(observation)
                if inspect.isawaitable(recovered):
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError:
                        recovered = asyncio.run(_await_recovery(recovered))
                    else:
                        raise RuntimeError("use await advance_once() with asynchronous market recovery")
            except Exception as error:
                return self.record_data_stale(error)
            for recovered_observation in recovered:
                self._advance(recovered_observation)
        return self._advance(observation)

    def advance_from_feed_sync(self) -> dict[str, Any]:
        """Advance while converting provider failure into the Data Stale overlay."""
        return self.advance_once_sync()

    def _advance(self, observation: MarketObservation) -> dict[str, Any]:
        with self._owner_lock:
            try:
                return self._advance_uncommitted(observation)
            except BaseException:
                restored = self.store.load_active_account()
                if restored is None:
                    raise RuntimeError("failed transition left no active Paper Account") from None
                self.state = restored
                raise

    def _advance_uncommitted(self, observation: MarketObservation) -> dict[str, Any]:
        with self._owner_lock:
            if self._closed:
                raise RuntimeError("Paper Account application is closed")
            if observation.timestamp <= (self.state.last_observation_at or datetime.min.replace(tzinfo=UTC)):
                raise ValueError("observations must advance UTC Market Time")
            events: list[tuple[str, datetime, dict[str, Any], str | None, str | None]] = []
            self.state.operator_error = None
            was_stale = self.state.data_status is DataStatus.STALE
            self.state.data_status = DataStatus.FRESH
            self.state.data_error = None
            self.state.stale_since = None
            if was_stale:
                events.append(("DataRecovered", observation.timestamp, {}, None, None))

            if (
                self.state.last_observation_at is not None
                and observation.timestamp - self.state.last_observation_at > timedelta(minutes=1)
            ):
                events.append(
                    (
                        "OperatingGap",
                        observation.timestamp,
                        {
                            "start": self.state.last_observation_at,
                            "end": observation.timestamp,
                            "reconstructed": observation.reconstructed,
                        },
                        None,
                        None,
                    )
                )

            pending = self.state.pending_execution
            if (
                not observation.reconstructed
                and pending is not None
                and pending.kind == "policy"
                and observation.quote_observed_at > pending.expires_at
            ):
                self.state.simulation.pending = None
                self.state.pending_execution = None
                events.append(
                    (
                        "MissedExecution",
                        observation.timestamp,
                        {
                            "decision_id": pending.decision_id,
                            "eligible_at": pending.eligible_at,
                            "expired_at": pending.expires_at,
                        },
                        pending.decision_id,
                        None,
                    )
                )
                pending = None

            prior_pending = pending
            prior_funding = self.state.simulation.funding_cashflow
            prior_risk_stop = self.state.simulation.risk_stop_time
            preserved_pending = self.state.simulation.pending if observation.reconstructed else None
            if observation.reconstructed:
                self.state.simulation.pending = None
            result, simulation = advance_simulation(
                [
                    MarketMinute(
                        timestamp=observation.timestamp,
                        mark_prices=observation.mark_prices,
                        bid=observation.bid,
                        ask=observation.ask,
                        quote_exchange_times=observation.quote_exchange_times,
                        quote_observed_at=observation.quote_observed_at,
                        funding_rates=observation.funding_rates,
                        funding_mark_prices=observation.funding_mark_prices,
                    )
                ],
                self.config,
                self.state.simulation,
            )
            if observation.reconstructed and simulation.risk_stop_time == prior_risk_stop:
                simulation.pending = preserved_pending
            self.state.simulation = simulation
            self.state.last_observation_at = observation.timestamp
            if self.state.benchmarks is None:
                self.state.benchmarks = PassiveBenchmarks.flat_start(
                    self.state.tickers,
                    observation.mark_prices,
                    starting_equity=self.state.starting_equity,
                    transaction_cost_rate=self.config.transaction_cost_rate,
                )
            funding_cashflow = simulation.funding_cashflow - prior_funding
            if abs(funding_cashflow) > 0.0:
                self.state.accounting.apply_funding(funding_cashflow)
            if self.state.hold_benchmark is not None:
                self.state.hold_benchmark.apply_funding(observation.funding_rates, observation.funding_mark_prices)
                self.state.hold_benchmark.mark(observation.mark_prices)
            if observation.funding_rates:
                self.state.benchmarks.apply_funding(
                    observation.funding_rates,
                    observation.funding_mark_prices,
                )
            for trade in result.trades:
                cost = abs(trade.quantity) * abs(trade.effective_fill - trade.reference_price)
                self.state.accounting.apply_fill(
                    trade.ticker,
                    quantity_delta=trade.quantity,
                    reference_price=trade.reference_price,
                    transaction_cost=cost,
                )
            account_snapshot = self.state.accounting.mark(
                observation.mark_prices,
                target_weights=self.state.target_weights,
                authoritative_equity=simulation.equity,
            )
            benchmark_mark = self.state.benchmarks.mark(
                observation.mark_prices,
                reconstructed=observation.reconstructed,
            )
            equity_point = result.equity[-1]
            events.append(
                (
                    "AccountMarkReconstructed" if observation.reconstructed else "AccountMarked",
                    observation.timestamp,
                    {
                        "equity": equity_point.equity,
                        "drawdown": equity_point.drawdown,
                        "gross_exposure": equity_point.gross_exposure,
                        "marks": observation.mark_prices,
                        "candles": observation.candles,
                        "funding_cashflow": funding_cashflow,
                        "reconstructed": observation.reconstructed,
                        "hold_benchmark": self._hold_snapshot(observation.mark_prices),
                        "cash_benchmark": benchmark_mark.cash_equity,
                        "equal_weight_benchmark": benchmark_mark.equal_weight_equity,
                        "net_exposure": account_snapshot.risk.net_exposure,
                        "current_weights": account_snapshot.risk.instrument_weights,
                        "target_weights": self.state.target_weights,
                    },
                    None,
                    None,
                )
            )
            if observation.funding_rates:
                events.append(
                    (
                        "FundingApplied",
                        observation.timestamp,
                        {
                            "rates": observation.funding_rates,
                            "mark_prices": observation.funding_mark_prices,
                            "cashflow": funding_cashflow,
                            "reconstructed": observation.reconstructed,
                        },
                        None,
                        None,
                    )
                )

            if not observation.reconstructed and prior_pending is not None and simulation.pending is None:
                self.state.pending_execution = None
                if result.trades:
                    total_cost = sum(
                        abs(trade.quantity) * abs(trade.effective_fill - trade.reference_price)
                        for trade in result.trades
                    )
                    events.append(
                        (
                            "PortfolioChangeExecuted",
                            observation.timestamp,
                            {
                                "decision_id": prior_pending.decision_id,
                                "kind": prior_pending.kind,
                                "transaction_cost": total_cost,
                                "fill_count": len(result.trades),
                            },
                            prior_pending.decision_id,
                            None,
                        )
                    )
                    for trade in result.trades:
                        events.append(
                            (
                                "InstrumentFilled",
                                trade.timestamp,
                                asdict(trade),
                                prior_pending.decision_id,
                                trade.ticker,
                            )
                        )
                    self._queue_notification(
                        f"executed:{prior_pending.decision_id}",
                        "executed_portfolio_change",
                        {"decision_id": prior_pending.decision_id, "fill_count": len(result.trades)},
                        observation.timestamp,
                    )
                elif prior_pending.kind == "policy":
                    decision_record = next(
                        event
                        for event in reversed(self.store.events(self.state.account_id))
                        if event.event_type == "DecisionRecord" and event.decision_id == prior_pending.decision_id
                    )
                    events.append(
                        (
                            "DecisionCompleted",
                            observation.timestamp,
                            {
                                "decision_id": prior_pending.decision_id,
                                "outcome": decision_record.payload["threshold_outcome"],
                            },
                            prior_pending.decision_id,
                            None,
                        )
                    )

                if prior_pending.kind == "flatten":
                    simulation.flattened = False
                    self.state.lifecycle = LifecycleState.PAUSED
                    self.state.pending_control = None
                elif prior_pending.kind == "reset":
                    self.state.pending_control = None
                elif prior_pending.kind == "risk_stop":
                    self.state.lifecycle = LifecycleState.RISK_STOPPED

            if simulation.risk_stop_time is not None and self.state.lifecycle is not LifecycleState.RISK_STOPPED:
                self.state.lifecycle = LifecycleState.RISK_STOPPED
                if simulation.pending is not None:
                    decision_id = f"risk-stop:{simulation.risk_stop_time.isoformat()}"
                    self.state.pending_execution = PendingExecution(
                        decision_id=decision_id,
                        signal_time=simulation.risk_stop_time,
                        eligible_at=simulation.pending.eligible_at,
                        expires_at=simulation.pending.eligible_at + timedelta(minutes=2),
                        target_weights=dict.fromkeys(self.config.tickers, 0.0),
                        kind="risk_stop",
                    )
                events.append(("RiskStop", observation.timestamp, {"drawdown": equity_point.drawdown}, None, None))
                self._queue_notification(
                    f"risk-stop:{self.state.account_id}",
                    "risk_stop",
                    {"drawdown": equity_point.drawdown},
                    observation.timestamp,
                )

            if (
                not observation.reconstructed
                and observation.closes_decision_bar
                and self.state.lifecycle is LifecycleState.TRADING
                and self.state.last_decision_at != observation.timestamp
                and self.state.pending_execution is None
                and self.state.revision.get("status") != "draining"
                and not self._policy_preparing
            ):
                current_weights = marked_weights(simulation, observation.mark_prices, self.config.tickers)
                supplied = self.policy_backend.decide(observation, current_weights)
                if inspect.isawaitable(supplied):
                    raise RuntimeError("use advance_once with a synchronous policy backend for serialized advancement")
                decision = self._coerce_decision(supplied, observation, current_weights)
                self._validate_target(decision.target_weights)
                if self.state.model_id != "unknown" and decision.model_id != self.state.model_id:
                    raise RuntimeError("ordinary decision cannot replace the durable Fitted Policy")
                if self.state.protocol_id != "unknown" and decision.protocol_id != self.state.protocol_id:
                    raise RuntimeError("ordinary decision cannot bypass the durable Policy Revision")
                decision_id = hashlib.sha256(
                    f"{self.state.account_id}:{observation.timestamp.isoformat()}:{decision.input_id}".encode()
                ).hexdigest()[:32]
                projected_turnover = sum(
                    abs(decision.target_weights[ticker] - current_weights[ticker]) for ticker in self.config.tickers
                )
                eligible_at = observation.timestamp + timedelta(seconds=self.config.decision_latency_seconds)
                schedule = PendingExecution(
                    decision_id=decision_id,
                    signal_time=observation.timestamp,
                    eligible_at=eligible_at,
                    expires_at=eligible_at + timedelta(minutes=2),
                    target_weights=decision.target_weights,
                )
                simulation.previous_timestamp = observation.timestamp
                simulation.previous_marks = observation.mark_prices
                qualifies = schedule_portfolio_change(
                    simulation, observation.timestamp, decision.target_weights, self.config
                )
                self.state.pending_execution = schedule if qualifies else None
                self.state.last_decision_at = observation.timestamp
                self.state.target_weights = decision.target_weights.copy()
                events.append(
                    (
                        "DecisionRecord",
                        observation.timestamp,
                        {
                            "decision_id": decision_id,
                            "signal_time": observation.timestamp,
                            "model_id": decision.model_id,
                            "protocol_id": decision.protocol_id,
                            "input_id": decision.input_id,
                            "current_portfolio": current_weights,
                            "raw_target_weights": decision.raw_target_weights,
                            "constrained_target_weights": decision.target_weights,
                            "target_weights": decision.target_weights,
                            "projected_turnover": projected_turnover,
                            "threshold": self.config.minimum_turnover,
                            "threshold_outcome": (
                                "unchanged"
                                if projected_turnover <= 1e-12
                                else (
                                    "executable"
                                    if projected_turnover >= self.config.minimum_turnover
                                    else "below_threshold"
                                )
                            ),
                            "eligible_at": eligible_at,
                            "expires_at": schedule.expires_at,
                            "constraints": decision.constraints,
                            "outcome": "pending" if qualifies else "completed",
                            "eligibility": "signal_time",
                            "attribution": {"status": "pending"},
                        },
                        decision_id,
                        None,
                    )
                )

            if events and events[-1][0] == "DecisionRecord" and self.state.pending_execution is None:
                record = events[-1]
                events.append(
                    (
                        "DecisionCompleted",
                        observation.timestamp,
                        {"decision_id": record[3], "outcome": record[2]["threshold_outcome"]},
                        record[3],
                        None,
                    )
                )
            self._commit(events)
            if not observation.reconstructed:
                self._observed_in_window = True
                self._activate_staged_revision()
            if prior_pending is not None and prior_pending.kind == "reset" and self._is_flat():
                self._complete_reset(observation.timestamp)
            self._deliver_notifications()
            return self.live_snapshot()

    def record_data_stale(self, error: Exception | str) -> dict[str, Any]:
        with self._owner_lock:
            now = self._now()
            message = str(error)
            events: list[tuple[str, datetime, dict[str, Any], str | None, str | None]] = []
            if self.state.data_status is not DataStatus.STALE:
                self.state.data_status = DataStatus.STALE
                self.state.stale_since = now
                events.append(("DataStale", now, {"error": message}, None, None))
            self.state.data_error = message
            stale_for = now - (self.state.stale_since or now)
            if stale_for >= timedelta(minutes=5):
                self.store.enqueue_notification(
                    self.state.account_id,
                    f"data-stale:{self.state.account_id}:{(self.state.stale_since or now).isoformat()}",
                    "data_stale",
                    {"error": message, "stale_seconds": stale_for.total_seconds()},
                    now,
                )
            if events:
                self._commit(events)
            self._deliver_notifications()
            return self.live_snapshot()

    def record_operator_error(self, error: Exception | str) -> dict[str, Any]:
        """Keep internal policy/account failures distinct from public-data staleness."""
        with self._owner_lock:
            now = self._now()
            self.state.operator_error = str(error)
            self._commit([("OperatorError", now, {"error": self.state.operator_error}, None, None)])
            return self.system_snapshot()

    def control(
        self,
        action: str,
        *,
        expected_version: int,
        idempotency_key: str,
        confirmation: str | bool | None = None,
    ) -> dict[str, Any]:
        with self._owner_lock:
            try:
                return self._control_uncommitted(
                    action,
                    expected_version=expected_version,
                    idempotency_key=idempotency_key,
                    confirmation=confirmation,
                )
            except BaseException:
                restored = self.store.load_active_account()
                if restored is None:
                    raise RuntimeError("failed control left no active Paper Account") from None
                self.state = restored
                raise

    def _control_uncommitted(
        self,
        action: str,
        *,
        expected_version: int,
        idempotency_key: str,
        confirmation: str | bool | None = None,
    ) -> dict[str, Any]:
        with self._owner_lock:
            now = self._now()
            request_payload = {
                "action": action,
                "expected_version": expected_version,
                "confirmation": confirmation,
            }
            request_hash = hashlib.sha256(
                json.dumps(request_payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            prior = self.store.idempotent_response(idempotency_key, action, request_hash)
            if prior is not None:
                return prior
            if expected_version != self.state.state_version:
                raise StateVersionConflict(
                    f"expected state version {expected_version}, found {self.state.state_version}"
                )

            events: list[tuple[str, datetime, dict[str, Any], str | None, str | None]] = []
            if action == "pause":
                if self.state.lifecycle is LifecycleState.TRADING:
                    cancelled = self.state.pending_execution
                    self.state.pending_execution = None
                    self.state.simulation.pending = None
                    self.state.lifecycle = LifecycleState.PAUSED
                    events.append(
                        (
                            "OperatorIntervention",
                            now,
                            {"action": "pause", "cancelled_decision_id": cancelled.decision_id if cancelled else None},
                            None,
                            None,
                        )
                    )
                elif self.state.lifecycle is not LifecycleState.PAUSED:
                    raise ValueError(f"Pause is not valid from {self.state.lifecycle.value}")
            elif action == "resume":
                if self.state.lifecycle is LifecycleState.PAUSED and self.state.pending_control is None:
                    self.state.lifecycle = LifecycleState.TRADING
                    events.append(("OperatorIntervention", now, {"action": "resume"}, None, None))
                elif self.state.lifecycle is not LifecycleState.TRADING:
                    raise ValueError(f"Resume is not valid from {self.state.lifecycle.value}")
            elif action == "flatten":
                self._require_confirmation(confirmation, "FLATTEN")
                if self.state.lifecycle in {
                    LifecycleState.RISK_STOPPED,
                    LifecycleState.MIGRATION_REQUIRED,
                    LifecycleState.RESET_PENDING,
                }:
                    raise ValueError(f"Flatten and Pause is not valid from {self.state.lifecycle.value}")
                self.state.lifecycle = LifecycleState.PAUSED
                self.state.simulation.pending = None
                self.state.pending_execution = None
                events.append(("OperatorIntervention", now, {"action": "flatten_and_pause"}, None, None))
                if not self._is_flat():
                    self._schedule_control_fill("flatten", now)
            elif action == "reset":
                self._require_confirmation(confirmation, "RESET")
                if self.state.lifecycle is LifecycleState.MIGRATION_REQUIRED:
                    raise ValueError("Manual Reset cannot guess an incompatible migration")
                if self.state.lifecycle is not LifecycleState.RESET_PENDING:
                    self.state.lifecycle = LifecycleState.RESET_PENDING
                    self.state.simulation.pending = None
                    self.state.pending_execution = None
                    events.append(("OperatorIntervention", now, {"action": "manual_reset"}, None, None))
                    if not self._is_flat():
                        self._schedule_control_fill("reset", now)
            else:
                raise ValueError(f"unknown Paper Account control {action!r}")

            if action == "reset" and self._is_flat():
                self._complete_reset(
                    now,
                    include_intervention=True,
                    idempotency=(idempotency_key, action, request_hash, now),
                )
                return self.live_snapshot()
            if events:
                stored_response = {
                    "accepted": True,
                    "message": f"{action.replace('_', ' ').title()} accepted",
                    "account": {
                        "id": self.state.account_id,
                        "version": self.state.state_version + 1,
                    },
                }
                self._commit(
                    events,
                    idempotency=(
                        idempotency_key,
                        action,
                        request_hash,
                        stored_response,
                        now,
                    ),
                )
                return self.live_snapshot()
            response = self.live_snapshot()
            self.store.record_idempotent_response(idempotency_key, action, request_hash, response, now)
            return response

    def complete_policy_handoff(
        self,
        *,
        new_model_id: str,
        checkpoint: str,
        protocol_id: str | None = None,
    ) -> dict[str, Any]:
        with self._owner_lock:
            now = self._now()
            old_model_id = self.state.model_id
            self.state.model_id = new_model_id
            self.state.model_checkpoint = checkpoint
            if protocol_id is not None:
                self.state.protocol_id = protocol_id
            self.state.model_fitted_at = now.isoformat()
            self.state.fitting = {
                "status": "idle",
                "due_at": self.state.fitting.get("due_at"),
                "started_at": self.state.fitting.get("started_at"),
                "completed_at": now.isoformat(),
                "last_successful_fit_at": now.isoformat(),
                "attempt_count": 0,
                "cycle_at": self.state.fitting.get("cycle_at"),
                "next_retry_at": None,
            }
            self._commit(
                [
                    (
                        "PolicyHandoff",
                        now,
                        {
                            "old_model_id": old_model_id,
                            "new_model_id": new_model_id,
                            "checkpoint": checkpoint,
                            "protocol_id": self.state.protocol_id,
                            "due_at": self.state.fitting.get("due_at"),
                            "started_at": self.state.fitting.get("started_at"),
                            "completed_at": self.state.fitting["completed_at"],
                        },
                        None,
                        None,
                    )
                ]
            )
            return self.system_snapshot()

    def register_initial_fitted_policy(
        self,
        *,
        model_id: str,
        checkpoint: str,
        fitted_at: str | None = None,
    ) -> dict[str, Any]:
        """Select the bootstrap model without fabricating a completed weekly fit."""
        with self._owner_lock:
            if self.state.model_id != "unknown":
                raise RuntimeError("the Paper Account already has a selected Fitted Policy")
            now = self._now()
            self.state.model_id = model_id
            self.state.model_checkpoint = checkpoint
            self.state.model_fitted_at = fitted_at
            self._commit(
                [
                    (
                        "FittedPolicySelected",
                        now,
                        {
                            "model_id": model_id,
                            "checkpoint": checkpoint,
                            "protocol_id": self.state.protocol_id,
                        },
                        None,
                        None,
                    )
                ]
            )
            return self.system_snapshot()

    async def maybe_start_policy_fitting(self) -> None:
        with self._owner_lock:
            fitter = self.policy_fitter
            if (
                fitter is None
                or self._policy_preparing
                or self.state.last_observation_at is None
                or self.state.lifecycle not in {LifecycleState.TRADING, LifecycleState.PAUSED}
                or self.state.revision.get("status") == "draining"
            ):
                return
            if self._fitting_task is not None and not self._fitting_task.done():
                return
            if any(not task.done() for task in self._attribution_tasks):
                return
            if self.state.fitting.get("status") == "running":
                self.fail_policy_fitting("previous fitting was interrupted before Policy Handoff")
            now = self._now()
            if not fitter.is_due(now, self.state.fitting):
                return
            self.start_policy_fitting(due_at=now)
            marks = self.state.simulation.previous_marks or dict.fromkeys(self.state.tickers, 0.0)
            current_weights = marked_weights(self.state.simulation, marks, self.state.tickers)
            observed_at = self.state.last_observation_at
            self._fitting_task = asyncio.create_task(
                self._run_policy_fitting(fitter, observed_at, current_weights),
                name="paper-account-policy-fitting",
            )

    async def _run_policy_fitting(
        self,
        fitter: PolicyFitter,
        observed_at: datetime,
        current_weights: dict[str, float],
    ) -> None:
        try:
            candidate = await asyncio.get_running_loop().run_in_executor(
                self._fitting_executor,
                fitter.fit,
                observed_at,
                current_weights,
            )
            with self._owner_lock:
                rollback = fitter.activate(candidate)
                try:
                    self.complete_policy_handoff(
                        new_model_id=candidate.model_id,
                        checkpoint=candidate.checkpoint,
                    )
                except BaseException:
                    rollback()
                    raise
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.fail_policy_fitting(error)

    def start_policy_fitting(self, *, due_at: datetime) -> dict[str, Any]:
        """Record the background-fitting boundary without changing the active Fitted Policy."""
        with self._owner_lock:
            if self.state.fitting.get("status") == "running":
                return self.system_snapshot()
            now = self._now()
            retrying = self.state.fitting.get("status") == "failed"
            prior = self.state.fitting
            self.state.fitting = {
                "status": "running",
                "cycle_at": prior.get("cycle_at", prior.get("due_at"))
                if retrying
                else due_at.astimezone(UTC).isoformat(),
                "due_at": prior.get("due_at") if retrying else due_at.astimezone(UTC).isoformat(),
                "started_at": now.isoformat(),
                "attempt_count": int(prior.get("attempt_count", 0)) + 1 if retrying else 1,
                "last_successful_fit_at": self.state.model_fitted_at,
                "next_retry_at": None,
            }
            self._commit([("PolicyFittingStarted", now, self.state.fitting.copy(), None, None)])
            return self.system_snapshot()

    def fail_policy_fitting(self, error: Exception | str) -> dict[str, Any]:
        """Retain the current Fitted Policy while surfacing an actionable fitting failure."""
        with self._owner_lock:
            now = self._now()
            self.state.fitting = {
                **self.state.fitting,
                "status": "failed",
                "error": str(error),
                "failed_at": now.isoformat(),
                "attempt_count": max(1, int(self.state.fitting.get("attempt_count", 1))),
                "last_successful_fit_at": self.state.model_fitted_at,
                "next_retry_at": (
                    now
                    + timedelta(
                        minutes=(5, 15, 30, 60)[min(3, max(0, int(self.state.fitting.get("attempt_count", 1)) - 1))]
                    )
                ).isoformat(),
            }
            self._queue_notification(
                f"fitting-failed:{self.state.account_id}:{now.date().isoformat()}",
                "fitting_failure",
                self.state.fitting,
                now,
            )
            self._commit([("PolicyFittingFailed", now, self.state.fitting.copy(), None, None)])
            self._deliver_notifications()
            return self.system_snapshot()

    def complete_attribution(
        self,
        *,
        decision_id: str,
        attribution: dict[str, Any],
    ) -> dict[str, Any]:
        """Append approximate influence evidence after its exact Decision Record is durable."""
        with self._owner_lock:
            decisions = {
                event.decision_id
                for event in self.store.events(self.state.account_id)
                if event.event_type == "DecisionRecord"
            }
            if decision_id not in decisions:
                raise KeyError(decision_id)
            now = self._now()
            payload = {
                **attribution,
                "status": "complete",
                "label": "Approximate post-hoc influence evidence",
            }
            self._commit([("ModelAttribution", now, payload, decision_id, None)])
            event = next(
                event
                for event in reversed(self.store.events(self.state.account_id))
                if event.event_type == "ModelAttribution" and event.decision_id == decision_id
            )
            return self.event_snapshot(event.event_id)

    def _schedule_attribution(self, decision: AccountEvent) -> None:
        backend = self.attribution_backend
        if backend is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        input_id = str(decision.payload["input_id"])
        decision_id = decision.decision_id
        if decision_id is None:
            raise RuntimeError("Decision Record lacks its durable decision identity")
        if decision_id in self._attribution_decisions:
            return
        payload = decision.payload.copy()
        try:
            retained = self.store.attribution_input(input_id)
        except KeyError:
            retained = None
        except RuntimeError as error:
            self._commit(
                [
                    (
                        "ModelAttributionFailed",
                        self._now(),
                        {"decision_id": decision_id, "error": str(error)},
                        decision_id,
                        None,
                    )
                ]
            )
            return
        if retained is not None and retained.get("payload") is not None:
            blob = retained["payload"]
            if hashlib.sha256(blob).hexdigest() != retained["input_hash"]:
                self._commit(
                    [
                        (
                            "ModelAttributionFailed",
                            self._now(),
                            {"decision_id": decision_id, "error": "durable attribution input checksum mismatch"},
                            decision_id,
                            None,
                        )
                    ]
                )
                return
            payload["_durable_attribution_input"] = blob
        self._attribution_decisions.add(decision_id)
        task = loop.create_task(
            self._run_attribution(
                backend,
                decision_id,
                input_id,
                payload,
            ),
            name=f"paper-account-attribution-{decision_id}",
        )
        self._attribution_tasks.add(task)

        def finish(completed: asyncio.Task[None]) -> None:
            self._finish_attribution_task(completed, decision_id)

        task.add_done_callback(finish)

    def resume_pending_attributions(self) -> None:
        """Requeue durable Decision Records whose lower-priority explanation was interrupted."""
        with self._owner_lock:
            fitting_active = self._fitting_task is not None and not self._fitting_task.done()
            if self._policy_preparing or self.state.pending_execution is not None or fitting_active:
                return
            events = self.store.events(self.state.account_id)
            terminal = {
                event.decision_id
                for event in events
                if event.event_type in {"ModelAttribution", "ModelAttributionFailed"}
            }
            for event in events:
                if event.event_type == "DecisionRecord" and event.decision_id not in terminal:
                    self._schedule_attribution(event)

    def _finish_attribution_task(self, task: asyncio.Task[None], decision_id: str) -> None:
        self._attribution_tasks.discard(task)
        self._attribution_decisions.discard(decision_id)

    async def _run_attribution(
        self,
        backend: AttributionBackend,
        decision_id: str,
        input_id: str,
        decision: Mapping[str, Any],
    ) -> None:
        try:
            attribution = await asyncio.get_running_loop().run_in_executor(
                self._attribution_executor,
                backend.attribute,
                input_id,
                decision,
            )
            self.complete_attribution(decision_id=decision_id, attribution=attribution)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            with self._owner_lock:
                now = self._now()
                self._commit(
                    [
                        (
                            "ModelAttributionFailed",
                            now,
                            {"decision_id": decision_id, "error": str(error)},
                            decision_id,
                            None,
                        )
                    ]
                )

    def stage_policy_revision(
        self,
        *,
        protocol_id: str,
        compatibility: dict[str, Any],
        candidate: FittedPolicyCandidate,
        validation: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._owner_lock:
            if self.state.revision.get("status") != "draining" and self.state.lifecycle not in {
                LifecycleState.TRADING,
                LifecycleState.PAUSED,
            }:
                raise ValueError("Policy Revision requires a trading or paused account")
            if self._fitting_task is not None and not self._fitting_task.done():
                raise ValueError("Policy Revision cannot overlap fitting")
            if not (
                policy_revision_is_compatible(
                    self.state.compatibility_manifest, compatibility, lifecycle=self.state.lifecycle
                )
                or eligibility_revision_is_supported(self.state.compatibility_manifest, compatibility)
            ):
                raise ValueError("unsupported Policy Revision migration")
            validate_revision_evidence(
                validation,
                protocol_id=protocol_id,
                model_id=candidate.model_id,
                drawdown_limit=self.config.drawdown_limit,
            )
            blob = Path(candidate.checkpoint).read_bytes()
            if hashlib.sha256(blob).hexdigest() != candidate.model_id:
                raise ValueError("candidate checkpoint checksum mismatch")
            if self.state.revision.get("status") == "draining":
                if (
                    self.state.revision.get("protocol_id") != protocol_id
                    or self.state.revision.get("model_id") != candidate.model_id
                ):
                    raise ValueError("another Policy Revision is already draining")
                self._activate_staged_revision()
                return self.system_snapshot()
            if self.state.simulation.previous_marks is None:
                raise ValueError("Policy Revision requires an observed account valuation")
            self.state.revision = {
                "status": "draining",
                "protocol_id": protocol_id,
                "compatibility": compatibility,
                "model_id": candidate.model_id,
                "checkpoint": str(Path(candidate.checkpoint).resolve()),
                "validation": {**validation, "checkpoint": str(Path(candidate.checkpoint).resolve())},
                "requested_at": self._now().isoformat(),
            }
            self._commit([("PolicyRevisionStaged", self._now(), self.state.revision.copy(), None, None)])
            self._activate_staged_revision()
            return self.system_snapshot()

    def _activate_staged_revision(self) -> None:
        if (
            self._policy_preparing
            or self.state.revision.get("status") != "draining"
            or self.state.pending_execution is not None
        ):
            return
        if self.state.lifecycle not in {LifecycleState.TRADING, LifecycleState.PAUSED}:
            return
        if (
            not self._observed_in_window
            or self.state.last_observation_at is None
            or self.state.data_status is not DataStatus.FRESH
            or self._now() - self.state.last_observation_at > timedelta(minutes=2)
        ):
            return
        staged = self.state.revision.copy()
        blob = Path(staged["checkpoint"]).read_bytes()
        if hashlib.sha256(blob).hexdigest() != staged["model_id"]:
            raise ValueError("staged checkpoint checksum mismatch")
        self.register_policy_revision(
            protocol_id=staged["protocol_id"],
            compatibility=staged["compatibility"],
            fitted_policy=FittedPolicyCandidate(staged["model_id"], staged["checkpoint"], blob),
            validated_transition=True,
        )

    def register_policy_revision(
        self,
        *,
        protocol_id: str,
        compatibility: dict[str, Any],
        fitted_policy: FittedPolicyCandidate | None = None,
        model_compatibility: Mapping[str, Any] | None = None,
        validated_transition: bool = False,
    ) -> dict[str, Any]:
        with self._owner_lock:
            now = self._now()
            compatible = policy_revision_is_compatible(
                self.state.compatibility_manifest,
                compatibility,
                lifecycle=self.state.lifecycle,
            )
            if validated_transition:
                if self.state.revision.get("status") != "draining" or self.state.pending_execution is not None:
                    raise ValueError("revision activation requires a drained staged candidate")
                compatible = compatible or eligibility_revision_is_supported(
                    self.state.compatibility_manifest, compatibility
                )
            if compatible:
                revision_changed = (
                    self.state.protocol_id != protocol_id or self.state.compatibility_manifest != compatibility
                )
                model_changed = fitted_policy is not None and fitted_policy.model_id != self.state.model_id
                if not revision_changed and not model_changed:
                    return self.live_snapshot()
                old_protocol_id = self.state.protocol_id
                old_model_id = self.state.model_id
                compatibility_evidence = dict(model_compatibility) if model_compatibility is not None else None
                if (
                    revision_changed
                    and old_model_id != "unknown"
                    and fitted_policy is None
                    and compatibility_evidence is None
                ):
                    raise RuntimeError("a compatible Policy Revision must provide active-model compatibility evidence")
                if compatibility_evidence is not None and compatibility_evidence.get("model_id") != old_model_id:
                    raise ValueError("model compatibility evidence does not identify the active Fitted Policy")
                before_activation = deepcopy(self.state)
                rollback: Callable[[], None] | None = None
                if model_changed:
                    if self.policy_fitter is None or fitted_policy is None:
                        raise RuntimeError("a replacement Fitted Policy requires the configured policy fitter")
                    rollback = self.policy_fitter.activate(fitted_policy)
                try:
                    self.state.protocol_id = protocol_id
                    self.state.compatibility_manifest = compatibility.copy()
                    self.state.proposed_protocol_id = None
                    self.state.proposed_compatibility_manifest = {}
                    if validated_transition:
                        self.state.hold_benchmark = HoldBenchmark.start(
                            self.state.simulation.equity,
                            self.state.simulation.quantities,
                            self.state.simulation.previous_marks or {},
                        )
                        self.state.revision = {
                            **self.state.revision,
                            "status": "active",
                            "activated_at": now.isoformat(),
                        }
                    events: list[tuple[str, datetime, dict[str, Any], str | None, str | None]] = []
                    if revision_changed:
                        events.append(
                            (
                                "PolicyRevision",
                                now,
                                {
                                    "old_protocol_id": old_protocol_id,
                                    "new_protocol_id": protocol_id,
                                    "compatibility": compatibility,
                                    "model_compatibility": compatibility_evidence,
                                    "starting_equity": self.state.simulation.equity,
                                    "hold_benchmark": self._hold_snapshot(self.state.simulation.previous_marks or {}),
                                },
                                None,
                                None,
                            )
                        )
                    if model_changed and fitted_policy is not None:
                        self.state.model_id = fitted_policy.model_id
                        self.state.model_checkpoint = fitted_policy.checkpoint
                        self.state.model_fitted_at = (
                            self.state.revision.get("validation", {}).get("fitted_at")
                            if validated_transition
                            else now.isoformat()
                        )
                        self.state.fitting = {
                            "status": "idle",
                            "completed_at": now.isoformat(),
                            "last_successful_fit_at": now.isoformat(),
                            "attempt_count": 0,
                            "next_retry_at": None,
                        }
                        events.append(
                            (
                                "PolicyHandoff",
                                now,
                                {
                                    "kind": "policy_revision",
                                    "old_model_id": old_model_id,
                                    "new_model_id": fitted_policy.model_id,
                                    "checkpoint": fitted_policy.checkpoint,
                                    "protocol_id": protocol_id,
                                    "due_at": None,
                                    "started_at": None,
                                    "completed_at": now.isoformat(),
                                },
                                None,
                                None,
                            )
                        )
                    self._commit(events)
                except BaseException:
                    if rollback is not None:
                        rollback()
                    self.state = before_activation
                    raise
                self._deliver_notifications()
                return self.live_snapshot()
            else:
                if (
                    self.state.lifecycle is LifecycleState.MIGRATION_REQUIRED
                    and self.state.proposed_protocol_id == protocol_id
                    and self.state.proposed_compatibility_manifest == compatibility
                ):
                    return self.live_snapshot()
                cancelled = self.state.pending_execution
                self.state.lifecycle = LifecycleState.MIGRATION_REQUIRED
                self.state.simulation.pending = None
                self.state.pending_execution = None
                self.state.proposed_protocol_id = protocol_id
                self.state.proposed_compatibility_manifest = compatibility.copy()
                payload = {"proposed_protocol_id": protocol_id, "compatibility": compatibility}
                self._queue_notification(
                    f"migration-required:{self.state.account_id}:{protocol_id}",
                    "incompatible_policy_revision",
                    payload,
                    now,
                )
            events = [("MigrationRequired", now, payload, None, None)]
            if cancelled is not None and cancelled.kind == "policy":
                events.append(
                    (
                        "DecisionCompleted",
                        now,
                        {
                            "decision_id": cancelled.decision_id,
                            "outcome": "cancelled_by_policy_revision",
                        },
                        cancelled.decision_id,
                        None,
                    )
                )
            self._commit(events)
            self._deliver_notifications()
            return self.live_snapshot()

    def live_snapshot(self) -> dict[str, Any]:
        with self._owner_lock:
            now = self._now()
            state = self.state
            marks = state.simulation.previous_marks or dict.fromkeys(state.tickers, 0.0)
            if state.simulation.previous_marks is not None:
                account_snapshot = deepcopy(state.accounting).mark(
                    marks,
                    target_weights=state.target_weights,
                    authoritative_equity=state.simulation.equity,
                )
            else:
                account_snapshot = None
            events = self.store.events(state.account_id)
            material_events = [event for event in events if _is_material_event(event)]
            recent_trades = sorted(
                (event for event in events if event.event_type == "InstrumentFilled"),
                key=lambda event: (event.occurred_at, event.sequence),
                reverse=True,
            )[:5]
            activity = self._activity(events)
            latest_decision = self._latest_decision(events)
            positions = (
                [
                    {"ticker": ticker, **asdict(position)}
                    for ticker, position in account_snapshot.positions.items()
                    if position.side != "flat"
                ]
                if account_snapshot is not None
                else []
            )
            account_metrics = account_snapshot.account if account_snapshot is not None else None
            risk_metrics = account_snapshot.risk if account_snapshot is not None else None
            cash_balance = account_metrics.cash_balance if account_metrics else state.starting_equity
            realized_pnl = account_metrics.realized_pnl if account_metrics else 0.0
            unrealized_pnl = account_metrics.unrealized_pnl if account_metrics else 0.0
            composed_equity = account_metrics.composed_equity if account_metrics else state.starting_equity
            reconciliation_difference = (
                account_metrics.reconciliation_difference
                if account_metrics
                else state.simulation.equity - composed_equity
            )
            last_observation = state.last_observation_at
            freshness_age = max(0.0, (now - last_observation).total_seconds()) if last_observation else None
            current_window = self.store.operating_windows(state.account_id)[-1]
            exposure_notional = sum(
                abs(state.simulation.quantities[ticker] * marks[ticker]) for ticker in state.tickers
            )
            estimated_flatten_cost = exposure_notional * self.config.transaction_cost_rate
            return {
                "as_of": now.isoformat(),
                "trading_universe": list(state.tickers),
                "account": {
                    "id": state.account_id,
                    "state": state.lifecycle.value,
                    "version": state.state_version,
                    "started_at": state.created_at.isoformat(),
                    "starting_equity": state.starting_equity,
                    "current_equity": state.simulation.equity,
                    "cash": cash_balance,
                    "net_pnl": state.simulation.equity - state.starting_equity,
                    "compounded_net_return": state.simulation.equity / state.starting_equity - 1.0,
                    "gross_trading_pnl": account_metrics.gross_trading_pnl if account_metrics else 0.0,
                    "realized_pnl": realized_pnl,
                    "unrealized_pnl": unrealized_pnl,
                    "transaction_cost": state.simulation.transaction_cost,
                    "funding": state.simulation.funding_cashflow,
                    "turnover": state.simulation.turnover_notional,
                    "composed_equity": composed_equity,
                    "reconciliation_difference": reconciliation_difference,
                    "marked_equity_reconciliation": {
                        "formula": "cash_balance + unrealized_pnl + reconciliation_difference",
                        "cash_balance": cash_balance,
                        "unrealized_pnl": unrealized_pnl,
                        "composed_equity": composed_equity,
                        "difference": reconciliation_difference,
                        "authoritative_marked_equity": state.simulation.equity,
                    },
                },
                "risk": {
                    "current_drawdown": risk_metrics.current_drawdown if risk_metrics else 0.0,
                    "maximum_drawdown": state.simulation.max_drawdown,
                    "high_water_equity": state.simulation.high_water,
                    "drawdown_limit": self.config.drawdown_limit,
                    "gross_exposure": risk_metrics.gross_exposure if risk_metrics else 0.0,
                    "net_exposure": risk_metrics.net_exposure if risk_metrics else 0.0,
                    "cash_weight": risk_metrics.cash_weight if risk_metrics else 1.0,
                    "concentrations": (
                        {ticker: abs(weight) for ticker, weight in risk_metrics.instrument_weights.items()}
                        if risk_metrics
                        else dict.fromkeys(state.tickers, 0.0)
                    ),
                },
                "positions": positions,
                "activity": activity,
                "latest_decision": latest_decision,
                "decision_blocker": self._decision_blocker(),
                "freshness": {
                    "status": state.data_status.value,
                    "observed_at": last_observation.isoformat() if last_observation else None,
                    "age_seconds": freshness_age,
                    "error": state.data_error,
                },
                "operating_window": {"id": current_window["window_id"], "started_at": current_window["started_at"]},
                "next_decision_at": self._next_decision_at(now).isoformat(),
                "pending_fill": _pending_payload(state.pending_execution),
                "fitting": state.fitting,
                "revision": state.revision,
                "hold_benchmark": self._hold_snapshot(marks),
                "controls": self._control_availability(),
                "control_estimates": {
                    "pause": {"estimated_cost": 0.0},
                    "resume": {"estimated_cost": 0.0},
                    "flatten": {
                        "current_exposure": exposure_notional,
                        "estimated_cost": estimated_flatten_cost,
                    },
                    "reset": {
                        "current_exposure": exposure_notional,
                        "estimated_cost": estimated_flatten_cost,
                    },
                },
                "recent_events": [self._event_summary(event) for event in reversed(material_events[-20:])],
                "recent_trades": [self._event_summary(event) for event in recent_trades],
            }

    def history_snapshot(self, account_id: str | None = None) -> dict[str, Any]:
        with self._owner_lock:
            return self._history_snapshot(account_id)

    def _history_snapshot(self, account_id: str | None = None) -> dict[str, Any]:
        selected = account_id or self.state.account_id
        accounts = self.store.accounts()
        if selected not in {str(account["account_id"]) for account in accounts}:
            raise KeyError(selected)
        account_rows: list[dict[str, Any]] = []
        comparisons: list[dict[str, Any]] = []
        segments_by_account: dict[str, list[dict[str, Any]]] = {}
        for account in accounts:
            restored_id = str(account["account_id"])
            restored = self.store.load_account(restored_id)
            restored_events = self.store.events(restored_id)
            net_pnl = restored.simulation.equity - restored.starting_equity
            compounded_return = restored.simulation.equity / restored.starting_equity - 1.0
            segments_by_account[restored_id] = self._protocol_performance_segments(restored, restored_events)
            account_rows.append(
                {
                    "id": restored_id,
                    "label": f"Paper Account {restored_id[:8]}",
                    "active": bool(account["active"]),
                    "started_at": account["created_at"],
                    "archived_at": account["archived_at"],
                    "protocol_id": restored.protocol_id,
                    "starting_equity": restored.starting_equity,
                    "current_equity": restored.simulation.equity,
                    "net_pnl": net_pnl,
                    "compounded_net_return": compounded_return,
                }
            )
            comparisons.append(
                {
                    "account_id": restored.account_id,
                    "protocol_id": restored.protocol_id,
                    "starting_equity": restored.starting_equity,
                    "current_equity": restored.simulation.equity,
                    "net_pnl": net_pnl,
                    "compounded_net_return": compounded_return,
                    "maximum_drawdown": restored.simulation.max_drawdown,
                }
            )
        selected_comparison = next(item for item in comparisons if item["account_id"] == selected)
        selected_events = self.store.events(selected)
        return {
            "accounts": account_rows,
            "selected_account_id": selected,
            "events": [self._event_summary(event) for event in selected_events if _is_material_event(event)],
            "activity": self._activity(selected_events),
            "protocol_segments": segments_by_account[selected],
            "comparison": {**selected_comparison, "accounts": comparisons},
        }

    def system_snapshot(self) -> dict[str, Any]:
        with self._owner_lock:
            return self._system_snapshot()

    def _system_snapshot(self) -> dict[str, Any]:
        state = self.state
        now = self._now()
        latest_backup = self.store.latest_backup()
        backup = latest_backup or {"backup_name": None, "created_at": None, "error": state.backup_error}
        created_at = backup["created_at"]
        backup["age_seconds"] = (
            max(0.0, (now - datetime.fromisoformat(str(created_at))).total_seconds())
            if created_at is not None
            else None
        )
        self._collect_notifications()
        notification_health = self.store.notification_health()
        last_handoff = next(
            (
                event.payload
                for event in reversed(self.store.events(state.account_id))
                if event.event_type == "PolicyHandoff"
            ),
            None,
        )
        diagnostics = getattr(self.policy_backend, "diagnostics", None)
        try:
            compute = diagnostics() if callable(diagnostics) else {"backend": "unreported"}
        except Exception as error:
            compute = {"backend": "unavailable", "error": str(error)}
        return {
            "as_of": now.isoformat(),
            "market_feed": {
                "status": state.data_status.value,
                "observed_at": state.last_observation_at.isoformat() if state.last_observation_at else None,
                "error": state.data_error,
            },
            "policy": {
                "protocol_id": state.protocol_id,
                "model_id": state.model_id,
                "model_checkpoint": state.model_checkpoint,
                "fitted_at": state.model_fitted_at,
                "age_seconds": max(0.0, (now - datetime.fromisoformat(state.model_fitted_at)).total_seconds())
                if state.model_fitted_at
                else None,
                "lifecycle": state.lifecycle.value,
                "compatibility": state.compatibility_manifest,
                "proposed_protocol_id": state.proposed_protocol_id,
                "proposed_compatibility": state.proposed_compatibility_manifest,
            },
            "fitting": state.fitting,
            "revision": state.revision,
            "last_policy_handoff": last_handoff,
            "compute": compute,
            "operating_windows": self.store.operating_windows(state.account_id),
            "notifications": notification_health,
            "database": {
                "path": str(self.store.path),
                "journal_mode": self.store.journal_mode,
                "state_version": state.state_version,
            },
            "backup": backup,
            "service": {
                "owner": "single-process",
                "policy_preparing": self._policy_preparing,
                "status": "running" if not self._closed else "stopped",
                "error": state.operator_error,
            },
        }

    def chart_snapshot(
        self,
        ticker: str,
        range_name: str = "full",
        *,
        pixels: int = 900,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        with self._owner_lock:
            return self._chart_snapshot(ticker, range_name, pixels=pixels, account_id=account_id)

    def _chart_snapshot(
        self,
        ticker: str,
        range_name: str = "full",
        *,
        pixels: int = 900,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        state = self.state if account_id is None else self.store.load_account(account_id)
        if ticker not in state.tickers:
            raise KeyError(ticker)
        events = self.store.events(state.account_id)
        start = self._chart_start(range_name, state.account_id)
        events = [event for event in events if start is None or event.occurred_at >= start]
        portfolio: list[dict[str, Any]] = []
        gaps: list[dict[str, Any]] = []
        markers: list[dict[str, Any]] = []
        weights: list[dict[str, Any]] = []
        candles: list[dict[str, Any]] = []
        for event in events:
            payload = event.payload
            if event.event_type in {"AccountMarked", "AccountMarkReconstructed"}:
                portfolio.append(
                    {
                        "time": event.occurred_at.isoformat(),
                        "equity": payload["equity"],
                        "cash_benchmark": payload.get("cash_benchmark", state.starting_equity),
                        "equal_weight_benchmark": payload.get("equal_weight_benchmark"),
                        "hold_benchmark": payload.get("hold_benchmark"),
                        "drawdown": payload["drawdown"],
                        "gross_exposure": payload["gross_exposure"],
                        "net_exposure": payload.get("net_exposure"),
                    }
                )
                mark = payload.get("marks", {}).get(ticker)
                if mark is not None:
                    candle = payload.get("candles", {}).get(ticker, {})
                    candles.append(
                        {
                            "time": event.occurred_at.isoformat(),
                            "open": candle.get("open", mark),
                            "high": candle.get("high", mark),
                            "low": candle.get("low", mark),
                            "close": candle.get("close", mark),
                        }
                    )
                current_weights = payload.get("current_weights", {})
                target_weights = payload.get("target_weights", {})
                if ticker in current_weights and ticker in target_weights:
                    weights.append(
                        {
                            "time": event.occurred_at.isoformat(),
                            "current": current_weights[ticker],
                            "target": target_weights[ticker],
                        }
                    )
            elif event.event_type == "OperatingGap":
                gaps.append({"start": payload["start"], "end": payload["end"]})
            if event.event_type == "DecisionRecord":
                target = payload.get("target_weights", {}).get(ticker, 0.0)
                current = payload.get("current_portfolio", {}).get(ticker, 0.0)
                weights.append({"time": event.occurred_at.isoformat(), "current": current, "target": target})
            marker_applies = event.event_type != "InstrumentFilled" or event.ticker == ticker
            if event.event_type == "FundingApplied":
                marker_applies = ticker in event.payload.get("rates", {})
            if (
                event.event_type
                in {
                    "DecisionRecord",
                    "InstrumentFilled",
                    "MissedExecution",
                    "FundingApplied",
                    "OperatorIntervention",
                }
                and marker_applies
            ):
                marker_price = event.payload.get("reference_price")
                if event.event_type == "FundingApplied":
                    marker_price = event.payload.get("mark_prices", {}).get(ticker)
                marker_type = "Signal" if event.event_type == "DecisionRecord" else event.event_type
                marker_label = (
                    self._instrument_fill_summary(event)[0]
                    if event.event_type == "InstrumentFilled"
                    else _title(marker_type)
                )
                markers.append(
                    {
                        "id": event.event_id,
                        "time": event.occurred_at.isoformat(),
                        "type": marker_type,
                        "label": marker_label,
                        "price": marker_price,
                        "ticker": ticker if event.event_type == "FundingApplied" else event.ticker,
                        "material": True,
                    }
                )
        return {
            "ticker": ticker,
            "range": range_name,
            "candles": _downsample(candles, pixels),
            "weights": _downsample(weights, pixels),
            "portfolio": _downsample(portfolio, pixels),
            "gaps": gaps,
            "markers": markers,
        }

    def _chart_start(self, range_name: str, account_id: str) -> datetime | None:
        if range_name in {"full", "account"}:
            return None
        if range_name in {"current", "current_window"}:
            window = self.store.operating_windows(account_id)[-1]
            return datetime.fromisoformat(str(window["started_at"]))
        if range_name in {"24h", "24-hour"}:
            return self._now() - timedelta(hours=24)
        if range_name in {"7d", "seven-day"}:
            return self._now() - timedelta(days=7)
        raise ValueError(f"unknown chart range {range_name!r}")

    def event_snapshot(self, event_id: str) -> dict[str, Any]:
        with self._owner_lock:
            return self._event_snapshot(event_id)

    def _event_snapshot(self, event_id: str) -> dict[str, Any]:
        events = self.store.events()
        try:
            event = next(item for item in events if item.event_id == event_id)
        except StopIteration as error:
            raise KeyError(event_id) from error
        decision_event = event
        if event.decision_id and event.event_type != "DecisionRecord":
            decision_event = next(
                (
                    item
                    for item in events
                    if item.event_type == "DecisionRecord" and item.decision_id == event.decision_id
                ),
                event,
            )
        related = [item for item in events if item.decision_id and item.decision_id == decision_event.decision_id]
        execution_event = next(
            (item for item in related if item.event_type in {"PortfolioChangeExecuted", "MissedExecution"}),
            None,
        )
        execution = (
            {
                **execution_event.payload,
                "outcome": execution_event.event_type,
                "fills": [item.payload for item in related if item.event_type == "InstrumentFilled"],
            }
            if execution_event is not None
            else None
        )
        decision = dict(decision_event.payload) if decision_event.event_type == "DecisionRecord" else None
        completion = next(
            (item for item in related if item.event_type == "DecisionCompleted"),
            None,
        )
        if decision is not None:
            if execution_event is not None:
                decision["outcome"] = execution_event.event_type
                decision["execution_time"] = execution_event.occurred_at.isoformat()
            elif completion is not None:
                decision["outcome"] = completion.payload.get("outcome", "completed")
        attribution_event = next(
            (item for item in related if item.event_type in {"ModelAttribution", "ModelAttributionFailed"}),
            None,
        )
        if attribution_event is not None:
            attribution = {
                **attribution_event.payload,
                "status": ("complete" if attribution_event.event_type == "ModelAttribution" else "failed"),
                "label": "Approximate post-hoc influence evidence",
                "event_id": attribution_event.event_id,
            }
        else:
            attribution = (decision or {}).get(
                "attribution",
                {
                    "status": "not_applicable",
                    "label": "Approximate post-hoc influence evidence",
                    "top_influences": [],
                },
            )
        if attribution.get("status") == "pending":
            attribution = {
                **attribution,
                "label": "Approximate post-hoc influence evidence",
                "top_influences": [],
            }
        if decision is not None:
            decision["attribution"] = {
                "status": attribution.get("status", "not_applicable"),
                "event_id": attribution_event.event_id if attribution_event is not None else None,
            }
        return {
            **self._event_summary(event),
            "details": event.payload,
            "decision": decision,
            "execution": execution,
            "attribution": attribution,
        }

    def _commit(
        self,
        events: list[tuple[str, datetime, dict[str, Any], str | None, str | None]],
        *,
        idempotency: tuple[str, str, str, dict[str, Any], datetime] | None = None,
        operating_window: tuple[str, datetime, datetime | None] | None = None,
        close_window: tuple[str, datetime, str] | None = None,
    ) -> list[AccountEvent]:
        if not events:
            if self._pending_notifications:
                raise RuntimeError("notifications require the account event that caused them")
            return []
        expected = self.state.state_version
        self.state.state_version += 1
        self.state.snapshot_version += 1
        notifications = self._pending_notifications
        self._pending_notifications = []
        try:
            attribution_inputs = []
            export = getattr(self.policy_backend, "export_attribution_input", None)
            for kind, occurred_at, payload, decision_id, _ in events:
                if kind == "DecisionRecord" and export is not None and decision_id is not None:
                    input_id = str(payload["input_id"])
                    blob = export(input_id)
                    digest = hashlib.sha256(blob).hexdigest()
                    payload["attribution_input_hash"] = digest
                    attribution_inputs.append(
                        (input_id, decision_id, str(payload["model_id"]), digest, blob, occurred_at)
                    )
            return self.store.commit(
                self.state,
                events,
                expected_state_version=expected,
                idempotency=idempotency,
                operating_window=operating_window,
                close_window=close_window,
                notifications=notifications,
                attribution_inputs=attribution_inputs,
            )
        except BaseException:
            restored = self.store.load_active_account()
            if restored is None:
                self.state.state_version = expected
                self.state.snapshot_version -= 1
            else:
                self.state = restored
            raise

    def _schedule_control_fill(self, kind: str, now: datetime) -> None:
        target = dict.fromkeys(self.config.tickers, 0.0)
        decision_id = f"operator:{kind}:{hashlib.sha256(f'{self.state.account_id}:{now}'.encode()).hexdigest()[:16]}"
        self.state.simulation.pending = PendingPortfolioChange(now, target, forced=True)
        self.state.pending_execution = PendingExecution(
            decision_id=decision_id,
            signal_time=now,
            eligible_at=now,
            expires_at=datetime.max.replace(tzinfo=UTC),
            target_weights=target,
            kind=kind,
        )
        self.state.pending_control = kind

    def _complete_reset(
        self,
        now: datetime,
        *,
        include_intervention: bool = False,
        idempotency: tuple[str, str, str, datetime] | None = None,
    ) -> None:
        self.state, self.window_id = self.store.reset_and_replace(
            self.state,
            now,
            window_id=self.window_id,
            starting_equity=self.config.initial_equity,
            include_intervention=include_intervention,
            idempotency=idempotency,
        )
        self._deliver_notifications()

    def _queue_notification(self, dedupe_key: str, kind: str, payload: dict[str, Any], now: datetime) -> None:
        self._pending_notifications.append((dedupe_key, kind, payload, now))

    def _collect_notifications(self) -> None:
        for notification_id, job in list(self._notification_jobs.items()):
            if not job.done():
                continue
            error = None
            try:
                result = job.result()
                if inspect.isawaitable(result):
                    raise RuntimeError("notification sink must be synchronous")
            except Exception as exception:
                error = str(exception)
            self.store.mark_notification(notification_id, self._now(), error)
            self.state.notification_error = error
            del self._notification_jobs[notification_id]

    def _deliver_notifications(self) -> None:
        self._collect_notifications()
        for row in self.store.pending_notifications(self._now()):
            notification_id = str(row["notification_id"])
            if notification_id not in self._notification_jobs:
                self._notification_jobs[notification_id] = self._notification_executor.submit(
                    self.notifications.notify, str(row["kind"]), json.loads(str(row["payload"]))
                )
        self._collect_notifications()

    def _coerce_decision(
        self,
        supplied: PolicyDecision | Mapping[str, float],
        observation: MarketObservation,
        current_weights: dict[str, float],
    ) -> PolicyDecision:
        if isinstance(supplied, PolicyDecision):
            return supplied
        target = {ticker: float(supplied.get(ticker, current_weights[ticker])) for ticker in self.config.tickers}
        return PolicyDecision(target, target, "unknown", observation.input_id)

    def _recover_before(
        self,
        observation: MarketObservation,
    ) -> list[MarketObservation] | Awaitable[list[MarketObservation]]:
        if self.state.last_observation_at is None:
            return []
        recover = getattr(self.market_feed, "recover", None)
        if recover is None:
            return []
        result = recover(self.state.last_observation_at, observation.timestamp)
        return cast(list[MarketObservation] | Awaitable[list[MarketObservation]], result)

    def _validate_target(self, target: Mapping[str, float]) -> None:
        if set(target) != set(self.config.tickers):
            raise ValueError("Target Weights must cover the fixed Trading Universe")
        if any(abs(weight) > self.config.max_instrument_weight + 1e-6 for weight in target.values()):
            raise ValueError("Target Weights exceed the concentration limit")
        if sum(abs(weight) for weight in target.values()) > self.config.max_gross_exposure + 1e-6:
            raise ValueError("Target Weights exceed No Leverage")

    def _hold_snapshot(self, marks: Mapping[str, float]) -> dict[str, float | str | None] | None:
        if self.state.hold_benchmark is None:
            return None
        result = deepcopy(self.state.hold_benchmark).mark(marks)
        excess_pnl = self.state.simulation.equity - result["equity"]
        return {
            **result,
            "excess_pnl": excess_pnl,
            "excess_return": excess_pnl / result["starting_equity"],
            "started_at": self._hold_benchmark_started_at(),
        }

    def _hold_benchmark_started_at(self) -> str | None:
        revision_start = self.state.revision.get("activated_at")
        if revision_start is not None:
            return str(revision_start)
        for event in reversed(self.store.events(self.state.account_id)):
            if event.event_type == "PolicyRevision":
                return event.occurred_at.isoformat()
        return None

    def _is_flat(self) -> bool:
        return all(abs(quantity) <= 1e-12 for quantity in self.state.simulation.quantities.values())

    @staticmethod
    def _require_confirmation(value: str | bool | None, expected: str) -> None:
        if value is True:
            return
        if not isinstance(value, str) or value.strip().upper() != expected:
            raise ValueError(f"confirmation {expected!r} is required")

    def _activity(self, events: list[AccountEvent]) -> dict[str, int]:
        event_types = [event.event_type for event in events]
        decisions = [event for event in events if event.event_type == "DecisionRecord"]
        return {
            "decisions": len(decisions),
            "executable_changes": sum(
                event.payload.get("kind") == "policy"
                for event in events
                if event.event_type == "PortfolioChangeExecuted"
            ),
            "fills": event_types.count("InstrumentFilled"),
            "below_threshold": sum(
                event.payload.get("outcome") == "below_threshold"
                for event in events
                if event.event_type == "DecisionCompleted"
            ),
            "unchanged_targets": sum(event.payload.get("threshold_outcome") == "unchanged" for event in decisions),
            "missed_executions": event_types.count("MissedExecution"),
            "interventions": event_types.count("OperatorIntervention"),
        }

    def _decision_blocker(self) -> str | None:
        if self.state.lifecycle is not LifecycleState.TRADING:
            return "account_not_trading"
        if self.state.data_status is not DataStatus.FRESH:
            return "market_data_unavailable"
        if self._policy_preparing:
            return "policy_preparing"
        if self.state.revision.get("status") == "draining":
            return "revision_draining"
        if self.state.pending_execution is not None:
            return "pending_execution"
        return None

    @staticmethod
    def _latest_decision(events: list[AccountEvent]) -> dict[str, str] | None:
        decision = next(
            (event for event in reversed(events) if event.event_type == "DecisionRecord"),
            None,
        )
        if decision is None or decision.decision_id is None:
            return None

        related = [event for event in events if event.decision_id == decision.decision_id]
        completion = next(
            (event for event in reversed(related) if event.event_type == "DecisionCompleted"),
            None,
        )
        if any(event.event_type == "PortfolioChangeExecuted" for event in related):
            execution_status = "executed"
        elif any(event.event_type == "MissedExecution" for event in related):
            execution_status = "missed"
        elif completion is not None:
            if str(completion.payload.get("outcome", "")).startswith("cancelled"):
                execution_status = "cancelled"
            else:
                execution_status = "not_required"
        elif any(
            event.sequence > decision.sequence
            and (
                event.event_type == "RiskStop"
                or (
                    event.event_type == "OperatorIntervention"
                    and event.payload.get("action") in {"pause", "flatten_and_pause", "manual_reset"}
                )
            )
            for event in events
        ):
            execution_status = "cancelled"
        else:
            execution_status = "pending"

        return {
            "event_id": decision.event_id,
            "decision_id": decision.decision_id,
            "signal_time": str(decision.payload.get("signal_time", decision.occurred_at.isoformat())),
            "threshold_outcome": str(decision.payload.get("threshold_outcome", "unknown")),
            "execution_status": execution_status,
        }

    @staticmethod
    def _protocol_performance_segments(
        state: PaperAccountState,
        events: list[AccountEvent],
    ) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        last_equity = state.starting_equity

        def start(protocol_id: str, started_at: datetime) -> dict[str, Any]:
            return {
                "protocol_id": protocol_id,
                "started_at": started_at.isoformat(),
                "ended_at": None,
                "starting_equity": last_equity,
                "current_equity": last_equity,
                "net_pnl": 0.0,
                "compounded_net_return": 0.0,
                "maximum_drawdown": 0.0,
                "decisions": 0,
                "executable_changes": 0,
                "_high_water": last_equity,
                "hold_benchmark": None,
                "transaction_cost": 0.0,
                "funding": 0.0,
                "turnover": 0.0,
                "missed_executions": 0,
                "below_threshold": 0,
                "interventions": 0,
            }

        for event in events:
            if event.event_type == "PolicyRevision":
                if current is not None:
                    current["ended_at"] = event.occurred_at.isoformat()
                    segments.append(current)
                protocol_id = str(event.payload.get("new_protocol_id", state.protocol_id))
                last_equity = float(event.payload.get("starting_equity", last_equity))
                current = start(protocol_id, event.occurred_at)
                current["hold_benchmark"] = event.payload.get("hold_benchmark")
                continue
            if current is None and event.event_type in {
                "AccountMarked",
                "AccountMarkReconstructed",
                "DecisionRecord",
                "PortfolioChangeExecuted",
            }:
                current = start(state.protocol_id, state.created_at)
            if current is None:
                continue
            if event.event_type in {"AccountMarked", "AccountMarkReconstructed"}:
                last_equity = float(event.payload["equity"])
                high_water = max(float(current["_high_water"]), last_equity)
                current["_high_water"] = high_water
                current["current_equity"] = last_equity
                current["hold_benchmark"] = event.payload.get("hold_benchmark")
                current["gross_exposure"] = event.payload.get("gross_exposure")
                current["net_pnl"] = last_equity - float(current["starting_equity"])
                current["compounded_net_return"] = last_equity / float(current["starting_equity"]) - 1.0
                current["maximum_drawdown"] = max(
                    float(current["maximum_drawdown"]),
                    1.0 - last_equity / high_water,
                )
            elif event.event_type == "FundingApplied":
                current["funding"] += float(event.payload["cashflow"])
            elif event.event_type == "InstrumentFilled":
                current["turnover"] += abs(float(event.payload["quantity"]) * float(event.payload["reference_price"]))
            elif event.event_type == "MissedExecution":
                current["missed_executions"] += 1
            elif event.event_type == "DecisionCompleted" and event.payload.get("outcome") == "below_threshold":
                current["below_threshold"] += 1
            elif event.event_type == "DecisionRecord":
                current["decisions"] = int(current["decisions"]) + 1
            elif event.event_type == "PortfolioChangeExecuted":
                if event.payload.get("kind") == "policy":
                    current["executable_changes"] = int(current["executable_changes"]) + 1
                current["transaction_cost"] += float(event.payload.get("transaction_cost", 0.0))
            elif event.event_type == "OperatorIntervention":
                current["interventions"] += 1

        if current is None:
            current = start(state.protocol_id, state.created_at)
        segments.append(current)
        for segment in segments:
            segment.pop("_high_water")
        return segments

    def _event_summary(self, event: AccountEvent) -> dict[str, Any]:
        summary = event.event_type
        title = _title(event.event_type)
        trade = None
        if event.event_type == "DecisionRecord":
            summary = f"Target turnover {event.payload.get('projected_turnover', 0.0):.2%}"
        elif event.event_type == "InstrumentFilled":
            title, trade = self._instrument_fill_summary(event)
            summary = f"{trade['quantity']:.8g} {event.ticker} at {trade['price']:.8g}; cost {trade['cost']:.8g}"
        result: dict[str, Any] = {
            "id": event.event_id,
            "type": event.event_type,
            "time": event.occurred_at.isoformat(),
            "display_time": {
                "kyiv": event.occurred_at.astimezone(KYIV_TIME_ZONE).isoformat(),
                "utc": event.occurred_at.astimezone(UTC).isoformat(),
            },
            "title": title,
            "summary": summary,
            "ticker": event.ticker,
            "decision_id": event.decision_id,
            "sequence": event.sequence,
        }
        if trade is not None:
            result["trade"] = trade
        return result

    @staticmethod
    def _instrument_fill_summary(event: AccountEvent) -> tuple[str, dict[str, str | float]]:
        quantity = float(event.payload["quantity"])
        effective_fill = float(event.payload["effective_fill"])
        reference_price = float(event.payload["reference_price"])
        side = "buy" if quantity > 0.0 else "sell"
        trade: dict[str, str | float] = {
            "side": side,
            "quantity": abs(quantity),
            "price": effective_fill,
            "cost": abs(quantity) * abs(effective_fill - reference_price),
        }
        return f"{side.title()} {event.ticker}", trade

    def _control_availability(self) -> dict[str, bool]:
        lifecycle = self.state.lifecycle
        return {
            "pause": lifecycle in {LifecycleState.TRADING, LifecycleState.PAUSED},
            "resume": lifecycle in {LifecycleState.PAUSED, LifecycleState.TRADING}
            and self.state.pending_control is None,
            "flatten": lifecycle in {LifecycleState.TRADING, LifecycleState.PAUSED},
            "reset": lifecycle is not LifecycleState.MIGRATION_REQUIRED,
        }

    @staticmethod
    def _next_decision_at(now: datetime) -> datetime:
        minute = (now.minute // 15 + 1) * 15
        if minute == 60:
            return now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return now.replace(minute=minute, second=0, microsecond=0)

    def _now(self) -> datetime:
        now = self.clock.now()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("application clock must return timezone-aware UTC")
        return now.astimezone(UTC)

    def _check_daily_backup(self) -> None:
        """Attempt the active UTC day's online backup without stopping trading."""
        with self._owner_lock:
            now = self._now()
            try:
                self.store.create_daily_backup(now)
                self.state.backup_error = None
            except Exception as error:
                message = str(error)
                self.state.backup_error = message
                # The in-memory overlay still makes a broken persistence path visible.
                with suppress(Exception):
                    self.store.record_backup_failure(now, message)


def create_application(
    *,
    database_path: Path | str,
    tickers: tuple[str, ...] = DEFAULT_TICKERS,
    clock: Clock | None = None,
    market_feed: MarketFeed | None = None,
    policy_backend: PolicyBackend | None = None,
    notifications: NotificationSink | None = None,
    policy_fitter: PolicyFitter | None = None,
    attribution_backend: AttributionBackend | None = None,
    simulation_config: SimulationConfig | None = None,
    starting_equity: float = 10_000.0,
    backup_directory: Path | None = None,
    static_directory: Path | None = None,
    operator_interval_seconds: float | None = None,
    policy_preparer: Callable[[], object] | None = None,
) -> FastAPI:
    """Construct the complete deterministic ASGI seam used by production and tests."""
    paper = PaperDashboardApplication(
        database_path=database_path,
        tickers=tickers,
        clock=clock,
        market_feed=market_feed,
        policy_backend=policy_backend,
        notifications=notifications,
        policy_fitter=policy_fitter,
        attribution_backend=attribution_backend,
        simulation_config=simulation_config,
        starting_equity=starting_equity,
        backup_directory=backup_directory,
    )

    paper._policy_preparing = policy_preparer is not None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        task: asyncio.Task[None] | None = None
        preparation_task: asyncio.Task[None] | None = None
        if policy_preparer is not None:
            preparation_task = asyncio.create_task(
                _run_policy_preparation(paper, policy_preparer),
                name="paper-account-policy-preparation",
            )
        if operator_interval_seconds is not None:
            task = asyncio.create_task(
                _operator_loop(paper, operator_interval_seconds),
                name="paper-account-operator",
            )
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if preparation_task is not None:
                preparation_task.cancel()
                await asyncio.gather(preparation_task, return_exceptions=True)
            if paper._fitting_task is not None:
                paper._fitting_task.cancel()
                await asyncio.gather(paper._fitting_task, return_exceptions=True)
            for attribution_task in tuple(paper._attribution_tasks):
                attribution_task.cancel()
            if paper._attribution_tasks:
                await asyncio.gather(*paper._attribution_tasks, return_exceptions=True)
            paper.close()

    app = FastAPI(title="Net Growth Paper Account", lifespan=lifespan)
    app.state.paper_dashboard = paper

    @app.get("/api/live")
    def live() -> dict[str, Any]:
        return paper.live_snapshot()

    @app.get("/api/history")
    def history(account_id: str | None = None) -> dict[str, Any]:
        try:
            return paper.history_snapshot(account_id)
        except KeyError as error:
            raise HTTPException(404, "Paper Account not found") from error

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        return paper.system_snapshot()

    @app.get("/api/chart")
    def chart(
        ticker: str = tickers[0],
        range: str = "full",  # noqa: A002
        pixels: int = 900,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            return paper.chart_snapshot(
                ticker,
                range,
                pixels=pixels,
                account_id=account_id,
            )
        except KeyError as error:
            raise HTTPException(404, "ticker not found") from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/events/{event_id}")
    def event(event_id: str) -> dict[str, Any]:
        try:
            return paper.event_snapshot(event_id)
        except KeyError as error:
            raise HTTPException(404, "event not found") from error

    @app.get("/api/csrf")
    async def csrf(request: Request) -> dict[str, str]:
        return {"token": _csrf_token(request)}

    @app.post("/api/controls/{action}")
    def control(action: str, command: ControlRequest, request: Request) -> dict[str, Any]:
        _protect_control_request(request)
        key = request.headers.get("x-idempotency-key")
        if not key:
            raise HTTPException(400, "X-Idempotency-Key is required")
        try:
            return paper.control(
                action,
                expected_version=command.expected_version,
                idempotency_key=key,
                confirmation=command.confirmation,
            )
        except StateVersionConflict as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    roots = static_directory or Path(__file__).with_name("static")

    @app.get("/{page:path}", response_model=None)
    async def dashboard(page: str) -> FileResponse | HTMLResponse:
        relative = "index.html" if page in {"", "live", "history", "system"} else page
        candidate = (roots / relative).resolve()
        if candidate.is_relative_to(roots.resolve()) and candidate.is_file():
            return FileResponse(candidate, headers={"Cache-Control": "no-cache"})
        if page in {"", "live", "history", "system"}:
            return HTMLResponse(
                "<!doctype html><title>Paper Account</title><main><h1>Paper Account</h1>"
                "<p>Use the local dashboard API.</p></main>"
            )
        raise HTTPException(404, "asset not found")

    return app


create_app = create_application


async def _await_observation(value: Awaitable[MarketObservation]) -> MarketObservation:
    return await value


async def _await_recovery(value: Awaitable[list[MarketObservation]]) -> list[MarketObservation]:
    return await value


async def _operator_loop(paper: PaperDashboardApplication, interval_seconds: float) -> None:
    if interval_seconds <= 0.0:
        raise ValueError("operator interval must be positive")
    startup_delay = _operator_start_delay(paper, interval_seconds)
    if startup_delay > 0.0:
        await asyncio.sleep(startup_delay)
    backoff = interval_seconds
    while True:
        try:
            advance = asyncio.create_task(
                asyncio.to_thread(paper.advance_once_sync),
                name="paper-account-market-advance",
            )
            try:
                snapshot = await asyncio.shield(advance)
            except asyncio.CancelledError:
                # The provider call is synchronous and cannot be interrupted safely. Keep
                # shutdown ordered behind its durable transition before closing SQLite.
                await asyncio.shield(advance)
                raise
            await paper.maybe_start_policy_fitting()
            paper.resume_pending_attributions()
            backoff = (
                interval_seconds
                if snapshot["freshness"]["status"] == DataStatus.FRESH.value
                else min(
                    max(interval_seconds, backoff * 2.0),
                    300.0,
                )
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            paper.record_operator_error(error)
            backoff = min(max(interval_seconds, backoff * 2.0), 300.0)
        await asyncio.sleep(backoff)


async def _run_policy_preparation(
    paper: PaperDashboardApplication,
    prepare: Callable[[], object],
) -> None:
    work = asyncio.create_task(asyncio.to_thread(prepare), name="paper-account-policy-preparation-work")
    try:
        await asyncio.shield(work)
    except asyncio.CancelledError:
        await asyncio.shield(work)
        raise
    except Exception as error:
        paper.record_operator_error(f"policy preparation failed: {error}")
    finally:
        paper._policy_preparing = False


def _operator_start_delay(paper: PaperDashboardApplication, interval_seconds: float) -> float:
    last_observation = paper.state.last_observation_at
    if last_observation is None:
        return 0.0
    publication_boundary = last_observation + timedelta(seconds=interval_seconds + 2.0)
    return max(0.0, (publication_boundary - paper._now()).total_seconds())


def _pending_payload(pending: PendingExecution | None) -> dict[str, Any] | None:
    if pending is None:
        return None
    return {
        "decision_id": pending.decision_id,
        "signal_time": pending.signal_time.isoformat(),
        "eligible_at": pending.eligible_at.isoformat(),
        "expires_at": pending.expires_at.isoformat(),
        "target_weights": pending.target_weights,
        "kind": pending.kind,
    }


def _downsample(points: list[dict[str, Any]], pixels: int) -> list[dict[str, Any]]:
    limit = max(1, pixels)
    if len(points) <= limit:
        return points
    step = ceil(len(points) / limit)
    sampled = points[::step]
    if sampled[-1] is not points[-1]:
        sampled.append(points[-1])
    return sampled


def _title(event_type: str) -> str:
    names = {
        "FlatStart": "Flat Start",
        "Signal": "Policy signal",
        "AccountMarked": "Account mark",
        "AccountMarkReconstructed": "Reconstructed account mark",
        "DecisionRecord": "Decision Record",
        "PortfolioChangeExecuted": "Executable Portfolio Change",
        "InstrumentFilled": "Instrument fill",
        "MissedExecution": "Missed Execution",
        "OperatorIntervention": "Operator Intervention",
        "RiskStop": "Risk Stop",
        "ManualResetCompleted": "Manual Reset completed",
        "OperatingGap": "Operating gap",
    }
    return names.get(event_type, event_type)


def _csrf_token(request: Request) -> str:
    paper = cast(PaperDashboardApplication, request.app.state.paper_dashboard)
    return paper.csrf_token


def _protect_control_request(request: Request) -> None:
    content_type = request.headers.get("content-type", "")
    if not content_type.lower().startswith("application/json"):
        raise HTTPException(415, "controls require application/json")
    origin = request.headers.get("origin")
    expected_origin = str(request.base_url).rstrip("/")
    if origin != expected_origin:
        raise HTTPException(403, "same-origin request required")
    if request.headers.get("x-csrf-token") != _csrf_token(request):
        raise HTTPException(403, "valid X-CSRF-Token is required")
