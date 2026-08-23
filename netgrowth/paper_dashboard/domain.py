"""Domain records owned by the persistent Paper Account application."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

from netgrowth.accounting import AccountAccounting, PassiveBenchmarks
from netgrowth.simulation import SimulationState


class LifecycleState(StrEnum):
    """Durable Paper Account lifecycle, separate from operational overlays."""

    TRADING = "Trading"
    PAUSED = "Paused"
    RESET_PENDING = "Reset Pending"
    RISK_STOPPED = "Risk Stopped"
    MIGRATION_REQUIRED = "Migration Required"


class DataStatus(StrEnum):
    FRESH = "Fresh"
    STALE = "Data Stale"


@dataclass(frozen=True, slots=True)
class AccountEvent:
    event_id: str
    sequence: int
    account_id: str
    event_type: str
    occurred_at: datetime
    payload: dict[str, Any] = field(default_factory=dict)
    decision_id: str | None = None
    ticker: str | None = None


@dataclass(frozen=True, slots=True)
class PendingExecution:
    decision_id: str
    signal_time: datetime
    eligible_at: datetime
    expires_at: datetime
    target_weights: dict[str, float]
    kind: str = "policy"


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    raw_target_weights: dict[str, float]
    target_weights: dict[str, float]
    model_id: str
    input_id: str
    protocol_id: str = "unknown"
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MarketObservation:
    """One execution-complete public observation supplied to ``advance_once``."""

    timestamp: datetime
    mark_prices: dict[str, float]
    bid: dict[str, float]
    ask: dict[str, float]
    quote_exchange_times: dict[str, datetime]
    quote_observed_at: datetime
    funding_rates: dict[str, float] = field(default_factory=dict)
    funding_mark_prices: dict[str, float] = field(default_factory=dict)
    input_id: str = "unknown"
    decision_bar_closed: bool | None = None
    reconstructed: bool = False
    candles: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_utc(self.timestamp, "observation timestamp")
        _require_utc(self.quote_observed_at, "quote observation time")
        for value in self.quote_exchange_times.values():
            _require_utc(value, "quote exchange time")

    @property
    def closes_decision_bar(self) -> bool:
        if self.decision_bar_closed is not None:
            return self.decision_bar_closed
        return self.timestamp.second == 0 and self.timestamp.minute % 15 == 0


@dataclass(slots=True)
class PaperAccountState:
    account_id: str
    created_at: datetime
    starting_equity: float
    tickers: tuple[str, ...]
    simulation: SimulationState
    accounting: AccountAccounting
    benchmarks: PassiveBenchmarks | None = None
    lifecycle: LifecycleState = LifecycleState.TRADING
    state_version: int = 1
    snapshot_version: int = 1
    target_weights: dict[str, float] = field(default_factory=dict)
    pending_execution: PendingExecution | None = None
    pending_control: str | None = None
    last_observation_at: datetime | None = None
    last_decision_at: datetime | None = None
    data_status: DataStatus = DataStatus.FRESH
    data_error: str | None = None
    stale_since: datetime | None = None
    protocol_id: str = "unknown"
    model_id: str = "unknown"
    model_checkpoint: str | None = None
    compatibility_manifest: dict[str, Any] = field(default_factory=dict)
    fitting: dict[str, Any] = field(default_factory=lambda: {"status": "idle"})
    notification_error: str | None = None
    backup_error: str | None = None
    operator_error: str | None = None
    active: bool = True

    @classmethod
    def flat_start(
        cls,
        account_id: str,
        created_at: datetime,
        tickers: tuple[str, ...],
        *,
        starting_equity: float = 10_000.0,
    ) -> PaperAccountState:
        _require_utc(created_at, "Flat Start time")
        return cls(
            account_id=account_id,
            created_at=created_at,
            starting_equity=starting_equity,
            tickers=tickers,
            simulation=SimulationState(
                quantities=dict.fromkeys(tickers, 0.0),
                equity=starting_equity,
                high_water=starting_equity,
                max_instrument_exposure=dict.fromkeys(tickers, 0.0),
            ),
            accounting=AccountAccounting.flat_start(
                tickers,
                starting_equity=starting_equity,
            ),
            target_weights=dict.fromkeys(tickers, 0.0),
        )


class Clock(Protocol):
    def now(self) -> datetime: ...


@dataclass(frozen=True, slots=True)
class SystemClock:
    def now(self) -> datetime:
        return datetime.now(tz=UTC)


def _require_utc(value: datetime, label: str) -> None:
    offset = value.utcoffset()
    if value.tzinfo is None or offset is None:
        raise ValueError(f"{label} must be timezone-aware UTC")
    if offset.total_seconds() != 0:
        raise ValueError(f"{label} must use UTC")
