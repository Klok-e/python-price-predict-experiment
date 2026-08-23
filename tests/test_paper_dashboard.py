from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from netgrowth.config import load_config
from netgrowth.paper_dashboard.application import (
    FittedPolicyCandidate,
    _operator_loop,
    _operator_start_delay,
    create_application,
)
from netgrowth.paper_dashboard.domain import MarketObservation, PolicyDecision
from netgrowth.paper_dashboard.production import ProductionMarketFeed, ProductionPolicyBackend

TICKERS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")


@dataclass
class FakeClock:
    current: datetime

    def now(self) -> datetime:
        return self.current


@dataclass
class FakeFeed:
    observations: list[MarketObservation]

    def observe(self, after: datetime | None) -> MarketObservation:
        del after
        return self.observations.pop(0)


@dataclass
class UnavailableFeed:
    error: Exception

    def observe(self, after: datetime | None) -> MarketObservation:
        del after
        raise self.error


@dataclass
class RecoveringFeed:
    current: MarketObservation
    reconstructed: list[MarketObservation]

    def observe(self, after: datetime | None) -> MarketObservation:
        del after
        return self.current

    def recover(self, after: datetime, before: datetime) -> list[MarketObservation]:
        assert after < before
        return self.reconstructed


@dataclass
class FakePolicy:
    targets: list[dict[str, float]]

    def decide(self, observation: MarketObservation, current_weights: dict[str, float]) -> PolicyDecision:
        del current_weights
        target = self.targets.pop(0)
        return PolicyDecision(target, target, "model-1", observation.input_id, "protocol-1")


@dataclass
class NotificationRecorder:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def notify(self, kind: str, payload: dict[str, Any]) -> None:
        self.calls.append((kind, payload))


@dataclass
class FailingNotifications:
    message: str = "desktop bus unavailable"

    def notify(self, kind: str, payload: dict[str, Any]) -> None:
        del kind, payload
        raise RuntimeError(self.message)


@dataclass
class FakeFitter:
    checkpoint: Path
    activated: list[FittedPolicyCandidate] = field(default_factory=list)

    def is_due(self, now: datetime, fitting: dict[str, Any]) -> bool:
        del now
        return fitting.get("status") == "idle"

    def fit(self, observed_at: datetime, current_weights: dict[str, float]) -> FittedPolicyCandidate:
        assert observed_at == observation(1).timestamp
        assert current_weights == dict.fromkeys(TICKERS, 0.0)
        return FittedPolicyCandidate("model-2", str(self.checkpoint), b"model-2")

    def activate(self, candidate: FittedPolicyCandidate):
        self.activated.append(candidate)

        def rollback() -> None:
            self.activated.remove(candidate)

        return rollback


@dataclass
class FakeAttribution:
    calls: list[str] = field(default_factory=list)

    def attribute(self, input_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        assert decision["input_id"] == input_id
        self.calls.append(input_id)
        return {
            "method": "Integrated Gradients",
            "parameters": {"steps": 8, "top_k": 3},
            "input_hash": f"hash:{input_id}",
            "model_hash": "model-hash",
            "top_influences": [{"label": "BTC momentum", "value": 0.25}],
        }


def observation(offset: int, *, decision: bool = False) -> MarketObservation:
    timestamp = datetime(2026, 8, 23, 18, 0, tzinfo=UTC) + timedelta(minutes=offset)
    prices = dict.fromkeys(TICKERS, 100.0)
    return MarketObservation(
        timestamp=timestamp,
        mark_prices=prices,
        bid=prices,
        ask=prices,
        quote_exchange_times=dict.fromkeys(TICKERS, timestamp),
        quote_observed_at=timestamp,
        input_id=f"input-{offset}",
        decision_bar_closed=decision,
    )


def make_app(database: Path, clock: FakeClock, feed: FakeFeed, policy: FakePolicy):
    return create_application(
        database_path=database,
        tickers=TICKERS,
        clock=clock,
        market_feed=feed,
        policy_backend=policy,
        notifications=NotificationRecorder(),
    )


async def control_headers(client: httpx.AsyncClient, key: str) -> dict[str, str]:
    token = (await client.get("/api/csrf")).json()["token"]
    return {
        "Origin": "http://test",
        "X-CSRF-Token": token,
        "X-Idempotency-Key": key,
    }


@pytest.mark.anyio
async def test_complete_application_marks_decides_fills_and_restores_the_same_account(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
    app = make_app(database, clock, FakeFeed([observation(0, decision=True), observation(1)]), FakePolicy([target]))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        initial = (await client.get("/api/live")).json()
        account_id = initial["account"]["id"]
        assert initial["account"]["current_equity"] == 10_000.0
        assert initial["positions"] == []

        app.state.paper_dashboard.advance_once_sync()
        scheduled = (await client.get("/api/live")).json()
        assert scheduled["pending_fill"]["decision_id"]
        assert scheduled["activity"]["decisions"] == 1

        app.state.paper_dashboard.advance_once_sync()
        filled = (await client.get("/api/live")).json()
        assert filled["positions"][0]["ticker"] == "BTCUSDT"
        assert filled["positions"][0]["average_entry"] == 100.0
        assert filled["positions"][0]["unrealized_pnl"] == 0.0
        assert filled["account"]["transaction_cost"] > 0.0
        assert filled["activity"]["fills"] == 1
        assert {event["type"] for event in filled["recent_events"]} >= {
            "PortfolioChangeExecuted",
            "InstrumentFilled",
        }
    app.state.paper_dashboard.close()

    clock.current += timedelta(hours=1)
    restored_app = make_app(database, clock, FakeFeed([]), FakePolicy([]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restored_app), base_url="http://test") as client:
        restored = (await client.get("/api/live")).json()
        assert restored["account"]["id"] == account_id
        assert restored["positions"][0]["quantity"] == filled["positions"][0]["quantity"]
        history = (await client.get("/api/history")).json()
        assert [event["id"] for event in history["events"]] == list(
            dict.fromkeys(event["id"] for event in history["events"])
        )
    restored_app.state.paper_dashboard.close()


@pytest.mark.anyio
async def test_asgi_snapshot_restores_mixed_long_short_positions_and_signed_funding(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5, "ETHUSDT": -0.25}
    app = make_app(database, clock, FakeFeed([]), FakePolicy([target]))
    paper = app.state.paper_dashboard
    paper.advance_once_sync(observation(0, decision=True))
    paper.advance_once_sync(observation(1))
    funding = replace(
        observation(2),
        funding_rates={"BTCUSDT": 0.001, "ETHUSDT": 0.001},
        funding_mark_prices={"BTCUSDT": 100.0, "ETHUSDT": 100.0},
    )
    paper.advance_once_sync(funding)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        live = (await client.get("/api/live")).json()
        sides = {position["ticker"]: position["side"] for position in live["positions"]}
        assert sides == {"BTCUSDT": "long", "ETHUSDT": "short"}
        assert live["account"]["funding"] < 0.0
        assert live["risk"]["gross_exposure"] > abs(live["risk"]["net_exposure"])
    paper.close()

    restored_app = make_app(database, clock, FakeFeed([]), FakePolicy([]))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=restored_app),
        base_url="http://test",
    ) as client:
        restored = (await client.get("/api/live")).json()
        assert {position["ticker"]: position["side"] for position in restored["positions"]} == sides
        assert restored["account"]["funding"] == live["account"]["funding"]
    restored_app.state.paper_dashboard.close()


def test_drawdown_breach_durably_flattens_and_blocks_later_policy_decisions(tmp_path) -> None:
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
    app = make_app(
        tmp_path / "paper.sqlite3",
        FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        FakeFeed([]),
        FakePolicy([target]),
    )
    paper = app.state.paper_dashboard
    paper.advance_once_sync(observation(0, decision=True))
    paper.advance_once_sync(observation(1))
    crashed_prices = {**dict.fromkeys(TICKERS, 100.0), "BTCUSDT": 50.0}
    paper.advance_once_sync(
        replace(
            observation(2),
            mark_prices=crashed_prices,
            bid=crashed_prices,
            ask=crashed_prices,
        )
    )

    stopped = paper.live_snapshot()
    assert stopped["account"]["state"] == "Risk Stopped"
    assert stopped["pending_fill"]["kind"] == "risk_stop"
    paper.advance_once_sync(
        replace(
            observation(3),
            mark_prices=crashed_prices,
            bid=crashed_prices,
            ask=crashed_prices,
        )
    )
    paper.advance_once_sync(
        replace(
            observation(15, decision=True),
            mark_prices=crashed_prices,
            bid=crashed_prices,
            ask=crashed_prices,
        )
    )

    terminal = paper.live_snapshot()
    assert terminal["account"]["state"] == "Risk Stopped"
    assert terminal["positions"] == []
    assert terminal["pending_fill"] is None
    assert terminal["activity"]["decisions"] == 1
    assert [event.event_type for event in paper.store.events()].count("RiskStop") == 1
    paper.close()


@pytest.mark.anyio
async def test_controls_are_protected_idempotent_versioned_and_durable(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    app = make_app(database, clock, FakeFeed([]), FakePolicy([]))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        live = (await client.get("/api/live")).json()
        version = live["account"]["version"]
        rejected = await client.post(
            "/api/controls/pause",
            json={"expected_version": version},
            headers={"X-Idempotency-Key": "pause-1"},
        )
        assert rejected.status_code == 403
        assert (await client.get("/api/live")).json()["account"]["version"] == version

        headers = await control_headers(client, "pause-1")
        paused = await client.post(
            "/api/controls/pause",
            json={"expected_version": version},
            headers=headers,
        )
        duplicate = await client.post(
            "/api/controls/pause",
            json={"expected_version": version},
            headers=headers,
        )
        assert paused.status_code == duplicate.status_code == 200
        assert duplicate.json()["account"]["version"] == paused.json()["account"]["version"]
        assert paused.json()["activity"]["interventions"] == 1

        stale = await client.post(
            "/api/controls/resume",
            json={"expected_version": version},
            headers=await control_headers(client, "resume-stale"),
        )
        assert stale.status_code == 409
    app.state.paper_dashboard.close()

    clock.current += timedelta(hours=1)
    restored_app = make_app(database, clock, FakeFeed([]), FakePolicy([]))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=restored_app), base_url="http://test") as client:
        restored = (await client.get("/api/live")).json()
        assert restored["account"]["state"] == "Paused"
        assert restored["activity"]["interventions"] == 1
    restored_app.state.paper_dashboard.close()


@pytest.mark.anyio
async def test_slow_synchronous_feed_does_not_block_the_asgi_event_loop(tmp_path) -> None:
    class SlowUnavailableFeed:
        def observe(self, after: datetime | None) -> MarketObservation:
            del after
            time.sleep(0.2)
            raise RuntimeError("slow provider outage")

    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        market_feed=SlowUnavailableFeed(),
        policy_backend=FakePolicy([]),
        notifications=NotificationRecorder(),
    )
    paper = app.state.paper_dashboard
    started = time.monotonic()
    operator = asyncio.create_task(_operator_loop(paper, 60.0))

    await asyncio.sleep(0.02)
    event_loop_delay = time.monotonic() - started
    operator.cancel()
    await asyncio.gather(operator, return_exceptions=True)
    await asyncio.sleep(0.25)
    paper.close()

    assert event_loop_delay < 0.1


def test_restarted_operator_waits_for_the_next_minute_publication_boundary(tmp_path) -> None:
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, 30, tzinfo=UTC))
    app = make_app(tmp_path / "paper.sqlite3", clock, FakeFeed([]), FakePolicy([]))
    paper = app.state.paper_dashboard
    paper.advance_once_sync(observation(0))

    assert _operator_start_delay(paper, 60.0) == 32.0
    clock.current += timedelta(minutes=2)
    assert _operator_start_delay(paper, 60.0) == 0.0
    paper.close()


@pytest.mark.anyio
async def test_financial_reads_wait_for_a_slow_transition_without_blocking_asgi(tmp_path) -> None:
    class SlowPolicy:
        def decide(
            self,
            observed: MarketObservation,
            current_weights: dict[str, float],
        ) -> PolicyDecision:
            del current_weights
            time.sleep(0.2)
            target = dict.fromkeys(TICKERS, 0.0)
            return PolicyDecision(target, target, "model-1", observed.input_id, "protocol-1")

    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=SlowPolicy(),
        notifications=NotificationRecorder(),
    )
    paper = app.state.paper_dashboard
    advance = asyncio.create_task(asyncio.to_thread(paper.advance_once_sync, observation(0, decision=True)))
    await asyncio.sleep(0.02)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        started = time.monotonic()
        blocked_live = asyncio.create_task(client.get("/api/live"))
        blocked_system = asyncio.create_task(client.get("/api/system"))
        await asyncio.sleep(0.01)
        assert not blocked_live.done()
        assert not blocked_system.done()
        csrf = await client.get("/api/csrf")
        elapsed = time.monotonic() - started
        await asyncio.gather(blocked_live, blocked_system)

    await advance
    paper.close()
    assert csrf.status_code == 200
    assert elapsed < 0.1


@pytest.mark.anyio
async def test_application_lifespan_prepares_policy_in_the_background(tmp_path) -> None:
    prepared = threading.Event()
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([]),
        notifications=NotificationRecorder(),
        policy_preparer=prepared.set,
    )

    async with app.router.lifespan_context(app):
        assert await asyncio.to_thread(prepared.wait, 0.5)


@pytest.mark.anyio
async def test_overdue_pending_change_becomes_one_missed_execution_without_a_fill(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
    first = make_app(database, clock, FakeFeed([observation(0, decision=True)]), FakePolicy([target]))
    first.state.paper_dashboard.advance_once_sync()
    first.state.paper_dashboard.close()

    clock.current += timedelta(minutes=4)
    recovered_observation = observation(4)
    second = make_app(database, clock, FakeFeed([recovered_observation]), FakePolicy([]))
    second.state.paper_dashboard.advance_once_sync()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=second), base_url="http://test") as client:
        live = (await client.get("/api/live")).json()
        assert live["pending_fill"] is None
        assert live["positions"] == []
        assert live["activity"]["missed_executions"] == 1
        history = (await client.get("/api/history")).json()
        assert [event["type"] for event in history["events"]].count("MissedExecution") == 1
    second.state.paper_dashboard.close()


def test_recovery_marks_do_not_invent_fills_but_current_quote_can_fill_in_window(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
    first = make_app(database, clock, FakeFeed([observation(0, decision=True)]), FakePolicy([target]))
    first.state.paper_dashboard.advance_once_sync()
    first.state.paper_dashboard.close()

    clock.current += timedelta(minutes=3)
    feed = RecoveringFeed(
        current=observation(3),
        reconstructed=[
            replace(observation(1), reconstructed=True, decision_bar_closed=False),
            replace(observation(2), reconstructed=True, decision_bar_closed=False),
        ],
    )
    second = create_application(
        database_path=database,
        tickers=TICKERS,
        clock=clock,
        market_feed=feed,
        policy_backend=FakePolicy([]),
        notifications=NotificationRecorder(),
    )
    live = second.state.paper_dashboard.advance_once_sync()

    assert live["activity"]["fills"] == 1
    assert live["activity"]["decisions"] == 1
    reconstructed = [
        event
        for event in second.state.paper_dashboard.store.events(second.state.paper_dashboard.state.account_id)
        if event.event_type == "AccountMarkReconstructed"
    ]
    assert [event.occurred_at for event in reconstructed] == [observation(1).timestamp, observation(2).timestamp]
    fill = next(
        event
        for event in second.state.paper_dashboard.store.events(second.state.paper_dashboard.state.account_id)
        if event.event_type == "InstrumentFilled"
    )
    assert fill.occurred_at == observation(3).timestamp
    second.state.paper_dashboard.close()


def test_data_stale_is_an_overlay_and_notifies_once_after_five_minutes(tmp_path) -> None:
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    notifications = NotificationRecorder()
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=clock,
        market_feed=UnavailableFeed(RuntimeError("Binance unavailable")),
        policy_backend=FakePolicy([]),
        notifications=notifications,
    )

    first = app.state.paper_dashboard.advance_once_sync()
    assert first["account"]["state"] == "Trading"
    assert first["freshness"]["status"] == "Data Stale"
    assert notifications.calls == []

    clock.current += timedelta(minutes=6)
    app.state.paper_dashboard.advance_once_sync()
    app.state.paper_dashboard.advance_once_sync()
    assert [kind for kind, _ in notifications.calls] == ["data_stale"]
    app.state.paper_dashboard.close()


@pytest.mark.anyio
async def test_flatten_and_reset_use_confirmed_serialized_controls(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
    app = make_app(
        database,
        clock,
        FakeFeed([observation(0, decision=True), observation(1), observation(2)]),
        FakePolicy([target]),
    )
    app.state.paper_dashboard.advance_once_sync()
    app.state.paper_dashboard.advance_once_sync()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        positioned = (await client.get("/api/live")).json()
        flatten = await client.post(
            "/api/controls/flatten",
            json={"expected_version": positioned["account"]["version"], "confirmation": "FLATTEN"},
            headers=await control_headers(client, "flatten-1"),
        )
        assert flatten.status_code == 200
        assert flatten.json()["pending_fill"]["kind"] == "flatten"
        app.state.paper_dashboard.advance_once_sync()
        flat = (await client.get("/api/live")).json()
        assert flat["account"]["state"] == "Paused"
        assert flat["positions"] == []

        old_account_id = flat["account"]["id"]
        reset_headers = await control_headers(client, "reset-1")
        reset = await client.post(
            "/api/controls/reset",
            json={"expected_version": flat["account"]["version"], "confirmation": "RESET"},
            headers=reset_headers,
        )
        duplicate = await client.post(
            "/api/controls/reset",
            json={"expected_version": flat["account"]["version"], "confirmation": "RESET"},
            headers=reset_headers,
        )
        assert reset.status_code == duplicate.status_code == 200
        assert reset.json()["account"]["id"] != old_account_id
        assert duplicate.json()["account"]["id"] == reset.json()["account"]["id"]
        history = (await client.get("/api/history")).json()
        assert len(history["accounts"]) == 2
        assert sum(account["active"] for account in history["accounts"]) == 1
    app.state.paper_dashboard.close()


def test_failed_transition_reloads_the_durable_snapshot_without_claiming_data_stale(
    tmp_path,
    monkeypatch,
) -> None:
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    app = make_app(tmp_path / "paper.sqlite3", clock, FakeFeed([]), FakePolicy([]))
    paper = app.state.paper_dashboard
    initial_version = paper.state.state_version

    def fail_commit(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise sqlite3.OperationalError("injected snapshot failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(paper.store, "commit", fail_commit)
        with pytest.raises(sqlite3.OperationalError, match="injected snapshot failure"):
            paper.advance_once_sync(observation(1))

    assert paper.state.state_version == initial_version
    assert paper.state.last_observation_at is None
    assert paper.live_snapshot()["freshness"]["status"] == "Fresh"
    assert [event.event_type for event in paper.store.events(paper.state.account_id)] == [
        "FlatStart",
        "OperatingWindowOpened",
    ]
    paper.close()


def test_failed_fill_commit_cannot_publish_a_ghost_execution_notification(
    tmp_path,
    monkeypatch,
) -> None:
    notifications = NotificationRecorder()
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([target]),
        notifications=notifications,
    )
    paper = app.state.paper_dashboard
    paper.advance_once_sync(observation(0, decision=True))

    def fail_commit(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise sqlite3.OperationalError("injected fill commit failure")

    with monkeypatch.context() as patcher:
        patcher.setattr(paper.store, "commit", fail_commit)
        with pytest.raises(sqlite3.OperationalError, match="fill commit failure"):
            paper.advance_once_sync(observation(1))

    assert paper.state.pending_execution is not None
    assert paper.store.notification_health()["pending"] == 0
    assert notifications.calls == []
    assert "PortfolioChangeExecuted" not in {event.event_type for event in paper.store.events()}
    paper.close()


def test_reset_retry_after_commit_before_response_creates_exactly_one_replacement(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "paper.sqlite3"
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    app = make_app(database, clock, FakeFeed([]), FakePolicy([]))
    paper = app.state.paper_dashboard
    old_account_id = paper.state.account_id
    expected_version = paper.state.state_version
    original_reset = paper.store.reset_and_replace

    def commit_then_crash(*args: Any, **kwargs: Any) -> None:
        original_reset(*args, **kwargs)
        raise RuntimeError("process exited after SQLite commit")

    with monkeypatch.context() as patcher:
        patcher.setattr(paper.store, "reset_and_replace", commit_then_crash)
        with pytest.raises(RuntimeError, match="after SQLite commit"):
            paper.control(
                "reset",
                expected_version=expected_version,
                idempotency_key="crash-reset-1",
                confirmation="RESET",
            )

    replacement_id = paper.state.account_id
    assert replacement_id != old_account_id
    paper.store.close()
    paper._closed = True

    clock.current += timedelta(minutes=10)
    reopened = make_app(database, clock, FakeFeed([]), FakePolicy([]))
    restored = reopened.state.paper_dashboard
    repeated = restored.control(
        "reset",
        expected_version=expected_version,
        idempotency_key="crash-reset-1",
        confirmation="RESET",
    )

    assert repeated["account"]["id"] == replacement_id
    assert len(restored.store.accounts()) == 2
    assert sum(bool(account["active"]) for account in restored.store.accounts()) == 1
    assert [event.event_type for event in restored.store.events()].count("ManualResetCompleted") == 1
    restored.close()


def test_crashed_operating_window_ends_at_last_durable_observation(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    app = make_app(database, clock, FakeFeed([]), FakePolicy([]))
    paper = app.state.paper_dashboard
    paper.advance_once_sync(observation(1))
    paper.store.close()
    paper._closed = True

    clock.current += timedelta(minutes=20)
    reopened = make_app(database, clock, FakeFeed([]), FakePolicy([]))
    restored = reopened.state.paper_dashboard
    windows = restored.store.operating_windows(restored.state.account_id)

    assert len(windows) == 2
    assert windows[0]["ended_at"] == observation(1).timestamp.isoformat()
    assert windows[0]["close_reason"] == "detected-restart"
    assert windows[1]["started_at"] == clock.current.isoformat()
    restored.close()


def test_restart_completes_a_positioned_reset_that_crashed_after_flatten_commit(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "paper.sqlite3"
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
    app = make_app(
        database,
        clock,
        FakeFeed([observation(0, decision=True), observation(1), observation(2)]),
        FakePolicy([target]),
    )
    paper = app.state.paper_dashboard
    paper.advance_once_sync()
    paper.advance_once_sync()
    old_account_id = paper.state.account_id
    paper.control(
        "reset",
        expected_version=paper.state.state_version,
        idempotency_key="positioned-reset-1",
        confirmation="RESET",
    )

    def crash_before_archive(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise RuntimeError("crash before account archival")

    with monkeypatch.context() as patcher:
        patcher.setattr(paper, "_complete_reset", crash_before_archive)
        with pytest.raises(RuntimeError, match="before account archival"):
            paper.advance_once_sync()

    assert paper.state.lifecycle.value == "Reset Pending"
    assert paper.state.pending_execution is None
    assert all(abs(quantity) <= 1e-12 for quantity in paper.state.simulation.quantities.values())
    paper.store.close()
    paper._closed = True

    clock.current += timedelta(minutes=10)
    reopened = make_app(database, clock, FakeFeed([]), FakePolicy([]))
    restored = reopened.state.paper_dashboard
    accounts = restored.store.accounts()

    assert restored.state.account_id != old_account_id
    assert restored.state.lifecycle.value == "Trading"
    assert len(accounts) == 2
    assert sum(bool(account["active"]) for account in accounts) == 1
    assert [event.event_type for event in restored.store.events()].count("ManualResetCompleted") == 1
    assert restored.store.operating_windows(old_account_id)[0]["ended_at"] == observation(2).timestamp.isoformat()
    restored.close()


@pytest.mark.anyio
async def test_no_trade_decision_completes_and_attribution_runs_after_durable_record(tmp_path) -> None:
    attribution = FakeAttribution()
    policy = FakePolicy([dict.fromkeys(TICKERS, 0.0)])
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=policy,
        notifications=NotificationRecorder(),
        attribution_backend=attribution,
    )
    paper = app.state.paper_dashboard

    await paper.advance_once(observation(0, decision=True))
    tasks = tuple(paper._attribution_tasks)
    assert tasks
    await asyncio.gather(*tasks)
    await paper.advance_once(observation(1))

    decision = next(
        event for event in paper.store.events(paper.state.account_id) if event.event_type == "DecisionRecord"
    )
    detail = paper.event_snapshot(decision.event_id)
    assert attribution.calls == ["input-0"]
    assert detail["decision"]["outcome"] == "unchanged"
    assert detail["decision"]["attribution"]["status"] == "complete"
    assert detail["execution"] is None
    assert detail["attribution"]["status"] == "complete"
    assert detail["attribution"]["input_hash"] == "hash:input-0"
    assert [event.event_type for event in paper.store.events()].count("DecisionCompleted") == 1
    paper.close()


@pytest.mark.anyio
async def test_restart_requeues_a_durable_decision_with_pending_attribution(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    first_attribution = FakeAttribution()
    first = create_application(
        database_path=database,
        tickers=TICKERS,
        clock=clock,
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([dict.fromkeys(TICKERS, 0.0)]),
        notifications=NotificationRecorder(),
        attribution_backend=first_attribution,
    )
    first_paper = first.state.paper_dashboard
    first_paper.advance_once_sync(observation(0, decision=True))
    decision = next(event for event in first_paper.store.events() if event.event_type == "DecisionRecord")
    assert first_paper.event_snapshot(decision.event_id)["attribution"]["status"] == "pending"
    assert first_attribution.calls == []
    for task in tuple(first_paper._attribution_tasks):
        task.cancel()
    await asyncio.gather(*first_paper._attribution_tasks, return_exceptions=True)
    first_paper.store.close()
    first_paper._closed = True

    clock.current += timedelta(minutes=5)
    resumed_attribution = FakeAttribution()
    reopened = create_application(
        database_path=database,
        tickers=TICKERS,
        clock=clock,
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([]),
        notifications=NotificationRecorder(),
        attribution_backend=resumed_attribution,
    )
    restored = reopened.state.paper_dashboard
    restored.resume_pending_attributions()
    tasks = tuple(restored._attribution_tasks)
    assert tasks
    await asyncio.gather(*tasks)

    detail = restored.event_snapshot(decision.event_id)
    assert resumed_attribution.calls == ["input-0"]
    assert detail["attribution"]["status"] == "complete"
    restored.close()


def test_notification_failure_remains_visible_after_process_restart(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    app = create_application(
        database_path=database,
        tickers=TICKERS,
        clock=clock,
        market_feed=UnavailableFeed(RuntimeError("Binance unavailable")),
        policy_backend=FakePolicy([]),
        notifications=FailingNotifications(),
    )
    paper = app.state.paper_dashboard
    paper.advance_once_sync()
    clock.current += timedelta(minutes=6)
    paper.advance_once_sync()
    failed = paper.system_snapshot()["notifications"]
    assert failed["healthy"] is False
    assert failed["error"] == "desktop bus unavailable"
    assert paper.state.lifecycle.value == "Trading"
    paper.store.close()
    paper._closed = True

    recorder = NotificationRecorder()
    reopened = create_application(
        database_path=database,
        tickers=TICKERS,
        clock=clock,
        market_feed=UnavailableFeed(RuntimeError("still unavailable")),
        policy_backend=FakePolicy([]),
        notifications=recorder,
    )
    restored = reopened.state.paper_dashboard
    persisted = restored.system_snapshot()["notifications"]
    restored.advance_once_sync()

    assert persisted["healthy"] is False
    assert persisted["failed_kind"] == "data_stale"
    assert persisted["error"] == "desktop bus unavailable"
    assert recorder.calls == []
    restored.close()


@pytest.mark.anyio
async def test_weekly_fitting_runs_in_background_and_hands_off_without_reset(tmp_path) -> None:
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    fitter = FakeFitter(tmp_path / "model-2.pt")
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=clock,
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([]),
        notifications=NotificationRecorder(),
        policy_fitter=fitter,
    )
    paper = app.state.paper_dashboard
    account_id = paper.state.account_id
    await paper.advance_once(observation(1))

    await paper.maybe_start_policy_fitting()
    assert paper.state.fitting["status"] == "running"
    assert paper._fitting_task is not None
    await paper._fitting_task

    assert paper.state.account_id == account_id
    assert paper.state.simulation.quantities == dict.fromkeys(TICKERS, 0.0)
    assert paper.state.model_id == "model-2"
    assert paper.state.model_checkpoint == str(fitter.checkpoint)
    assert paper.state.fitting["status"] == "idle"
    assert [candidate.model_id for candidate in fitter.activated] == ["model-2"]
    assert {event.event_type for event in paper.store.events(account_id)} >= {
        "PolicyFittingStarted",
        "PolicyHandoff",
    }
    paper.close()


@pytest.mark.anyio
async def test_failed_policy_activation_never_advances_the_durable_model(tmp_path) -> None:
    class Fitter(FakeFitter):
        def activate(self, candidate: FittedPolicyCandidate):
            del candidate
            raise RuntimeError("candidate cannot be activated")

    fitter = Fitter(tmp_path / "invalid.pt")
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([]),
        notifications=NotificationRecorder(),
        policy_fitter=fitter,
    )
    paper = app.state.paper_dashboard
    await paper.advance_once(observation(1))
    old_model_id = paper.state.model_id
    await paper.maybe_start_policy_fitting()
    assert paper._fitting_task is not None
    await paper._fitting_task

    assert paper.state.model_id == old_model_id
    assert paper.state.fitting["status"] == "failed"
    assert "candidate cannot be activated" in paper.state.fitting["error"]
    assert "PolicyHandoff" not in {event.event_type for event in paper.store.events()}
    paper.close()


def test_production_schedule_runs_once_on_the_first_window_after_sunday_deadline() -> None:
    backend = object.__new__(ProductionPolicyBackend)
    backend.config = load_config("policy.toml")
    sunday = datetime(2026, 8, 23, 18, 0, tzinfo=UTC)

    assert backend.is_due(sunday, {"status": "idle"})
    assert not backend.is_due(
        sunday,
        {"status": "idle", "completed_at": datetime(2026, 8, 23, 0, 1, tzinfo=UTC).isoformat()},
    )
    assert backend.is_due(
        sunday + timedelta(days=7),
        {"status": "idle", "completed_at": datetime(2026, 8, 23, 0, 1, tzinfo=UTC).isoformat()},
    )


def test_production_market_feed_uses_the_low_latency_mark_path(monkeypatch) -> None:
    expected = observation(0)

    class Adapter:
        calls: list[datetime | None] = []

        def mark(self, after: datetime | None) -> object:
            self.calls.append(after)
            return object()

    adapter = Adapter()
    feed = ProductionMarketFeed(adapter)  # type: ignore[arg-type]
    monkeypatch.setattr(feed, "_observations", lambda observed: [expected])

    assert feed.observe(None) is expected
    assert adapter.calls == [None]


def test_compatible_policy_revisions_segment_history_metrics(tmp_path) -> None:
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    app = make_app(tmp_path / "paper.sqlite3", clock, FakeFeed([]), FakePolicy([]))
    paper = app.state.paper_dashboard
    compatibility = {"trading_universe": list(TICKERS), "account_currency": "USD"}
    paper.register_policy_revision(
        protocol_id="protocol-1",
        compatible=True,
        compatibility=compatibility,
    )
    paper.advance_once_sync(observation(1))
    clock.current += timedelta(minutes=2)
    paper.register_policy_revision(
        protocol_id="protocol-2",
        compatible=True,
        compatibility=compatibility,
    )
    paper.advance_once_sync(observation(2))

    history = paper.history_snapshot()

    assert history["comparison"]["protocol_id"] == "protocol-2"
    assert history["accounts"][0]["protocol_id"] == "protocol-2"
    assert [segment["protocol_id"] for segment in history["protocol_segments"]] == [
        "protocol-1",
        "protocol-2",
    ]
    assert history["protocol_segments"][0]["ended_at"] == clock.current.isoformat()
    assert all(segment["compounded_net_return"] == 0.0 for segment in history["protocol_segments"])
    paper.close()


def test_initial_fitted_policy_selection_does_not_count_as_a_completed_weekly_fit(tmp_path) -> None:
    app = make_app(
        tmp_path / "paper.sqlite3",
        FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        FakeFeed([]),
        FakePolicy([]),
    )
    paper = app.state.paper_dashboard

    paper.register_initial_fitted_policy(model_id="model-1", checkpoint="/models/model-1.pt")

    assert paper.state.model_id == "model-1"
    assert paper.state.model_checkpoint == "/models/model-1.pt"
    assert paper.state.fitting == {"status": "idle"}
    assert [event.event_type for event in paper.store.events()].count("FittedPolicySelected") == 1
    assert "PolicyHandoff" not in {event.event_type for event in paper.store.events()}
    paper.close()


def test_production_policy_preparation_warms_data_and_gpu_inference_without_a_decision(tmp_path) -> None:
    canonical = SimpleNamespace(identity_hash="canonical-market-state-1")

    class Adapter:
        calls = 0

        def load(self) -> SimpleNamespace:
            self.calls += 1
            return canonical

    class Backend:
        calls: list[dict[str, Any]] = []

        def paper(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            del args
            self.calls.append(kwargs)
            return SimpleNamespace(
                refitted=False,
                model_bytes=b"selected-policy",
                target_weights=dict.fromkeys(TICKERS, 0.0),
            )

    adapter = Adapter()
    inference = Backend()
    selected = b"selected-policy"
    backend = ProductionPolicyBackend(
        adapter=adapter,
        backend=inference,
        fitting_adapter=Adapter(),
        attribution_adapter=Adapter(),
        fitting_backend=Backend(),
        checkpoint_directory=tmp_path,
        config=load_config("policy.toml"),
        device="cuda",
        fitted_model=selected,
    )

    backend.prepare(observed_at=observation(0).timestamp)

    assert adapter.calls == 1
    assert inference.calls[0]["fitted_model"] == selected
    assert inference.calls[0]["observed_at"] == observation(0).timestamp
    assert inference.calls[0]["current_weights"] == dict.fromkeys(TICKERS, 0.0)
    assert backend._attribution_inputs == {}


def test_production_decision_identity_hashes_exact_market_state_and_current_portfolio(tmp_path) -> None:
    canonical = SimpleNamespace(identity_hash="canonical-market-state-1")

    class Adapter:
        def load(self) -> SimpleNamespace:
            return canonical

    class Backend:
        def paper(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            del args, kwargs
            return SimpleNamespace(
                refitted=False,
                model_bytes=b"selected-policy",
                target_weights=dict.fromkeys(TICKERS, 0.0),
            )

    config = load_config("policy.toml")
    selected = b"selected-policy"
    backend = ProductionPolicyBackend(
        adapter=Adapter(),
        backend=Backend(),
        fitting_adapter=Adapter(),
        attribution_adapter=Adapter(),
        fitting_backend=Backend(),
        checkpoint_directory=tmp_path,
        config=config,
        device="cpu",
        fitted_model=selected,
    )
    current = {ticker: index / 10 for index, ticker in enumerate(reversed(TICKERS))}
    decision = backend.decide(observation(0, decision=True), current)
    expected = hashlib.sha256(
        (
            canonical.identity_hash
            + ":"
            + observation(0).timestamp.isoformat()
            + ":"
            + json.dumps(current, sort_keys=True, separators=(",", ":"))
        ).encode()
    ).hexdigest()

    assert decision.input_id == expected
    changed = backend.decide(observation(0, decision=True), {**current, "BTCUSDT": 0.99})
    assert changed.input_id != decision.input_id
