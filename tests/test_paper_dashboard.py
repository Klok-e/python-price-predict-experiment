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
import pandas as pd
import pytest
import torch

from netgrowth.config import load_config
from netgrowth.market_data import CanonicalDataset, InstrumentData
from netgrowth.paper_dashboard.application import (
    FittedPolicyCandidate,
    _operator_loop,
    _operator_start_delay,
    create_application,
)
from netgrowth.paper_dashboard.domain import MarketObservation, PolicyDecision
from netgrowth.paper_dashboard.production import (
    ProductionMarketFeed,
    ProductionPolicyBackend,
    _causal_policy_input,
)

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


def canonical_policy_input(*, periods: int, corrected: bool = False) -> CanonicalDataset:
    index = pd.date_range("2026-08-23T17:57:00Z", periods=periods, freq="min")
    instruments: dict[str, InstrumentData] = {}
    for offset, ticker in enumerate(TICKERS):
        close = pd.Series([100.0 + offset + step / 10 for step in range(periods)], index=index)
        bars = pd.DataFrame(
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
        if corrected:
            bars.loc[index[1], "close"] += 1.0
        instruments[ticker] = InstrumentData(
            perpetual=bars,
            spot=bars.copy(),
            funding=pd.Series([0.0], index=index[:1], name="funding_rate"),
            open_interest=pd.Series([1_000.0], index=index[:1], name="open_interest"),
            premium=pd.Series([0.001], index=index[:1], name="premium"),
        )
    return CanonicalDataset(instruments=instruments, tickers=TICKERS)


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
async def test_filled_decision_detail_exposes_exact_targets_constraints_timing_and_costs(tmp_path) -> None:
    class ConstrainedPolicy:
        def decide(
            self,
            observed: MarketObservation,
            current_weights: dict[str, float],
        ) -> PolicyDecision:
            del current_weights
            raw = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.55}
            constrained = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
            return PolicyDecision(
                raw,
                constrained,
                "model-exact",
                observed.input_id,
                "protocol-exact",
                ("BTCUSDT concentration capped at 0.50",),
            )

    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 1, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=ConstrainedPolicy(),
        notifications=NotificationRecorder(),
    )
    paper = app.state.paper_dashboard
    paper.advance_once_sync(observation(0, decision=True))
    paper.advance_once_sync(observation(1))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        chart = (await client.get("/api/chart", params={"ticker": "BTCUSDT"})).json()
        signal = next(marker for marker in chart["markers"] if marker["type"] == "Signal")
        detail = (await client.get(f"/api/events/{signal['id']}")).json()

    decision = detail["decision"]
    fill = detail["execution"]["fills"][0]
    assert decision["signal_time"] == observation(0).timestamp.isoformat()
    assert decision["model_id"] == "model-exact"
    assert decision["input_id"] == "input-0"
    assert decision["protocol_id"] == "protocol-exact"
    assert decision["current_portfolio"] == dict.fromkeys(TICKERS, 0.0)
    assert decision["raw_target_weights"]["BTCUSDT"] == pytest.approx(0.55)
    assert decision["constrained_target_weights"]["BTCUSDT"] == pytest.approx(0.5)
    assert decision["projected_turnover"] == pytest.approx(0.5)
    assert decision["threshold_outcome"] == "executable"
    assert decision["eligible_at"] == observation(1).timestamp.isoformat()
    assert decision["expires_at"] == observation(3).timestamp.isoformat()
    assert decision["constraints"] == ["BTCUSDT concentration capped at 0.50"]
    assert decision["execution_time"] == observation(1).timestamp.isoformat()
    assert decision["outcome"] == "PortfolioChangeExecuted"
    assert detail["execution"]["transaction_cost"] > 0.0
    assert fill["reference_price"] == 100.0
    assert fill["effective_fill"] > fill["reference_price"]
    assert fill["timestamp"] == observation(1).timestamp.isoformat()
    paper.close()


@pytest.mark.anyio
async def test_selected_ticker_chart_retains_minute_weights_and_only_its_fill_markers(tmp_path) -> None:
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5, "ETHUSDT": -0.25}
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    app = make_app(
        tmp_path / "paper.sqlite3",
        clock,
        FakeFeed([]),
        FakePolicy([target]),
    )
    paper = app.state.paper_dashboard
    for offset in range(3):
        observed = observation(offset, decision=offset == 0)
        candles = {
            ticker: {
                "open": 99.0 + offset,
                "high": 101.0 + offset,
                "low": 98.0 + offset,
                "close": 100.0 + offset,
            }
            for ticker in TICKERS
        }
        if offset == 2:
            observed = replace(
                observed,
                funding_rates={"BTCUSDT": 0.001, "ETHUSDT": 0.001},
                funding_mark_prices={"BTCUSDT": 102.0, "ETHUSDT": 102.0},
            )
        paper.advance_once_sync(replace(observed, mark_prices=dict.fromkeys(TICKERS, 100.0 + offset), candles=candles))
    clock.current += timedelta(minutes=3)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        full = (await client.get("/api/chart", params={"ticker": "BTCUSDT", "range": "full"})).json()
        compressed = (
            await client.get(
                "/api/chart",
                params={"ticker": "BTCUSDT", "range": "full", "pixels": 1},
            )
        ).json()
        ranges = {
            range_name: (
                await client.get(
                    "/api/chart",
                    params={"ticker": "BTCUSDT", "range": range_name},
                )
            ).json()
            for range_name in ("current", "24h", "7d", "full")
        }
        sol = (await client.get("/api/chart", params={"ticker": "SOLUSDT", "range": "full"})).json()

    fill_markers = [marker for marker in full["markers"] if marker["type"] == "InstrumentFilled"]
    funding_marker = next(marker for marker in full["markers"] if marker["type"] == "FundingApplied")
    filled_weight = next(point for point in full["weights"] if point["time"] == observation(1).timestamp.isoformat())
    assert filled_weight["current"] > 0.0
    assert filled_weight["target"] == pytest.approx(0.5)
    assert fill_markers == [
        {
            "id": fill_markers[0]["id"],
            "time": observation(1).timestamp.isoformat(),
            "type": "InstrumentFilled",
            "label": "Instrument fill",
            "price": 100.0,
            "ticker": "BTCUSDT",
            "material": True,
        }
    ]
    assert funding_marker["price"] == 102.0
    assert all(marker["type"] != "FundingApplied" for marker in sol["markers"])
    assert {marker["id"] for marker in compressed["markers"]} == {marker["id"] for marker in full["markers"]}
    assert all(result["portfolio"] for result in ranges.values())
    assert {name: result["range"] for name, result in ranges.items()} == {
        "current": "current",
        "24h": "24h",
        "7d": "7d",
        "full": "full",
    }
    paper.close()


@pytest.mark.anyio
async def test_material_activity_stays_visible_while_minute_marks_remain_chartable(tmp_path) -> None:
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
    app = make_app(
        tmp_path / "paper.sqlite3",
        FakeClock(datetime(2026, 8, 23, 18, 30, tzinfo=UTC)),
        FakeFeed([]),
        FakePolicy([target]),
    )
    paper = app.state.paper_dashboard
    for offset in range(26):
        paper.advance_once_sync(observation(offset, decision=offset == 0))

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        live = (await client.get("/api/live")).json()
        history = (await client.get("/api/history")).json()
        chart = (await client.get("/api/chart", params={"ticker": "BTCUSDT"})).json()

    recent_types = {event["type"] for event in live["recent_events"]}
    history_types = {event["type"] for event in history["events"]}
    assert {"DecisionRecord", "InstrumentFilled"} <= recent_types
    assert {"DecisionRecord", "InstrumentFilled"} <= history_types
    assert not {"AccountMarked", "AccountMarkReconstructed"} & recent_types
    assert not {"AccountMarked", "AccountMarkReconstructed"} & history_types
    assert len(chart["portfolio"]) == 26
    paper.close()


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
        assert set(live["account"]) >= {
            "starting_equity",
            "current_equity",
            "net_pnl",
            "compounded_net_return",
            "gross_trading_pnl",
            "transaction_cost",
            "funding",
            "turnover",
            "marked_equity_reconciliation",
        }
        assert set(live["risk"]) == {
            "current_drawdown",
            "maximum_drawdown",
            "high_water_equity",
            "drawdown_limit",
            "gross_exposure",
            "net_exposure",
            "cash_weight",
            "concentrations",
        }
        assert all(
            set(position)
            >= {
                "ticker",
                "side",
                "quantity",
                "mark",
                "notional",
                "current_weight",
                "target_weight",
                "average_entry",
                "realized_pnl",
                "unrealized_pnl",
            }
            for position in live["positions"]
        )
        assert set(live["activity"]) == {
            "decisions",
            "executable_changes",
            "fills",
            "below_threshold",
            "unchanged_targets",
            "missed_executions",
            "interventions",
        }
        assert set(live) >= {
            "freshness",
            "operating_window",
            "next_decision_at",
            "pending_fill",
            "fitting",
            "controls",
            "recent_events",
        }
        system = (await client.get("/api/system")).json()
        assert set(system) >= {
            "as_of",
            "market_feed",
            "policy",
            "fitting",
            "last_policy_handoff",
            "compute",
            "operating_windows",
            "notifications",
            "database",
            "backup",
            "service",
        }
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
async def test_activity_keeps_operator_fills_out_of_executable_policy_changes(tmp_path) -> None:
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

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        positioned = (await client.get("/api/live")).json()
        response = await client.post(
            "/api/controls/flatten",
            json={"expected_version": positioned["account"]["version"], "confirmation": "FLATTEN"},
            headers=await control_headers(client, "activity-flatten"),
        )
        assert response.status_code == 200
        paper.advance_once_sync(observation(2))
        live = (await client.get("/api/live")).json()
        history = (await client.get("/api/history")).json()

    assert live["activity"] == {
        "decisions": 1,
        "executable_changes": 1,
        "fills": 2,
        "below_threshold": 0,
        "unchanged_targets": 0,
        "missed_executions": 0,
        "interventions": 1,
    }
    assert history["protocol_segments"][0]["executable_changes"] == 1
    paper.close()


@pytest.mark.anyio
async def test_history_exposes_durable_event_facts_with_kyiv_and_utc_times(tmp_path) -> None:
    clock = FakeClock(datetime(2026, 1, 15, 12, 0, tzinfo=UTC))
    app = make_app(tmp_path / "paper.sqlite3", clock, FakeFeed([]), FakePolicy([]))
    clock.current = datetime(2026, 8, 23, 18, 0, tzinfo=UTC)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        live = (await client.get("/api/live")).json()
        paused = await client.post(
            "/api/controls/pause",
            json={"expected_version": live["account"]["version"]},
            headers=await control_headers(client, "timezone-pause"),
        )
        assert paused.status_code == 200
        history = (await client.get("/api/history")).json()
        flat_start = next(event for event in history["events"] if event["type"] == "FlatStart")
        intervention = next(event for event in history["events"] if event["type"] == "OperatorIntervention")
        detail = (await client.get(f"/api/events/{intervention['id']}")).json()

    assert flat_start["display_time"] == {
        "kyiv": "2026-01-15T14:00:00+02:00",
        "utc": "2026-01-15T12:00:00+00:00",
    }
    assert intervention["display_time"] == {
        "kyiv": "2026-08-23T21:00:00+03:00",
        "utc": "2026-08-23T18:00:00+00:00",
    }
    assert detail["details"] == {"action": "pause", "cancelled_decision_id": None}
    app.state.paper_dashboard.close()


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
        chart = (await client.get("/api/chart", params={"ticker": "BTCUSDT"})).json()
        signal_marker = next(marker for marker in chart["markers"] if marker["type"] == "Signal")
        missed_marker = next(marker for marker in chart["markers"] if marker["type"] == "MissedExecution")
        detail = (await client.get(f"/api/events/{missed_marker['id']}")).json()
        assert signal_marker["time"] == observation(0).timestamp.isoformat()
        assert missed_marker["time"] == observation(4).timestamp.isoformat()
        assert detail["decision"]["signal_time"] == observation(0).timestamp.isoformat()
        assert detail["decision"]["outcome"] == "MissedExecution"
        assert detail["execution"]["outcome"] == "MissedExecution"
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


def test_operator_checks_daily_backup_and_surfaces_a_later_failed_attempt(tmp_path, monkeypatch) -> None:
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    backup_directory = tmp_path / "backups"
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        backup_directory=backup_directory,
        tickers=TICKERS,
        clock=clock,
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([]),
        notifications=NotificationRecorder(),
    )
    paper = app.state.paper_dashboard
    assert len(list(backup_directory.glob("paper-*.sqlite3"))) == 1

    clock.current += timedelta(days=1)
    paper.advance_once_sync(observation(24 * 60))
    assert len(list(backup_directory.glob("paper-*.sqlite3"))) == 2

    def fail_backup(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise OSError("backup disk is full")

    monkeypatch.setattr(paper.store, "_backup_file", fail_backup)
    clock.current += timedelta(days=1)
    paper.advance_once_sync(observation(2 * 24 * 60))

    backup = paper.system_snapshot()["backup"]
    assert backup["error"] == "backup disk is full"
    assert backup["created_at"] == clock.current.isoformat()
    paper.close()


def test_system_snapshot_exposes_backup_age(tmp_path) -> None:
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        backup_directory=tmp_path / "backups",
        tickers=TICKERS,
        clock=clock,
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([]),
        notifications=NotificationRecorder(),
    )
    paper = app.state.paper_dashboard

    clock.current += timedelta(hours=6)

    assert paper.system_snapshot()["backup"]["age_seconds"] == pytest.approx(6 * 60 * 60)
    paper.close()


def test_live_snapshot_documents_the_marked_equity_reconciliation(tmp_path) -> None:
    app = make_app(
        tmp_path / "paper.sqlite3",
        FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        FakeFeed([]),
        FakePolicy([]),
    )
    paper = app.state.paper_dashboard
    paper.advance_once_sync(observation(1))

    account = paper.live_snapshot()["account"]
    reconciliation = account["marked_equity_reconciliation"]
    assert reconciliation["formula"] == "cash_balance + unrealized_pnl + reconciliation_difference"
    assert reconciliation["composed_equity"] + reconciliation["difference"] == account["current_equity"]
    assert reconciliation["authoritative_marked_equity"] == account["current_equity"]
    paper.close()


def test_system_snapshot_exposes_policy_compute_diagnostics(tmp_path) -> None:
    class DiagnosticPolicy(FakePolicy):
        def diagnostics(self) -> dict[str, Any]:
            return {
                "backend": "ROCm",
                "requested_device": "cuda",
                "actual_device": "cuda:0",
                "device_name": "AMD Radeon RX 7800 XT",
                "rocm_version": "6.4",
                "inference": "ready",
                "fitting": "ready",
            }

    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=DiagnosticPolicy([]),
        notifications=NotificationRecorder(),
    )

    compute = app.state.paper_dashboard.system_snapshot()["compute"]
    assert compute["backend"] == "ROCm"
    assert compute["actual_device"] == "cuda:0"
    assert compute["inference"] == compute["fitting"] == "ready"
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
        active_before_archive_reads = (await client.get("/api/live")).json()["account"]
        archived_history = (await client.get("/api/history", params={"account_id": old_account_id})).json()
        archived_chart = (
            await client.get(
                "/api/chart",
                params={"account_id": old_account_id, "ticker": "BTCUSDT", "range": "full"},
            )
        ).json()
        archive_event = next(event for event in archived_history["events"] if event["type"] == "AccountArchived")
        archive_detail = (await client.get(f"/api/events/{archive_event['id']}")).json()
        active_after_archive_reads = (await client.get("/api/live")).json()["account"]
        assert archived_history["selected_account_id"] == old_account_id
        assert archived_history["comparison"]["account_id"] == old_account_id
        assert len(archived_history["comparison"]["accounts"]) == 2
        assert archived_chart["portfolio"]
        assert archive_detail["details"]["new_account_id"] == reset.json()["account"]["id"]
        assert active_after_archive_reads == active_before_archive_reads
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
    assert tuple(paper._attribution_tasks) == ()
    assert attribution.calls == []
    await paper.advance_once(observation(1))
    paper.resume_pending_attributions()
    tasks = tuple(paper._attribution_tasks)
    assert tasks
    await asyncio.gather(*tasks)

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
async def test_below_threshold_decision_remains_auditable_without_a_fill(tmp_path) -> None:
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.005}
    app = make_app(
        tmp_path / "paper.sqlite3",
        FakeClock(datetime(2026, 8, 23, 18, 1, tzinfo=UTC)),
        FakeFeed([]),
        FakePolicy([target]),
    )
    paper = app.state.paper_dashboard
    await paper.advance_once(observation(0, decision=True))
    await paper.advance_once(observation(1))
    decision = next(event for event in paper.store.events() if event.event_type == "DecisionRecord")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        live = (await client.get("/api/live")).json()
        detail = (await client.get(f"/api/events/{decision.event_id}")).json()

    assert live["activity"]["decisions"] == 1
    assert live["activity"]["below_threshold"] == 1
    assert live["activity"]["fills"] == 0
    assert detail["decision"]["threshold_outcome"] == "below_threshold"
    assert detail["decision"]["outcome"] == "below_threshold"
    assert detail["execution"] is None
    paper.close()


@pytest.mark.anyio
async def test_attribution_waits_for_the_scheduled_fill(tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingAttribution:
        def attribute(self, input_id: str, decision: dict[str, Any]) -> dict[str, Any]:
            started.set()
            assert release.wait(timeout=1.0)
            return FakeAttribution().attribute(input_id, decision)

    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 1, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([target]),
        notifications=NotificationRecorder(),
        attribution_backend=BlockingAttribution(),
    )
    paper = app.state.paper_dashboard
    await paper.advance_once(observation(0, decision=True))
    assert not started.is_set()
    assert tuple(paper._attribution_tasks) == ()
    decision = next(event for event in paper.store.events() if event.event_type == "DecisionRecord")
    assert paper.event_snapshot(decision.event_id)["attribution"]["status"] == "pending"

    filled = await paper.advance_once(observation(1))
    assert filled["activity"]["fills"] == 1
    assert filled["pending_fill"] is None
    paper.resume_pending_attributions()
    assert await asyncio.to_thread(started.wait, 1.0)
    release.set()
    await asyncio.gather(*paper._attribution_tasks)
    assert paper.event_snapshot(decision.event_id)["attribution"]["status"] == "complete"
    paper.close()


@pytest.mark.anyio
async def test_failed_attribution_is_visible_without_changing_the_exact_decision_record(tmp_path) -> None:
    class FailingAttribution:
        def attribute(self, input_id: str, decision: dict[str, Any]) -> dict[str, Any]:
            del input_id, decision
            raise RuntimeError("explanation worker ran out of memory")

    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([dict.fromkeys(TICKERS, 0.0)]),
        notifications=NotificationRecorder(),
        attribution_backend=FailingAttribution(),
    )
    paper = app.state.paper_dashboard
    await paper.advance_once(observation(0, decision=True))
    await paper.advance_once(observation(1))
    paper.resume_pending_attributions()
    await asyncio.gather(*paper._attribution_tasks)
    decision = next(event for event in paper.store.events() if event.event_type == "DecisionRecord")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        detail = (await client.get(f"/api/events/{decision.event_id}")).json()

    assert detail["decision"]["input_id"] == "input-0"
    assert detail["decision"]["attribution"] == {
        "status": "failed",
        "event_id": detail["attribution"]["event_id"],
    }
    assert detail["attribution"]["status"] == "failed"
    assert detail["attribution"]["label"] == "Approximate post-hoc influence evidence"
    assert detail["attribution"]["error"] == "explanation worker ran out of memory"
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
    await restored.advance_once(observation(5))
    restored.resume_pending_attributions()
    tasks = tuple(restored._attribution_tasks)
    assert tasks
    await asyncio.gather(*tasks)

    detail = restored.event_snapshot(decision.event_id)
    assert resumed_attribution.calls == ["input-0"]
    assert detail["attribution"]["status"] == "complete"
    restored.close()


@pytest.mark.anyio
async def test_pending_attribution_waits_for_active_policy_fitting(tmp_path) -> None:
    fitting_started = threading.Event()
    fitting_release = threading.Event()
    attribution_started = threading.Event()

    class BlockingFitter(FakeFitter):
        def fit(self, observed_at: datetime, current_weights: dict[str, float]) -> FittedPolicyCandidate:
            fitting_started.set()
            assert fitting_release.wait(timeout=1.0)
            return super().fit(observed_at, current_weights)

    class RecordingAttribution(FakeAttribution):
        def attribute(self, input_id: str, decision: dict[str, Any]) -> dict[str, Any]:
            attribution_started.set()
            return super().attribute(input_id, decision)

    attribution = RecordingAttribution()
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([dict.fromkeys(TICKERS, 0.0)]),
        notifications=NotificationRecorder(),
        policy_fitter=BlockingFitter(tmp_path / "model-2.pt"),
        attribution_backend=attribution,
    )
    paper = app.state.paper_dashboard
    paper.advance_once_sync(observation(0, decision=True))
    paper.advance_once_sync(observation(1))

    await paper.maybe_start_policy_fitting()
    assert await asyncio.to_thread(fitting_started.wait, 1.0)
    paper.resume_pending_attributions()
    await asyncio.sleep(0)
    assert tuple(paper._attribution_tasks) == ()
    assert not attribution_started.is_set()

    fitting_release.set()
    assert paper._fitting_task is not None
    await paper._fitting_task
    paper.resume_pending_attributions()
    await asyncio.gather(*paper._attribution_tasks)

    assert attribution.calls == ["input-0"]
    paper.close()


@pytest.mark.anyio
async def test_due_policy_fitting_never_overlaps_inflight_attribution(tmp_path) -> None:
    attribution_started = threading.Event()
    attribution_release = threading.Event()
    fitting_started = threading.Event()
    fitting_release = threading.Event()

    class BlockingAttribution(FakeAttribution):
        def attribute(self, input_id: str, decision: dict[str, Any]) -> dict[str, Any]:
            attribution_started.set()
            assert attribution_release.wait(timeout=1.0)
            return super().attribute(input_id, decision)

    class BlockingFitter(FakeFitter):
        def fit(self, observed_at: datetime, current_weights: dict[str, float]) -> FittedPolicyCandidate:
            fitting_started.set()
            assert fitting_release.wait(timeout=1.0)
            return super().fit(observed_at, current_weights)

    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([dict.fromkeys(TICKERS, 0.0)]),
        notifications=NotificationRecorder(),
        policy_fitter=BlockingFitter(tmp_path / "model-2.pt"),
        attribution_backend=BlockingAttribution(),
    )
    paper = app.state.paper_dashboard
    paper.advance_once_sync(observation(0, decision=True))
    paper.advance_once_sync(observation(1))
    paper.resume_pending_attributions()
    assert await asyncio.to_thread(attribution_started.wait, 1.0)

    await paper.maybe_start_policy_fitting()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        system = (await client.get("/api/system")).json()
        decision = next(event for event in paper.store.events() if event.event_type == "DecisionRecord")
        detail = (await client.get(f"/api/events/{decision.event_id}")).json()
    assert system["fitting"]["status"] == "idle"
    assert detail["attribution"]["status"] == "pending"
    assert not fitting_started.is_set()

    attribution_release.set()
    await asyncio.gather(*paper._attribution_tasks)
    await paper.maybe_start_policy_fitting()
    assert await asyncio.to_thread(fitting_started.wait, 1.0)
    fitting_release.set()
    assert paper._fitting_task is not None
    await paper._fitting_task
    paper.close()


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
    assert paper.state.fitting["due_at"] == clock.current.isoformat()
    assert paper.state.fitting["started_at"] == clock.current.isoformat()
    assert paper.state.fitting["completed_at"] == clock.current.isoformat()
    assert [candidate.model_id for candidate in fitter.activated] == ["model-2"]
    assert {event.event_type for event in paper.store.events(account_id)} >= {
        "PolicyFittingStarted",
        "PolicyHandoff",
    }
    system = paper.system_snapshot()
    assert system["policy"]["model_checkpoint"] == str(fitter.checkpoint)
    assert system["last_policy_handoff"] == {
        "old_model_id": "unknown",
        "new_model_id": "model-2",
        "checkpoint": str(fitter.checkpoint),
        "protocol_id": "unknown",
        "due_at": clock.current.isoformat(),
        "started_at": clock.current.isoformat(),
        "completed_at": clock.current.isoformat(),
    }
    paper.close()


@pytest.mark.anyio
async def test_paused_account_starts_due_policy_fitting_on_the_next_operating_window(tmp_path) -> None:
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
    await paper.advance_once(observation(1))
    paper.control(
        "pause",
        expected_version=paper.state.state_version,
        idempotency_key="pause-before-fit",
    )

    await paper.maybe_start_policy_fitting()
    assert paper._fitting_task is not None
    await paper._fitting_task

    assert paper.state.lifecycle.value == "Paused"
    assert paper.state.model_id == "model-2"
    assert paper.state.fitting["status"] == "idle"
    paper.close()


@pytest.mark.anyio
async def test_policy_activation_and_durable_handoff_are_one_serialized_boundary(tmp_path) -> None:
    activated = threading.Event()
    release_activation = threading.Event()
    competing_advance_completed = threading.Event()
    completed_before_handoff: list[bool] = []
    advance_errors: list[BaseException] = []

    @dataclass
    class SwitchingPolicy:
        model_id: str = "model-1"

        def decide(self, observed: MarketObservation, current_weights: dict[str, float]) -> PolicyDecision:
            del current_weights
            target = dict.fromkeys(TICKERS, 0.0)
            return PolicyDecision(target, target, self.model_id, observed.input_id, "protocol-1")

    @dataclass
    class BlockingActivationFitter(FakeFitter):
        policy: SwitchingPolicy = field(default_factory=SwitchingPolicy)

        def activate(self, candidate: FittedPolicyCandidate):
            previous = self.policy.model_id
            self.policy.model_id = candidate.model_id
            activated.set()
            assert release_activation.wait(timeout=1.0)

            def rollback() -> None:
                self.policy.model_id = previous

            return rollback

    policy = SwitchingPolicy()
    fitter = BlockingActivationFitter(tmp_path / "model-2.pt", policy=policy)
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=policy,
        notifications=NotificationRecorder(),
        policy_fitter=fitter,
    )
    paper = app.state.paper_dashboard
    paper.register_policy_revision(
        protocol_id="protocol-1",
        compatibility=load_config("policy.toml").compatibility_manifest,
    )
    paper.register_initial_fitted_policy(model_id="model-1", checkpoint="/models/model-1.pt")
    await paper.advance_once(observation(1))

    def competing_advance() -> None:
        assert activated.wait(timeout=1.0)
        try:
            paper.advance_once_sync(observation(15, decision=True))
        except BaseException as error:
            advance_errors.append(error)
        finally:
            competing_advance_completed.set()

    def release_after_probe() -> None:
        assert activated.wait(timeout=1.0)
        time.sleep(0.1)
        completed_before_handoff.append(competing_advance_completed.is_set())
        release_activation.set()

    advance_thread = threading.Thread(target=competing_advance)
    probe = threading.Thread(target=release_after_probe)
    advance_thread.start()
    probe.start()
    await paper.maybe_start_policy_fitting()
    assert paper._fitting_task is not None
    await paper._fitting_task
    advance_thread.join(timeout=1.0)
    probe.join(timeout=1.0)

    events = paper.store.events()
    paper.close()
    assert completed_before_handoff == [False]
    assert advance_errors == []
    assert paper.state.model_id == "model-2"
    handoff, decision = [event for event in events if event.event_type in {"PolicyHandoff", "DecisionRecord"}]
    assert handoff.sequence < decision.sequence
    assert decision.payload["model_id"] == "model-2"


@pytest.mark.anyio
async def test_observations_decisions_and_controls_continue_while_fitting_runs(tmp_path) -> None:
    fitting_started = threading.Event()
    fitting_release = threading.Event()

    class BlockingFitter(FakeFitter):
        def fit(self, observed_at: datetime, current_weights: dict[str, float]) -> FittedPolicyCandidate:
            fitting_started.set()
            assert fitting_release.wait(timeout=1.0)
            return super().fit(observed_at, current_weights)

    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([dict.fromkeys(TICKERS, 0.0)]),
        notifications=NotificationRecorder(),
        policy_fitter=BlockingFitter(tmp_path / "model-2.pt"),
    )
    paper = app.state.paper_dashboard
    paper.register_policy_revision(
        protocol_id="protocol-1",
        compatibility=load_config("policy.toml").compatibility_manifest,
    )
    paper.register_initial_fitted_policy(model_id="model-1", checkpoint="/models/model-1.pt")
    await paper.advance_once(observation(1))
    await paper.maybe_start_policy_fitting()
    assert await asyncio.to_thread(fitting_started.wait, 1.0)

    await paper.advance_once(observation(15, decision=True))
    decision = next(event for event in reversed(paper.store.events()) if event.event_type == "DecisionRecord")
    paper.control(
        "pause",
        expected_version=paper.state.state_version,
        idempotency_key="pause-during-fit",
    )

    assert decision.payload["model_id"] == "model-1"
    assert paper.state.last_observation_at == observation(15).timestamp
    assert paper.state.lifecycle.value == "Paused"
    fitting_release.set()
    assert paper._fitting_task is not None
    await paper._fitting_task
    assert paper.state.model_id == "model-2"
    assert paper.state.lifecycle.value == "Paused"
    paper.close()


@pytest.mark.anyio
async def test_failed_policy_activation_never_advances_the_durable_model(tmp_path) -> None:
    class Fitter(FakeFitter):
        def activate(self, candidate: FittedPolicyCandidate):
            del candidate
            raise RuntimeError("candidate cannot be activated")

    fitter = Fitter(tmp_path / "invalid.pt")
    notifications = NotificationRecorder()
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([]),
        notifications=notifications,
        policy_fitter=fitter,
    )
    paper = app.state.paper_dashboard
    await paper.advance_once(observation(1))
    old_model_id = paper.state.model_id
    account_id = paper.state.account_id
    quantities = paper.state.simulation.quantities.copy()
    await paper.maybe_start_policy_fitting()
    assert paper._fitting_task is not None
    await paper._fitting_task

    assert paper.state.model_id == old_model_id
    assert paper.state.account_id == account_id
    assert paper.state.simulation.quantities == quantities
    assert paper.state.fitting["status"] == "failed"
    assert "candidate cannot be activated" in paper.state.fitting["error"]
    assert paper.state.fitting["due_at"] == datetime(2026, 8, 23, 18, 0, tzinfo=UTC).isoformat()
    assert paper.state.fitting["started_at"] == datetime(2026, 8, 23, 18, 0, tzinfo=UTC).isoformat()
    assert notifications.calls[-1][0] == "fitting_failure"
    assert "PolicyHandoff" not in {event.event_type for event in paper.store.events()}
    paper.close()


@pytest.mark.anyio
async def test_interrupted_fitting_is_durably_failed_without_changing_account_or_model(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    first = create_application(
        database_path=database,
        tickers=TICKERS,
        clock=clock,
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([]),
        notifications=NotificationRecorder(),
        policy_fitter=FakeFitter(tmp_path / "model-2.pt"),
    ).state.paper_dashboard
    first.register_policy_revision(
        protocol_id="protocol-1",
        compatibility=load_config("policy.toml").compatibility_manifest,
    )
    first.register_initial_fitted_policy(model_id="model-1", checkpoint="/models/model-1.pt")
    await first.advance_once(observation(1))
    first.start_policy_fitting(due_at=clock.current)
    account_id = first.state.account_id
    first.close()

    clock.current += timedelta(minutes=1)
    notifications = NotificationRecorder()
    reopened = create_application(
        database_path=database,
        tickers=TICKERS,
        clock=clock,
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([]),
        notifications=notifications,
        policy_fitter=FakeFitter(tmp_path / "model-2.pt"),
    ).state.paper_dashboard
    await reopened.maybe_start_policy_fitting()

    assert reopened.state.account_id == account_id
    assert reopened.state.model_id == "model-1"
    assert reopened.state.fitting["status"] == "failed"
    assert reopened.state.fitting["due_at"] == datetime(2026, 8, 23, 18, 0, tzinfo=UTC).isoformat()
    assert reopened.state.fitting["started_at"] == datetime(2026, 8, 23, 18, 0, tzinfo=UTC).isoformat()
    assert "interrupted before Policy Handoff" in reopened.state.fitting["error"]
    assert notifications.calls[-1][0] == "fitting_failure"
    reopened.close()


def test_production_schedule_runs_once_on_the_first_window_after_sunday_deadline() -> None:
    backend = object.__new__(ProductionPolicyBackend)
    backend.config = load_config("policy.toml")
    sunday = datetime(2026, 8, 23, 18, 0, tzinfo=UTC)

    assert backend.is_due(sunday, {"status": "idle"})
    assert not backend.is_due(
        sunday,
        {"status": "idle", "completed_at": datetime(2026, 8, 23, 0, 1, tzinfo=UTC).isoformat()},
    )
    assert not backend.is_due(
        sunday,
        {
            "status": "failed",
            "due_at": sunday.isoformat(),
            "failed_at": (sunday + timedelta(minutes=1)).isoformat(),
            "next_retry_at": (sunday + timedelta(minutes=6)).isoformat(),
        },
    )
    assert backend.is_due(
        sunday + timedelta(days=7),
        {"status": "idle", "completed_at": datetime(2026, 8, 23, 0, 1, tzinfo=UTC).isoformat()},
    )
    assert backend.is_due(
        sunday + timedelta(days=7),
        {
            "status": "failed",
            "due_at": sunday.isoformat(),
            "failed_at": (sunday + timedelta(minutes=1)).isoformat(),
        },
    )


def test_scheduled_fitting_persists_a_content_addressed_immutable_checkpoint(tmp_path) -> None:
    candidate_bytes = b"scheduled-fitted-policy"

    class Adapter:
        def load(self) -> object:
            return object()

    class Backend:
        def paper(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            del args, kwargs
            return SimpleNamespace(
                refitted=True,
                model_bytes=candidate_bytes,
                target_weights=dict.fromkeys(TICKERS, 0.0),
            )

    backend = ProductionPolicyBackend(
        adapter=Adapter(),  # type: ignore[arg-type]
        backend=Backend(),  # type: ignore[arg-type]
        fitting_adapter=Adapter(),  # type: ignore[arg-type]
        attribution_adapter=Adapter(),  # type: ignore[arg-type]
        fitting_backend=Backend(),  # type: ignore[arg-type]
        checkpoint_directory=tmp_path,
        config=load_config("policy.toml"),
        device="cpu",
        fitted_model=b"current-policy",
    )

    first = backend.fit(observation(1).timestamp, dict.fromkeys(TICKERS, 0.0))
    repeated = backend.fit(observation(1).timestamp, dict.fromkeys(TICKERS, 0.0))
    checkpoint = Path(first.checkpoint)

    assert first.model_id == hashlib.sha256(candidate_bytes).hexdigest()
    assert checkpoint.name == f"{first.model_id}.pt"
    assert checkpoint.read_bytes() == candidate_bytes
    assert repeated == first
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="immutable Fitted Policy checkpoint hash collision"):
        backend.fit(observation(1).timestamp, dict.fromkeys(TICKERS, 0.0))


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
    compatibility = load_config("policy.toml").compatibility_manifest
    paper.register_policy_revision(
        protocol_id="protocol-1",
        compatibility=compatibility,
    )
    paper.advance_once_sync(observation(1))
    clock.current += timedelta(minutes=2)
    paper.register_policy_revision(
        protocol_id="protocol-2",
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


def test_compatible_policy_revision_segments_nonzero_account_performance(tmp_path) -> None:
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
    app = make_app(tmp_path / "paper.sqlite3", clock, FakeFeed([]), FakePolicy([target]))
    paper = app.state.paper_dashboard
    compatibility = load_config("policy.toml").compatibility_manifest
    paper.register_policy_revision(protocol_id="protocol-1", compatibility=compatibility)
    paper.register_initial_fitted_policy(model_id="model-1", checkpoint="/models/model-1.pt")
    paper.advance_once_sync(observation(0, decision=True))
    paper.advance_once_sync(observation(1))
    higher = dict.fromkeys(TICKERS, 100.0)
    higher["BTCUSDT"] = 120.0
    paper.advance_once_sync(replace(observation(2), mark_prices=higher))
    clock.current = observation(2).timestamp
    paper.register_policy_revision(
        protocol_id="protocol-2",
        compatibility=compatibility,
        model_compatibility={
            "model_id": "model-1",
            "method": "production-inference-preflight",
            "checked_at": clock.current.isoformat(),
        },
    )
    lower = dict.fromkeys(TICKERS, 100.0)
    lower["BTCUSDT"] = 80.0
    paper.advance_once_sync(replace(observation(3), mark_prices=lower))

    first, second = paper.history_snapshot()["protocol_segments"]

    assert first["protocol_id"] == "protocol-1"
    assert first["compounded_net_return"] > 0.0
    assert first["decisions"] == 1
    assert first["executable_changes"] == 1
    assert second["protocol_id"] == "protocol-2"
    assert second["compounded_net_return"] < 0.0
    assert second["starting_equity"] == first["current_equity"]
    paper.close()


def test_compatible_revision_requires_and_records_model_compatibility_evidence(tmp_path) -> None:
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    app = make_app(tmp_path / "paper.sqlite3", clock, FakeFeed([]), FakePolicy([]))
    paper = app.state.paper_dashboard
    compatibility = load_config("policy.toml").compatibility_manifest
    paper.register_policy_revision(protocol_id="protocol-1", compatibility=compatibility)
    paper.register_initial_fitted_policy(model_id="model-1", checkpoint="/models/model-1.pt")

    with pytest.raises(RuntimeError, match="compatibility evidence"):
        paper.register_policy_revision(protocol_id="protocol-2", compatibility=compatibility)

    evidence = {
        "model_id": "model-1",
        "method": "production-inference-preflight",
        "checked_at": clock.current.isoformat(),
    }
    paper.register_policy_revision(
        protocol_id="protocol-2",
        compatibility=compatibility,
        model_compatibility=evidence,
    )

    revision = next(event for event in reversed(paper.store.events()) if event.event_type == "PolicyRevision")
    assert revision.payload["model_compatibility"] == evidence
    assert paper.state.model_id == "model-1"
    assert paper.state.model_checkpoint == "/models/model-1.pt"
    paper.close()


def test_compatible_revision_and_replacement_policy_commit_as_one_boundary(tmp_path) -> None:
    @dataclass
    class SwitchingFitter(FakeFitter):
        active_model_id: str = "model-1"

        def activate(self, candidate: FittedPolicyCandidate):
            previous = self.active_model_id
            self.active_model_id = candidate.model_id

            def rollback() -> None:
                self.active_model_id = previous

            return rollback

    compatibility = load_config("policy.toml").compatibility_manifest
    fitter = SwitchingFitter(tmp_path / "model-2.pt")
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
    paper.register_policy_revision(protocol_id="protocol-1", compatibility=compatibility)
    paper.register_initial_fitted_policy(model_id="model-1", checkpoint="/models/model-1.pt")
    account_id = paper.state.account_id
    before_version = paper.state.state_version

    paper.register_policy_revision(
        protocol_id="protocol-2",
        compatibility=compatibility,
        fitted_policy=FittedPolicyCandidate("model-2", str(fitter.checkpoint), b"model-2"),
    )

    assert paper.state.account_id == account_id
    assert paper.state.state_version == before_version + 1
    assert paper.state.protocol_id == "protocol-2"
    assert paper.state.model_id == "model-2"
    assert paper.state.model_checkpoint == str(fitter.checkpoint)
    assert paper.state.fitting["status"] == "idle"
    assert paper.state.fitting["attempt_count"] == 0
    assert fitter.active_model_id == "model-2"
    boundary = [
        event.event_type for event in paper.store.events() if event.event_type in {"PolicyRevision", "PolicyHandoff"}
    ]
    assert boundary[-2:] == ["PolicyRevision", "PolicyHandoff"]
    paper.close()


def test_failed_revision_commit_rolls_back_the_replacement_policy(tmp_path, monkeypatch) -> None:
    @dataclass
    class SwitchingFitter(FakeFitter):
        active_model_id: str = "model-1"

        def activate(self, candidate: FittedPolicyCandidate):
            previous = self.active_model_id
            self.active_model_id = candidate.model_id

            def rollback() -> None:
                self.active_model_id = previous

            return rollback

    compatibility = load_config("policy.toml").compatibility_manifest
    fitter = SwitchingFitter(tmp_path / "model-2.pt")
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        notifications=NotificationRecorder(),
        policy_fitter=fitter,
    )
    paper = app.state.paper_dashboard
    paper.register_policy_revision(protocol_id="protocol-1", compatibility=compatibility)
    paper.register_initial_fitted_policy(model_id="model-1", checkpoint="/models/model-1.pt")
    committed = paper.store.commit

    def fail_revision_commit(*args: Any, **kwargs: Any):
        events = list(args[1])
        if any(event[0] == "PolicyRevision" for event in events):
            raise RuntimeError("simulated SQLite commit failure")
        return committed(args[0], events, *args[2:], **kwargs)

    monkeypatch.setattr(paper.store, "commit", fail_revision_commit)
    with pytest.raises(RuntimeError, match="simulated SQLite commit failure"):
        paper.register_policy_revision(
            protocol_id="protocol-2",
            compatibility=compatibility,
            fitted_policy=FittedPolicyCandidate("model-2", str(fitter.checkpoint), b"model-2"),
        )

    assert fitter.active_model_id == "model-1"
    assert paper.state.protocol_id == "protocol-1"
    assert paper.state.model_id == "model-1"
    assert paper.state.model_checkpoint == "/models/model-1.pt"
    paper.close()


@pytest.mark.parametrize(
    ("dimension", "replacement"),
    [
        ("trading_universe", ["BTCUSDT"]),
        ("account_currency", "EUR"),
        ("position_semantics", "spot-only-v1"),
        ("execution_semantics", "instant-fill-v1"),
        ("risk_semantics", "leveraged-v1"),
    ],
)
def test_incompatible_policy_revision_enters_durable_migration_required(
    tmp_path, dimension: str, replacement: object
) -> None:
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    notifications = NotificationRecorder()
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=clock,
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([]),
        notifications=notifications,
    )
    paper = app.state.paper_dashboard
    compatibility = load_config("policy.toml").compatibility_manifest
    paper.register_policy_revision(protocol_id="protocol-1", compatibility=compatibility)
    account_id = paper.state.account_id
    proposed = {**compatibility, dimension: replacement}

    paper.register_policy_revision(protocol_id="protocol-2", compatibility=proposed)
    paper.register_policy_revision(protocol_id="protocol-2", compatibility=proposed)

    assert paper.state.lifecycle.value == "Migration Required"
    assert paper.state.account_id == account_id
    assert paper.state.protocol_id == "protocol-1"
    assert paper.state.proposed_protocol_id == "protocol-2"
    assert paper.state.proposed_compatibility_manifest == proposed
    assert [event.event_type for event in paper.store.events()].count("MigrationRequired") == 1
    assert [kind for kind, _payload in notifications.calls] == ["incompatible_policy_revision"]
    paper.close()


def test_incompatible_revision_completes_a_pending_decision_as_cancelled(tmp_path) -> None:
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
    app = make_app(tmp_path / "paper.sqlite3", clock, FakeFeed([]), FakePolicy([target]))
    paper = app.state.paper_dashboard
    compatibility = load_config("policy.toml").compatibility_manifest
    paper.register_policy_revision(protocol_id="protocol-1", compatibility=compatibility)
    paper.register_initial_fitted_policy(model_id="model-1", checkpoint="/models/model-1.pt")
    paper.advance_once_sync(observation(0, decision=True))
    decision = next(event for event in paper.store.events() if event.event_type == "DecisionRecord")

    proposed = {**compatibility, "risk_semantics": "leveraged-v1"}
    paper.register_policy_revision(protocol_id="protocol-2", compatibility=proposed)

    assert paper.state.pending_execution is None
    assert paper.state.simulation.pending is None
    detail = paper.event_snapshot(decision.event_id)
    assert detail["decision"]["outcome"] == "cancelled_by_policy_revision"
    assert any(
        event.event_type == "DecisionCompleted"
        and event.decision_id == decision.decision_id
        and event.payload["outcome"] == "cancelled_by_policy_revision"
        for event in paper.store.events()
    )
    paper.close()


def test_trading_universe_revision_reopens_existing_account_for_migration_gate(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    now = datetime(2026, 8, 23, 18, 0, tzinfo=UTC)
    first = create_application(
        database_path=database,
        tickers=TICKERS,
        clock=FakeClock(now),
        notifications=NotificationRecorder(),
    ).state.paper_dashboard
    compatibility = load_config("policy.toml").compatibility_manifest
    first.register_policy_revision(protocol_id="protocol-1", compatibility=compatibility)
    account_id = first.state.account_id
    first.close()

    proposed_tickers = ("BTCUSDT",)
    reopened = create_application(
        database_path=database,
        tickers=proposed_tickers,
        clock=FakeClock(now + timedelta(minutes=1)),
        notifications=NotificationRecorder(),
    ).state.paper_dashboard
    proposed = {**compatibility, "trading_universe": list(proposed_tickers)}
    reopened.register_policy_revision(protocol_id="protocol-2", compatibility=proposed)

    assert reopened.state.account_id == account_id
    assert reopened.state.tickers == TICKERS
    assert reopened.state.lifecycle.value == "Migration Required"
    reopened.close()


def test_migration_required_preserves_positions_and_blocks_decisions_across_restart(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    clock = FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC))
    target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
    first = create_application(
        database_path=database,
        tickers=TICKERS,
        clock=clock,
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([target, target]),
        notifications=NotificationRecorder(),
    ).state.paper_dashboard
    compatibility = load_config("policy.toml").compatibility_manifest
    first.register_policy_revision(protocol_id="protocol-1", compatibility=compatibility)
    first.register_initial_fitted_policy(model_id="model-1", checkpoint="/models/model-1.pt")
    first.advance_once_sync(observation(0, decision=True))
    first.advance_once_sync(observation(1))
    account_id = first.state.account_id
    quantities = first.state.simulation.quantities.copy()
    proposed = {**compatibility, "risk_semantics": "leveraged-v1"}
    first.register_policy_revision(protocol_id="protocol-2", compatibility=proposed)
    first.advance_once_sync(observation(15, decision=True))
    assert [event.event_type for event in first.store.events()].count("DecisionRecord") == 1
    clock.current = observation(15).timestamp
    first.close()

    clock.current += timedelta(minutes=1)
    reopened = create_application(
        database_path=database,
        tickers=TICKERS,
        clock=clock,
        market_feed=FakeFeed([]),
        policy_backend=FakePolicy([target]),
        notifications=NotificationRecorder(),
    ).state.paper_dashboard
    reopened.register_policy_revision(protocol_id="protocol-2", compatibility=proposed)
    reopened.advance_once_sync(observation(30, decision=True))

    assert reopened.state.account_id == account_id
    assert reopened.state.simulation.quantities == quantities
    assert reopened.state.lifecycle.value == "Migration Required"
    assert reopened.state.model_id == "model-1"
    assert reopened.state.protocol_id == "protocol-1"
    assert [event.event_type for event in reopened.store.events()].count("MigrationRequired") == 1
    assert [event.event_type for event in reopened.store.events()].count("DecisionRecord") == 1
    reopened.close()


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
    assert paper.state.fitting["status"] == "idle"
    assert paper.state.fitting["attempt_count"] == 0
    assert [event.event_type for event in paper.store.events()].count("FittedPolicySelected") == 1
    assert "PolicyHandoff" not in {event.event_type for event in paper.store.events()}
    paper.close()


def test_decision_cannot_replace_the_durable_fitted_policy_or_protocol(tmp_path) -> None:
    app = make_app(
        tmp_path / "paper.sqlite3",
        FakeClock(datetime(2026, 8, 23, 18, 0, tzinfo=UTC)),
        FakeFeed([]),
        FakePolicy([dict.fromkeys(TICKERS, 0.0)]),
    )
    paper = app.state.paper_dashboard
    compatibility = load_config("policy.toml").compatibility_manifest
    paper.register_policy_revision(protocol_id="protocol-durable", compatibility=compatibility)
    paper.register_initial_fitted_policy(model_id="model-durable", checkpoint="/models/durable.pt")

    with pytest.raises(RuntimeError, match="durable Fitted Policy"):
        paper.advance_once_sync(observation(0, decision=True))

    assert paper.state.model_id == "model-durable"
    assert paper.state.protocol_id == "protocol-durable"
    assert "DecisionRecord" not in {event.event_type for event in paper.store.events()}
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
    canonical = canonical_policy_input(periods=3)

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
    backend._serialize_attribution_input = lambda **_kwargs: b"exact prepared input"  # type: ignore[method-assign]
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
    assert decision.protocol_id == config.protocol_id
    changed = backend.decide(observation(0, decision=True), {**current, "BTCUSDT": 0.99})
    assert changed.input_id != decision.input_id


def test_production_decision_rejects_implicit_refitting(tmp_path) -> None:
    canonical = canonical_policy_input(periods=3)

    class Adapter:
        def load(self) -> CanonicalDataset:
            return canonical

    class Backend:
        def paper(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            del args, kwargs
            return SimpleNamespace(
                refitted=True,
                model_bytes=b"replacement-policy",
                target_weights=dict.fromkeys(TICKERS, 0.0),
            )

    selected = b"selected-policy"
    backend = ProductionPolicyBackend(
        adapter=Adapter(),
        backend=Backend(),
        fitting_adapter=Adapter(),
        attribution_adapter=Adapter(),
        fitting_backend=Backend(),
        checkpoint_directory=tmp_path,
        config=load_config("policy.toml"),
        device="cpu",
        fitted_model=selected,
    )

    with pytest.raises(RuntimeError, match="ordinary decision must not replace"):
        backend.decide(observation(0, decision=True), dict.fromkeys(TICKERS, 0.0))

    assert backend.fitted_model == selected


def test_causal_policy_input_keeps_the_signal_time_hash_when_later_rows_arrive() -> None:
    signal_time = observation(0).timestamp
    original = canonical_policy_input(periods=3)
    later_rows = canonical_policy_input(periods=6)

    original_slice = _causal_policy_input(original, signal_time)
    later_slice = _causal_policy_input(later_rows, signal_time)

    assert original_slice.identity_hash == later_slice.identity_hash
    assert all(data.perpetual.index.max() < pd.Timestamp(signal_time) for data in later_slice.instruments.values())


def test_exporting_prepared_attribution_input_transfers_it_once() -> None:
    backend = object.__new__(ProductionPolicyBackend)
    backend._attribution_inputs = {"input-1": b"exact prepared input"}

    assert backend.export_attribution_input("input-1") == b"exact prepared input"
    with pytest.raises(RuntimeError, match="exact prepared attribution input is unavailable"):
        backend.export_attribution_input("input-1")


def test_production_diagnostics_report_the_rocm_runtime(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "AMD Radeon RX 7800 XT")
    monkeypatch.setattr(torch.version, "hip", "6.4.0")
    backend = object.__new__(ProductionPolicyBackend)
    backend.device = "cuda:0"

    diagnostics = backend.diagnostics()

    assert diagnostics == {
        "backend": "ROCm",
        "requested_device": "cuda:0",
        "actual_device": "cuda:0",
        "device_name": "AMD Radeon RX 7800 XT",
        "rocm_version": "6.4.0",
        "inference": "ready",
        "fitting": "ready",
    }
