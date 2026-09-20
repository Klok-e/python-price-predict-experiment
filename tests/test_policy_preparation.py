from __future__ import annotations

import asyncio
import hashlib
import threading

import httpx
import pytest

from netgrowth.config import load_config
from netgrowth.paper_dashboard.application import FittedPolicyCandidate, create_application
from netgrowth.paper_dashboard.domain import PolicyDecision
from tests.test_paper_dashboard import TICKERS, FakeClock, FakeFitter, observation


async def _wait_for_policy_preparation(client: httpx.AsyncClient) -> None:
    while (await client.get("/api/system")).json()["service"]["policy_preparing"]:
        await asyncio.sleep(0)


@pytest.mark.anyio
async def test_policy_preparation_keeps_market_and_account_reads_available(tmp_path) -> None:
    prepared = threading.Event()
    release_preparation = threading.Event()
    inference_lock = threading.Lock()

    def prepare() -> None:
        with inference_lock:
            prepared.set()
            assert release_preparation.wait(timeout=5.0)

    class Policy:
        def decide(self, observed, current_weights):
            with inference_lock:
                target = {**dict.fromkeys(TICKERS, 0.0), "BTCUSDT": 0.5}
                return PolicyDecision(target, target, "model-1", observed.input_id, "protocol-1")

    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(observation(0).timestamp),
        policy_backend=Policy(),
        policy_preparer=prepare,
    )
    paper = app.state.paper_dashboard

    async with app.router.lifespan_context(app):
        assert await asyncio.to_thread(prepared.wait, 0.5)
        advance = asyncio.create_task(asyncio.to_thread(paper.advance_once_sync, observation(0, decision=True)))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                live = await asyncio.wait_for(client.get("/api/live"), timeout=1.0)
            await asyncio.wait_for(asyncio.shield(advance), timeout=1.0)
            assert live.status_code == 200
            assert live.json()["account"]["current_equity"] == 10_000.0
            assert not any(event.event_type == "DecisionRecord" for event in paper.store.events())
        finally:
            release_preparation.set()
            await advance

        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await asyncio.wait_for(_wait_for_policy_preparation(client), timeout=1.0)
        paper.advance_once_sync(observation(1, decision=True))
        assert [event.event_type for event in paper.store.events()].count("DecisionRecord") == 1


@pytest.mark.anyio
async def test_policy_preparation_defers_staged_revision_activation_until_market_is_warm(tmp_path) -> None:
    prepared = threading.Event()
    release_preparation = threading.Event()

    def prepare() -> None:
        prepared.set()
        assert release_preparation.wait(timeout=5.0)

    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    candidate = FittedPolicyCandidate(hashlib.sha256(b"candidate").hexdigest(), str(checkpoint), b"candidate")
    app = create_application(
        database_path=tmp_path / "paper.sqlite3",
        tickers=TICKERS,
        clock=FakeClock(observation(0).timestamp),
        policy_fitter=FakeFitter(checkpoint),
        policy_preparer=prepare,
    )
    paper = app.state.paper_dashboard
    proposed = load_config("policy.toml").compatibility_manifest
    incumbent = {**proposed, "execution_semantics": "delayed-midpoint-adverse-cost-v1"}
    validation = {
        "protocol_id": "new",
        "trained_protocol_id": "new",
        "model_id": candidate.model_id,
        "fitted_at": observation(0).timestamp.isoformat(),
        "folds": [{"net_return": 0.01, "max_drawdown": 0.02} for _ in range(12)],
    }

    async with app.router.lifespan_context(app):
        assert await asyncio.to_thread(prepared.wait, 0.5)
        paper.register_policy_revision(protocol_id="old", compatibility=incumbent)
        paper.register_initial_fitted_policy(model_id="old-model", checkpoint="old.pt")
        paper.advance_once_sync(observation(0))
        paper.stage_policy_revision(
            protocol_id="new", compatibility=proposed, candidate=candidate, validation=validation
        )
        assert paper.state.revision["status"] == "draining"
        assert paper.state.protocol_id == "old"

        release_preparation.set()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await asyncio.wait_for(_wait_for_policy_preparation(client), timeout=1.0)
        paper.advance_once_sync(observation(1))
        assert paper.state.revision["status"] == "active"
        assert paper.state.protocol_id == "new"
