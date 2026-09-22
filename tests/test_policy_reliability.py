from datetime import timedelta

from netgrowth.paper_dashboard.application import create_application
from tests.test_paper_dashboard import TICKERS, FakeClock, FakePolicy, observation


def test_below_threshold_completes_at_signal_and_cannot_expire(tmp_path):
    clock = FakeClock(observation(0).timestamp)
    paper = create_application(
        database_path=tmp_path / "paper.db", clock=clock, policy_backend=FakePolicy([dict.fromkeys(TICKERS, 0.002)])
    ).state.paper_dashboard
    paper.advance_once_sync(observation(0, decision=True))
    assert paper.state.pending_execution is None
    events = paper.store.events(paper.state.account_id)
    assert any(e.event_type == "DecisionCompleted" and e.payload["outcome"] == "below_threshold" for e in events)
    clock.current += timedelta(minutes=10)
    paper.advance_once_sync(observation(10))
    assert not any(e.event_type == "MissedExecution" for e in paper.store.events(paper.state.account_id))
    paper.close()


def test_failed_fit_retains_success_and_persists_retry_deadline(tmp_path):
    clock = FakeClock(observation(0).timestamp)
    path = tmp_path / "paper.db"
    paper = create_application(database_path=path, clock=clock).state.paper_dashboard
    paper.complete_policy_handoff(new_model_id="retained", checkpoint="retained.pt")
    successful_at = clock.current.isoformat()
    clock.current += timedelta(days=7)
    paper.start_policy_fitting(due_at=clock.current)
    paper.fail_policy_fitting("provider unavailable")
    assert paper.state.model_id == "retained"
    assert paper.state.fitting["last_successful_fit_at"] == successful_at
    assert paper.state.fitting["attempt_count"] == 1
    assert paper.state.fitting["next_retry_at"] == (clock.current + timedelta(minutes=5)).isoformat()
    paper.close()
    paper = create_application(database_path=path, clock=clock).state.paper_dashboard
    assert paper.state.fitting["attempt_count"] == 1
    assert paper.state.fitting["next_retry_at"] == (clock.current + timedelta(minutes=5)).isoformat()
    paper.close()


def test_retry_backoff_caps_and_keeps_the_original_cycle(tmp_path):
    clock = FakeClock(observation(0).timestamp)
    paper = create_application(database_path=tmp_path / "paper.db", clock=clock).state.paper_dashboard
    cycle = clock.current.isoformat()
    for attempt, delay in enumerate((5, 15, 30, 60, 60), 1):
        paper.start_policy_fitting(due_at=clock.current)
        paper.fail_policy_fitting("unavailable")
        assert paper.state.fitting["attempt_count"] == attempt
        assert paper.state.fitting["cycle_at"] == cycle
        assert paper.state.fitting["next_retry_at"] == (clock.current + timedelta(minutes=delay)).isoformat()
        clock.current += timedelta(minutes=delay)
    paper.close()


def test_production_retries_failed_fit_when_deadline_arrives():
    from netgrowth.config import load_config
    from netgrowth.paper_dashboard.production import ProductionPolicyBackend

    backend = object.__new__(ProductionPolicyBackend)
    backend.config = load_config("policy.toml")
    now = observation(0).timestamp
    fitting = {"status": "failed", "due_at": now.isoformat(), "next_retry_at": (now + timedelta(minutes=5)).isoformat()}
    assert not backend.is_due(now + timedelta(minutes=4), fitting)
    assert backend.is_due(now + timedelta(minutes=5), fitting)
    assert backend.is_due(now + timedelta(days=2), fitting)


def test_revision_waits_for_fill_and_starts_hold_from_identical_positions(tmp_path):
    import hashlib
    from dataclasses import replace

    from netgrowth.config import load_config
    from netgrowth.paper_dashboard.application import FittedPolicyCandidate
    from tests.test_paper_dashboard import FakeFitter

    clock = FakeClock(observation(0).timestamp)
    checkpoint = tmp_path / "new.pt"
    checkpoint.write_bytes(b"new model")
    fitter = FakeFitter(checkpoint)
    paper = create_application(
        database_path=tmp_path / "paper.db",
        clock=clock,
        policy_backend=FakePolicy([dict.fromkeys(TICKERS, 0.1)]),
        policy_fitter=fitter,
    ).state.paper_dashboard
    manifest = load_config("policy.toml").compatibility_manifest
    old = {**manifest, "execution_semantics": "delayed-midpoint-adverse-cost-v1"}
    paper.register_policy_revision(protocol_id="protocol-1", compatibility=old)
    paper.register_initial_fitted_policy(model_id="model-1", checkpoint="old.pt")
    paper.advance_once_sync(observation(0, decision=True))
    paper.state.simulation.pending = replace(paper.state.simulation.pending, eligibility_at_signal=False)
    model_id = hashlib.sha256(b"new model").hexdigest()
    candidate = FittedPolicyCandidate(model_id, str(checkpoint), b"new model")
    validation = {
        "protocol_id": "protocol-2",
        "trained_protocol_id": "protocol-2",
        "model_id": model_id,
        "folds": [{"net_return": 0.01, "max_drawdown": 0.02} for _ in range(12)],
    }
    original_id = paper.state.account_id
    paper.stage_policy_revision(
        protocol_id="protocol-2", compatibility=manifest, candidate=candidate, validation=validation
    )
    assert paper.state.protocol_id == "protocol-1"
    assert paper.state.revision["status"] == "draining"
    clock.current = observation(1).timestamp
    paper.advance_once_sync(observation(1))
    assert paper.state.revision["status"] == "active"
    assert paper.state.protocol_id == "protocol-2"
    assert paper.state.account_id == original_id
    assert paper.state.hold_benchmark.quantities == paper.state.simulation.quantities
    hold = paper.live_snapshot()["hold_benchmark"]
    assert hold["equity"] == paper.state.simulation.equity
    assert hold["excess_pnl"] == 0
    assert hold["started_at"] == clock.current.isoformat()
    active_revision = paper.state.revision
    paper.state.revision = {"status": "draining"}
    assert paper.live_snapshot()["hold_benchmark"]["started_at"] == clock.current.isoformat()
    paper.state.revision = active_revision
    quantities = paper.state.simulation.quantities.copy()
    paper.close()
    paper = create_application(database_path=tmp_path / "paper.db", clock=clock).state.paper_dashboard
    assert paper.state.hold_benchmark.quantities == quantities
    assert paper.state.revision["status"] == "active"
    paper.close()


def test_hold_benchmark_counts_price_moves_and_funding_without_entry_cost():
    from netgrowth.accounting import HoldBenchmark

    hold = HoldBenchmark.start(10000, {"BTC": 2, "ETH": -3}, {"BTC": 100, "ETH": 10})
    assert hold.mark({"BTC": 100, "ETH": 10})["equity"] == 10000
    hold.apply_funding({"BTC": 0.01, "ETH": 0.02}, {"BTC": 110, "ETH": 12})
    result = hold.mark({"BTC": 110, "ETH": 12})
    assert abs(result["equity"] - 10012.52) < 1e-8
    assert abs(result["funding"] + 1.48) < 1e-12
    assert hold.quantities == {"BTC": 2, "ETH": -3}


def _revision_fixture(tmp_path):
    import hashlib

    from netgrowth.config import load_config
    from netgrowth.paper_dashboard.application import FittedPolicyCandidate
    from tests.test_paper_dashboard import FakeFitter

    clock = FakeClock(observation(0).timestamp)
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"candidate")
    model_id = hashlib.sha256(b"candidate").hexdigest()
    fitter = FakeFitter(checkpoint)
    paper = create_application(
        database_path=tmp_path / "paper.db", clock=clock, policy_fitter=fitter
    ).state.paper_dashboard
    proposed = load_config("policy.toml").compatibility_manifest
    original = {**proposed, "execution_semantics": "delayed-midpoint-adverse-cost-v1"}
    paper.register_policy_revision(protocol_id="old", compatibility=original)
    paper.register_initial_fitted_policy(model_id="old-model", checkpoint="old.pt")
    paper.advance_once_sync(observation(0))
    candidate = FittedPolicyCandidate(model_id, str(checkpoint), b"candidate")
    validation = {
        "protocol_id": "new",
        "trained_protocol_id": "new",
        "model_id": model_id,
        "fitted_at": clock.current.isoformat(),
        "folds": [{"net_return": 0.01, "max_drawdown": 0.02} for _ in range(12)],
    }
    return paper, clock, fitter, proposed, candidate, validation


def test_revision_failed_commit_restores_incumbent_and_retries_once(tmp_path, monkeypatch):
    import pytest

    paper, clock, fitter, manifest, candidate, validation = _revision_fixture(tmp_path)
    account_id = paper.state.account_id
    commit = paper.store.commit

    def fail_activation(state, events, **kwargs):
        if any(e[0] == "PolicyRevision" for e in events):
            raise OSError("disk full")
        return commit(state, events, **kwargs)

    monkeypatch.setattr(paper.store, "commit", fail_activation)
    with pytest.raises(OSError, match="disk full"):
        paper.stage_policy_revision(
            protocol_id="new", compatibility=manifest, candidate=candidate, validation=validation
        )
    assert paper.state.protocol_id == "old"
    assert paper.state.model_id == "old-model"
    assert paper.state.hold_benchmark is None
    assert fitter.activated == []
    assert paper.store.load_active_account().revision["status"] == "draining"
    monkeypatch.setattr(paper.store, "commit", commit)
    clock.current = observation(1).timestamp
    paper.advance_once_sync(observation(1))
    assert paper.state.protocol_id == "new"
    assert paper.state.account_id == account_id
    assert len(fitter.activated) == 1
    assert (
        len(
            [
                e
                for e in paper.store.events()
                if e.event_type == "PolicyRevision" and e.payload["new_protocol_id"] == "new"
            ]
        )
        == 1
    )
    paper.close()


def test_revision_rejects_failed_or_incomplete_evidence_without_mutation(tmp_path):
    import pytest

    paper, _, fitter, manifest, candidate, validation = _revision_fixture(tmp_path)
    version = paper.state.state_version
    for folds in (
        [],
        [{"net_return": -0.01, "max_drawdown": 0.02} for _ in range(12)],
        [{"net_return": 0.01, "max_drawdown": 0.21} for _ in range(12)],
    ):
        with pytest.raises(ValueError):
            paper.stage_policy_revision(
                protocol_id="new",
                compatibility=manifest,
                candidate=candidate,
                validation={**validation, "folds": folds},
            )
    assert paper.state.state_version == version
    assert paper.state.protocol_id == "old"
    assert not fitter.activated
    paper.close()


def test_revision_waits_for_fresh_mark_after_restart(tmp_path):
    paper, clock, fitter, manifest, candidate, validation = _revision_fixture(tmp_path)
    clock.current = observation(10).timestamp
    paper.stage_policy_revision(protocol_id="new", compatibility=manifest, candidate=candidate, validation=validation)
    assert paper.state.protocol_id == "old"
    paper.close()
    paper = create_application(
        database_path=tmp_path / "paper.db", clock=clock, policy_fitter=fitter
    ).state.paper_dashboard
    paper.stage_policy_revision(protocol_id="new", compatibility=manifest, candidate=candidate, validation=validation)
    assert paper.state.protocol_id == "old"
    assert len([e for e in paper.store.events() if e.event_type == "PolicyRevisionStaged"]) == 1
    paper.advance_once_sync(observation(10))
    assert paper.state.protocol_id == "new"
    assert paper.state.hold_benchmark is not None
    paper.close()


def test_slow_notification_does_not_block_market_advancement(tmp_path):
    import threading

    started, release = threading.Event(), threading.Event()

    class SlowSink:
        def notify(self, kind, payload):
            started.set()
            assert release.wait(5)

    clock = FakeClock(observation(0).timestamp)
    paper = create_application(
        database_path=tmp_path / "paper.db", clock=clock, notifications=SlowSink()
    ).state.paper_dashboard
    try:
        paper.store.enqueue_notification(paper.state.account_id, "slow", "data_stale", {}, clock.current)
        advanced = threading.Event()

        def advance():
            paper._deliver_notifications()
            paper.advance_once_sync(observation(1))
            advanced.set()

        thread = threading.Thread(target=advance)
        thread.start()
        assert started.wait(2)
        assert advanced.wait(2), "market advancement waited for desktop notification delivery"
        assert not release.is_set()
        assert paper.state.last_observation_at == observation(1).timestamp
    finally:
        release.set()
        thread.join(timeout=3)
        paper.close()


def test_risk_stopped_staged_revision_reopens_and_survives_manual_reset(tmp_path):
    from netgrowth.paper_dashboard.domain import LifecycleState

    paper, clock, fitter, manifest, candidate, validation = _revision_fixture(tmp_path)
    clock.current = observation(10).timestamp
    paper.stage_policy_revision(protocol_id="new", compatibility=manifest, candidate=candidate, validation=validation)
    paper.state.lifecycle = LifecycleState.RISK_STOPPED
    paper.state.model_fitted_at = observation(0).timestamp.isoformat()
    paper._commit([])
    paper.close()
    paper = create_application(
        database_path=tmp_path / "paper.db", clock=clock, policy_fitter=fitter
    ).state.paper_dashboard
    paper.stage_policy_revision(protocol_id="new", compatibility=manifest, candidate=candidate, validation=validation)
    assert paper.state.lifecycle is LifecycleState.RISK_STOPPED
    assert paper.state.protocol_id == "old"
    paper._complete_reset(clock.current)
    assert paper.state.revision["status"] == "draining"
    assert paper.state.model_fitted_at == observation(0).timestamp.isoformat()
    paper.close()
    paper = create_application(
        database_path=tmp_path / "paper.db", clock=clock, policy_fitter=fitter
    ).state.paper_dashboard
    paper.stage_policy_revision(protocol_id="new", compatibility=manifest, candidate=candidate, validation=validation)
    assert paper.state.protocol_id == "old"
    paper.advance_once_sync(observation(10))
    assert paper.state.protocol_id == "new"
    assert paper.state.hold_benchmark.starting_equity == paper.config.initial_equity
    paper.close()


def test_fresh_start_rejects_checkpoint_not_bound_to_validation(tmp_path, monkeypatch):
    import json

    import pytest

    from netgrowth.config import load_config
    from netgrowth.paper_dashboard.production import run_dashboard

    operational = tmp_path / "paper"
    (operational / "checkpoints").mkdir(parents=True)
    (operational / "checkpoints" / "current.pt").write_bytes(b"unvalidated incumbent")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "evidence-state.json").write_text(
        json.dumps(
            {
                "validation_passed": True,
                "validated_protocol": load_config("policy.toml").protocol_id,
                "validated_model_hash": "a-different-model",
            }
        )
    )
    with pytest.raises(RuntimeError, match="validated model identity"):
        run_dashboard("policy.toml", tmp_path / "data", operational, "127.0.0.1", 9999, "cpu")


def test_retained_candidate_survives_external_file_removal_and_model_handoff(tmp_path):
    from hashlib import sha256

    from netgrowth.paper_dashboard.production import ProductionPolicyBackend, _retain_checkpoint

    payload = b"immutable candidate"
    model_id = sha256(payload).hexdigest()
    source = tmp_path / "external.pt"
    source.write_bytes(payload)
    retained = _retain_checkpoint(tmp_path / "checkpoints", model_id, source.read_bytes())
    source.unlink()
    backend = object.__new__(ProductionPolicyBackend)
    backend.fitted_model = b"a later weekly fit"
    backend.checkpoint_directory = tmp_path / "checkpoints"
    assert retained.read_bytes() == payload
    assert backend._model_bytes(model_id) == payload
