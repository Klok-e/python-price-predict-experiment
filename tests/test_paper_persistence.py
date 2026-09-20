from __future__ import annotations

import sqlite3
from copy import deepcopy
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from netgrowth.paper_dashboard.domain import LifecycleState
from netgrowth.paper_dashboard.persistence import SCHEMA_VERSION, SQLitePaperStore, StateVersionConflict

TICKERS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")


def test_flat_start_is_atomic_and_reopening_does_not_create_another_account(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    started_at = datetime(2026, 8, 23, 18, 0, tzinfo=UTC)

    with SQLitePaperStore(database) as first:
        created = first.open_active_account(started_at, TICKERS)
        assert first.journal_mode == "wal"
        assert created.lifecycle is LifecycleState.TRADING
        assert created.starting_equity == 10_000.0
        assert created.simulation.equity == 10_000.0
        assert created.simulation.quantities == dict.fromkeys(TICKERS, 0.0)
        assert [event.event_type for event in first.events(created.account_id)] == ["FlatStart"]

    with SQLitePaperStore(database) as reopened:
        restored = reopened.open_active_account(started_at, TICKERS)
        assert restored.account_id == created.account_id
        assert restored.state_version == created.state_version
        assert [event.event_type for event in reopened.events(restored.account_id)] == ["FlatStart"]


def test_v1_database_is_backed_up_before_the_durable_recovery_migration(tmp_path) -> None:
    database = tmp_path / "paper.sqlite3"
    backups = tmp_path / "backups"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE accounts (account_id TEXT PRIMARY KEY);
        CREATE TABLE notification_outbox (
            notification_id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            dedupe_key TEXT NOT NULL UNIQUE,
            kind TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            delivered_at TEXT,
            error TEXT
        );
        PRAGMA user_version = 1;
        """
    )
    connection.close()

    with SQLitePaperStore(database, backup_directory=backups) as store:
        assert store._connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        columns = {row["name"] for row in store._connection.execute("PRAGMA table_info(notification_outbox)")}
        assert {"attempt_count", "failure_count", "last_attempt_at", "next_attempt_at"} <= columns

    assert list(backups.glob("pre-migration-v1-*.sqlite3"))


def test_notification_failure_retries_with_durable_backoff_and_retains_history(tmp_path) -> None:
    now = datetime(2026, 8, 23, 18, 0, tzinfo=UTC)
    with SQLitePaperStore(tmp_path / "paper.sqlite3") as store:
        state = store.open_active_account(now, TICKERS)
        assert store.enqueue_notification(state.account_id, "stale:1", "data_stale", {"reason": "offline"}, now)
        notification = store.pending_notifications(now)[0]

        store.mark_notification(str(notification["notification_id"]), now, "desktop unavailable")
        assert store.pending_notifications(now) == []
        retry = store.pending_notifications(now.replace(minute=5))[0]
        assert retry["notification_id"] == notification["notification_id"]

        store.mark_notification(str(notification["notification_id"]), now.replace(minute=5), "desktop unavailable")
        assert store.pending_notifications(now.replace(minute=19)) == []
        retry = store.pending_notifications(now.replace(minute=20))[0]
        assert retry["notification_id"] == notification["notification_id"]
        failed = store.notification_health()
        assert failed["healthy"] is False
        assert failed["retrying"] == 1
        assert failed["historical_failures"] == 2

        store.mark_notification(str(notification["notification_id"]), now.replace(minute=20), None)
        healthy = store.notification_health()
        assert healthy["healthy"] is True
        assert healthy["pending"] == 0
        assert healthy["historical_failures"] == 2


def test_durable_attribution_input_commits_with_its_decision_and_releases_only_on_completion(tmp_path) -> None:
    now = datetime(2026, 8, 23, 18, 0, tzinfo=UTC)
    payload = b"exact prepared model input"
    with SQLitePaperStore(tmp_path / "paper.sqlite3") as store:
        state = store.open_active_account(now, TICKERS)
        state.state_version += 1
        state.snapshot_version += 1
        store.commit(
            state,
            [("DecisionRecord", now, {"input_id": "input-1"}, "decision-1", None)],
            expected_state_version=1,
            attribution_inputs=[
                ("input-1", "decision-1", "model-1", sha256(payload).hexdigest(), payload, now),
            ],
        )
        recovered = store.attribution_input("input-1")
        assert recovered["decision_id"] == "decision-1"
        assert recovered["model_id"] == "model-1"
        assert recovered["payload"] == payload

        state.state_version += 1
        state.snapshot_version += 1
        store.commit(
            state,
            [("ModelAttribution", now, {"status": "complete"}, "decision-1", None)],
            expected_state_version=2,
        )
        with pytest.raises(RuntimeError, match="released after successful completion"):
            store.attribution_input("input-1")
        metadata = store._connection.execute(
            "SELECT input_hash, released_at FROM attribution_inputs WHERE input_id = 'input-1'"
        ).fetchone()
        assert metadata["input_hash"] == sha256(payload).hexdigest()
        assert metadata["released_at"] is not None


def test_restart_clamps_new_detected_window_end_and_labels_historical_invalid_timing(tmp_path) -> None:
    started_at = datetime(2026, 8, 23, 18, 0, tzinfo=UTC)
    with SQLitePaperStore(tmp_path / "paper.sqlite3") as store:
        state = store.open_active_account(started_at, TICKERS)
        state.state_version += 1
        state.snapshot_version += 1
        store.commit(
            state,
            [("OperatingWindowOpened", started_at, {"window_id": "old"}, None, None)],
            expected_state_version=1,
            operating_window=("old", started_at, None),
        )
        state.state_version += 1
        state.snapshot_version += 1
        restarted_at = started_at.replace(minute=5)
        store.commit(
            state,
            [("OperatingWindowOpened", restarted_at, {"window_id": "new"}, None, None)],
            expected_state_version=2,
            operating_window=("new", restarted_at, started_at.replace(minute=59, hour=17)),
        )
        first = store.operating_windows(state.account_id)[0]
        assert first["ended_at"] == started_at.isoformat()
        assert first["timing_valid"] is True

        store._connection.execute(
            """INSERT INTO operating_windows VALUES (?, ?, ?, ?, ?)""",
            ("historical-invalid", state.account_id, restarted_at.isoformat(), started_at.isoformat(), "legacy"),
        )
        historical = next(
            window
            for window in store.operating_windows(state.account_id)
            if window["window_id"] == "historical-invalid"
        )
        assert historical["timing_valid"] is False
        assert historical["timing_error"] == "ended_before_started"


def test_event_and_versioned_snapshot_commit_together_and_reject_a_stale_owner(tmp_path) -> None:
    now = datetime(2026, 8, 23, 18, 0, tzinfo=UTC)
    with SQLitePaperStore(tmp_path / "paper.sqlite3") as store:
        current = store.open_active_account(now, TICKERS)
        stale = deepcopy(current)
        current.lifecycle = LifecycleState.PAUSED
        current.state_version += 1
        current.snapshot_version += 1

        appended = store.commit(
            current,
            [("OperatorIntervention", now, {"action": "pause"}, None, None)],
            expected_state_version=1,
        )

        assert appended[0].sequence == 2
        assert store.load_active_account() == current
        stale.lifecycle = LifecycleState.PAUSED
        stale.state_version += 1
        stale.snapshot_version += 1
        with pytest.raises(StateVersionConflict):
            store.commit(
                stale,
                [("OperatorIntervention", now, {"action": "pause"}, None, None)],
                expected_state_version=1,
            )
        assert [event.sequence for event in store.events(current.account_id)] == [1, 2]


def test_daily_online_backups_are_deduplicated_and_rotate_to_seven(tmp_path) -> None:
    started_at = datetime(2026, 8, 1, 18, 0, tzinfo=UTC)
    backup_directory = tmp_path / "backups"
    with SQLitePaperStore(tmp_path / "paper.sqlite3", backup_directory=backup_directory) as store:
        store.open_active_account(started_at, TICKERS)
        first = store.create_daily_backup(started_at)
        assert first is not None and first.is_file()
        assert store.create_daily_backup(started_at.replace(hour=23)) is None
        for offset in range(1, 9):
            assert store.create_daily_backup(started_at.replace(day=started_at.day + offset)) is not None

        retained = sorted(backup_directory.glob("paper-*.sqlite3"))
        assert len(retained) == 7
        assert all("2026-08-01" not in backup.name and "2026-08-02" not in backup.name for backup in retained)
