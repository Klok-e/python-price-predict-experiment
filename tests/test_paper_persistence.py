from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from netgrowth.paper_dashboard.domain import LifecycleState
from netgrowth.paper_dashboard.persistence import SQLitePaperStore, StateVersionConflict

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
