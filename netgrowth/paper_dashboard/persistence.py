"""SQLite WAL persistence for the single-writer Paper Account application."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import asdict
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Self
from uuid import uuid4

from netgrowth.accounting import AccountAccounting, AverageCostPosition, PassiveBenchmarks
from netgrowth.simulation import SimulationState

from .domain import AccountEvent, DataStatus, LifecycleState, PaperAccountState, PendingExecution

SCHEMA_VERSION = 1


class StateVersionConflict(RuntimeError):
    """The command was based on a stale browser/application snapshot."""


class SQLitePaperStore(AbstractContextManager["SQLitePaperStore"]):
    """One application-owned SQLite connection and serialized writer."""

    def __init__(self, path: Path | str, *, backup_directory: Path | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_directory = backup_directory or self.path.parent / "backups"
        existed = self.path.exists() and self.path.stat().st_size > 0
        self._connection = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA synchronous = FULL")
        self._connection.execute("PRAGMA busy_timeout = 5000")
        self._connection.execute("PRAGMA journal_mode = WAL")
        current = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if existed and current != SCHEMA_VERSION:
            self._backup_file(
                f"pre-migration-v{current}",
                datetime.now(tz=UTC),
                record_status=False,
            )
        self._migrate(current)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def journal_mode(self) -> str:
        return str(self._connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()

    def close(self) -> None:
        self._connection.close()

    def _migrate(self, current: int) -> None:
        if current > SCHEMA_VERSION:
            raise RuntimeError(f"database schema {current} is newer than supported schema {SCHEMA_VERSION}")
        self._connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS accounts (
                account_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                archived_at TEXT,
                active INTEGER NOT NULL CHECK (active IN (0, 1)),
                state_version INTEGER NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_account
                ON accounts(active) WHERE active = 1;
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL UNIQUE,
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                event_type TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                payload TEXT NOT NULL,
                decision_id TEXT,
                ticker TEXT
            );
            CREATE INDEX IF NOT EXISTS event_account_order
                ON events(account_id, sequence);
            CREATE TABLE IF NOT EXISTS snapshots (
                account_id TEXT PRIMARY KEY REFERENCES accounts(account_id),
                snapshot_version INTEGER NOT NULL,
                event_sequence INTEGER NOT NULL,
                state TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS operating_windows (
                window_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                started_at TEXT NOT NULL,
                ended_at TEXT,
                close_reason TEXT
            );
            CREATE TABLE IF NOT EXISTS idempotency (
                idempotency_key TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS notification_outbox (
                notification_id TEXT PRIMARY KEY,
                account_id TEXT NOT NULL REFERENCES accounts(account_id),
                dedupe_key TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL,
                delivered_at TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS backup_status (
                backup_name TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                error TEXT
            );
            PRAGMA user_version = 1;
            COMMIT;
            """
        )

    def open_active_account(
        self,
        now: datetime,
        tickers: tuple[str, ...],
        *,
        starting_equity: float = 10_000.0,
    ) -> PaperAccountState:
        row = self._connection.execute("SELECT account_id FROM accounts WHERE active = 1").fetchone()
        if row is not None:
            state = self.load_account(str(row["account_id"]))
            if state.tickers != tickers:
                raise ValueError("active Paper Account Trading Universe is incompatible")
            return state

        account_id = str(uuid4())
        state = PaperAccountState.flat_start(
            account_id,
            now,
            tickers,
            starting_equity=starting_equity,
        )
        event_id = str(uuid4())
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            self._connection.execute(
                "INSERT INTO accounts(account_id, created_at, active, state_version) VALUES (?, ?, 1, ?)",
                (account_id, _timestamp(now), state.state_version),
            )
            cursor = self._connection.execute(
                """INSERT INTO events(event_id, account_id, event_type, occurred_at, payload)
                   VALUES (?, ?, 'FlatStart', ?, ?)""",
                (
                    event_id,
                    account_id,
                    _timestamp(now),
                    _json({"starting_equity": starting_equity, "tickers": tickers}),
                ),
            )
            self._connection.execute(
                "INSERT INTO snapshots(account_id, snapshot_version, event_sequence, state) VALUES (?, ?, ?, ?)",
                (account_id, state.snapshot_version, cursor.lastrowid, _json(_state_payload(state))),
            )
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        return state

    def load_active_account(self) -> PaperAccountState | None:
        row = self._connection.execute("SELECT account_id FROM accounts WHERE active = 1").fetchone()
        return self.load_account(str(row["account_id"])) if row is not None else None

    def load_account(self, account_id: str) -> PaperAccountState:
        row = self._connection.execute(
            """SELECT s.state, a.active, a.state_version
               FROM snapshots s JOIN accounts a USING(account_id) WHERE account_id = ?""",
            (account_id,),
        ).fetchone()
        if row is None:
            raise KeyError(account_id)
        payload = json.loads(str(row["state"]))
        payload["active"] = bool(row["active"])
        payload["state_version"] = int(row["state_version"])
        return _state_from_payload(payload)

    def commit(
        self,
        state: PaperAccountState,
        new_events: Iterable[tuple[str, datetime, dict[str, Any], str | None, str | None]],
        *,
        expected_state_version: int,
        idempotency: tuple[str, str, str, dict[str, Any], datetime] | None = None,
        operating_window: tuple[str, datetime, datetime | None] | None = None,
        close_window: tuple[str, datetime, str] | None = None,
        notifications: Iterable[tuple[str, str, dict[str, Any], datetime]] = (),
    ) -> list[AccountEvent]:
        """Atomically append events and the corresponding versioned recovery snapshot."""
        events = list(new_events)
        if not events:
            raise ValueError("an account transition requires at least one durable event")
        if state.state_version != expected_state_version + 1:
            raise ValueError("new state version must advance exactly once")
        inserted: list[AccountEvent] = []
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT state_version FROM accounts WHERE account_id = ?",
                (state.account_id,),
            ).fetchone()
            if row is None:
                raise KeyError(state.account_id)
            if int(row["state_version"]) != expected_state_version:
                raise StateVersionConflict(
                    f"expected state version {expected_state_version}, found {row['state_version']}"
                )
            if operating_window is not None:
                window_id, started_at, detected_end = operating_window
                previous = self._connection.execute(
                    "SELECT window_id, started_at FROM operating_windows WHERE ended_at IS NULL"
                ).fetchone()
                if previous is not None:
                    boundary = detected_end or datetime.fromisoformat(str(previous["started_at"]))
                    self._connection.execute(
                        """UPDATE operating_windows
                           SET ended_at = ?, close_reason = 'detected-restart'
                           WHERE window_id = ?""",
                        (_timestamp(boundary), str(previous["window_id"])),
                    )
                self._connection.execute(
                    "INSERT INTO operating_windows VALUES (?, ?, ?, NULL, NULL)",
                    (window_id, state.account_id, _timestamp(started_at)),
                )
            if close_window is not None:
                window_id, ended_at, reason = close_window
                self._connection.execute(
                    """UPDATE operating_windows SET ended_at = ?, close_reason = ?
                       WHERE window_id = ? AND ended_at IS NULL""",
                    (_timestamp(ended_at), reason, window_id),
                )
            last_sequence = int(
                self._connection.execute(
                    "SELECT event_sequence FROM snapshots WHERE account_id = ?",
                    (state.account_id,),
                ).fetchone()[0]
            )
            for event_type, occurred_at, payload, decision_id, ticker in events:
                event_id = str(uuid4())
                cursor = self._connection.execute(
                    """INSERT INTO events(
                           event_id, account_id, event_type, occurred_at, payload, decision_id, ticker
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        event_id,
                        state.account_id,
                        event_type,
                        _timestamp(occurred_at),
                        _json(payload),
                        decision_id,
                        ticker,
                    ),
                )
                if cursor.lastrowid is None:
                    raise RuntimeError("SQLite did not return the appended event sequence")
                last_sequence = cursor.lastrowid
                inserted.append(
                    AccountEvent(
                        event_id=event_id,
                        sequence=last_sequence,
                        account_id=state.account_id,
                        event_type=event_type,
                        occurred_at=occurred_at,
                        payload=payload,
                        decision_id=decision_id,
                        ticker=ticker,
                    )
                )
            updated = self._connection.execute(
                """UPDATE accounts SET state_version = ?
                   WHERE account_id = ? AND state_version = ?""",
                (state.state_version, state.account_id, expected_state_version),
            )
            if updated.rowcount != 1:
                raise StateVersionConflict("account state changed while committing")
            self._connection.execute(
                """UPDATE snapshots SET snapshot_version = ?, event_sequence = ?, state = ?
                   WHERE account_id = ?""",
                (
                    state.snapshot_version,
                    last_sequence,
                    _json(_state_payload(state)),
                    state.account_id,
                ),
            )
            if idempotency is not None:
                key, action, request_hash, response, created_at = idempotency
                self._connection.execute(
                    "INSERT INTO idempotency VALUES (?, ?, ?, ?, ?)",
                    (key, action, request_hash, _json(response), _timestamp(created_at)),
                )
            for dedupe_key, kind, payload, created_at in notifications:
                self._connection.execute(
                    """INSERT OR IGNORE INTO notification_outbox(
                           notification_id, account_id, dedupe_key, kind, payload, created_at
                       ) VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid4()),
                        state.account_id,
                        dedupe_key,
                        kind,
                        _json(payload),
                        _timestamp(created_at),
                    ),
                )
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        return inserted

    def events(self, account_id: str | None = None) -> list[AccountEvent]:
        query = "SELECT * FROM events"
        arguments: tuple[str, ...] = ()
        if account_id is not None:
            query += " WHERE account_id = ?"
            arguments = (account_id,)
        query += " ORDER BY sequence"
        return [
            AccountEvent(
                event_id=str(row["event_id"]),
                sequence=int(row["sequence"]),
                account_id=str(row["account_id"]),
                event_type=str(row["event_type"]),
                occurred_at=datetime.fromisoformat(str(row["occurred_at"])),
                payload=json.loads(str(row["payload"])),
                decision_id=str(row["decision_id"]) if row["decision_id"] is not None else None,
                ticker=str(row["ticker"]) if row["ticker"] is not None else None,
            )
            for row in self._connection.execute(query, arguments)
        ]

    def accounts(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._connection.execute("SELECT * FROM accounts ORDER BY created_at")]

    def operating_windows(self, account_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._connection.execute(
                "SELECT * FROM operating_windows WHERE account_id = ? ORDER BY started_at",
                (account_id,),
            )
        ]

    def idempotent_response(self, key: str, action: str, request_hash: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT action, request_hash, response FROM idempotency WHERE idempotency_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        if str(row["action"]) != action or str(row["request_hash"]) != request_hash:
            raise ValueError("idempotency key was already used for another command")
        response = json.loads(str(row["response"]))
        if not isinstance(response, dict):
            raise RuntimeError("stored idempotent response is not an object")
        return response

    def record_idempotent_response(
        self,
        key: str,
        action: str,
        request_hash: str,
        response: dict[str, Any],
        now: datetime,
    ) -> None:
        self._connection.execute(
            "INSERT INTO idempotency VALUES (?, ?, ?, ?, ?)",
            (key, action, request_hash, _json(response), _timestamp(now)),
        )

    def enqueue_notification(
        self,
        account_id: str,
        dedupe_key: str,
        kind: str,
        payload: dict[str, Any],
        now: datetime,
    ) -> bool:
        cursor = self._connection.execute(
            """INSERT OR IGNORE INTO notification_outbox(
                   notification_id, account_id, dedupe_key, kind, payload, created_at
               ) VALUES (?, ?, ?, ?, ?, ?)""",
            (str(uuid4()), account_id, dedupe_key, kind, _json(payload), _timestamp(now)),
        )
        return cursor.rowcount == 1

    def pending_notifications(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self._connection.execute(
                """SELECT * FROM notification_outbox
                   WHERE delivered_at IS NULL AND error IS NULL ORDER BY created_at"""
            )
        ]

    def mark_notification(self, notification_id: str, now: datetime, error: str | None) -> None:
        self._connection.execute(
            "UPDATE notification_outbox SET delivered_at = ?, error = ? WHERE notification_id = ?",
            (_timestamp(now) if error is None else None, error, notification_id),
        )

    def notification_health(self) -> dict[str, Any]:
        failed = self._connection.execute(
            """SELECT kind, created_at, error FROM notification_outbox
               WHERE error IS NOT NULL ORDER BY created_at DESC LIMIT 1"""
        ).fetchone()
        pending = int(
            self._connection.execute(
                """SELECT count(*) FROM notification_outbox
                   WHERE delivered_at IS NULL AND error IS NULL"""
            ).fetchone()[0]
        )
        return {
            "healthy": failed is None,
            "error": str(failed["error"]) if failed is not None else None,
            "failed_kind": str(failed["kind"]) if failed is not None else None,
            "failed_at": str(failed["created_at"]) if failed is not None else None,
            "pending": pending,
        }

    def create_daily_backup(self, now: datetime) -> Path | None:
        prefix = f"paper-{now.date().isoformat()}"
        if self.backup_directory.exists() and any(self.backup_directory.glob(f"{prefix}*.sqlite3")):
            return None
        return self._backup_file(prefix, now)

    def record_backup_failure(self, now: datetime, error: str) -> None:
        """Persist the latest failed daily attempt independently of prior successes."""
        self._connection.execute(
            "INSERT OR REPLACE INTO backup_status VALUES (?, ?, ?)",
            (f"paper-{now.date().isoformat()}-failed", _timestamp(now), error),
        )

    def _backup_file(self, label: str, now: datetime, *, record_status: bool = True) -> Path:
        self.backup_directory.mkdir(parents=True, exist_ok=True)
        destination = self.backup_directory / f"{label}-{now.strftime('%H%M%S')}.sqlite3"
        target = sqlite3.connect(destination)
        try:
            self._connection.backup(target)
        finally:
            target.close()
        backups = sorted(self.backup_directory.glob("paper-*.sqlite3"), key=lambda item: item.stat().st_mtime)
        for expired in backups[:-7]:
            expired.unlink()
        if record_status:
            self._connection.execute(
                "INSERT OR REPLACE INTO backup_status VALUES (?, ?, NULL)",
                (destination.name, _timestamp(now)),
            )
        return destination

    def latest_backup(self) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM backup_status ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row is not None else None

    def reset_and_replace(
        self,
        state: PaperAccountState,
        now: datetime,
        *,
        window_id: str,
        window_ended_at: datetime | None = None,
        starting_equity: float = 10_000.0,
        include_intervention: bool = False,
        idempotency: tuple[str, str, str, datetime] | None = None,
    ) -> tuple[PaperAccountState, str]:
        if any(abs(quantity) > 1e-12 for quantity in state.simulation.quantities.values()):
            raise ValueError("Manual Reset cannot archive an account with open positions")
        replacement = PaperAccountState.flat_start(str(uuid4()), now, state.tickers, starting_equity=starting_equity)
        replacement.protocol_id = state.protocol_id
        replacement.model_id = state.model_id
        replacement.model_checkpoint = state.model_checkpoint
        replacement.compatibility_manifest = state.compatibility_manifest.copy()
        replacement.fitting = state.fitting.copy()
        replacement_window = str(uuid4())
        window_boundary = window_ended_at or now
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            current = self._connection.execute(
                "SELECT state_version FROM accounts WHERE account_id = ? AND active = 1",
                (state.account_id,),
            ).fetchone()
            if current is None or int(current["state_version"]) != state.state_version:
                raise StateVersionConflict("active Paper Account changed before Manual Reset completion")
            last_old_sequence: int | None = None
            if include_intervention:
                intervention = self._connection.execute(
                    """INSERT INTO events(event_id, account_id, event_type, occurred_at, payload)
                       VALUES (?, ?, 'OperatorIntervention', ?, ?)""",
                    (
                        str(uuid4()),
                        state.account_id,
                        _timestamp(now),
                        _json({"action": "manual_reset"}),
                    ),
                )
                last_old_sequence = intervention.lastrowid
            window_closed = self._connection.execute(
                """INSERT INTO events(event_id, account_id, event_type, occurred_at, payload)
                   VALUES (?, ?, 'OperatingWindowClosed', ?, ?)""",
                (
                    str(uuid4()),
                    state.account_id,
                    _timestamp(window_boundary),
                    _json(
                        {
                            "window_id": window_id,
                            "ended_at": window_boundary,
                            "reason": "manual-reset",
                        }
                    ),
                ),
            )
            last_old_sequence = window_closed.lastrowid
            archive_event = self._connection.execute(
                """INSERT INTO events(event_id, account_id, event_type, occurred_at, payload)
                   VALUES (?, ?, 'AccountArchived', ?, ?)""",
                (
                    str(uuid4()),
                    state.account_id,
                    _timestamp(now),
                    _json({"reason": "manual_reset", "new_account_id": replacement.account_id}),
                ),
            )
            last_old_sequence = archive_event.lastrowid
            if last_old_sequence is None:
                raise RuntimeError("SQLite did not return the archival event sequence")
            state.active = False
            state.state_version += 1
            state.snapshot_version += 1
            self._connection.execute(
                """UPDATE accounts SET active = 0, archived_at = ?, state_version = ?
                   WHERE account_id = ? AND active = 1""",
                (_timestamp(now), state.state_version, state.account_id),
            )
            self._connection.execute(
                """UPDATE snapshots SET snapshot_version = ?, event_sequence = ?, state = ?
                   WHERE account_id = ?""",
                (
                    state.snapshot_version,
                    last_old_sequence,
                    _json(_state_payload(state)),
                    state.account_id,
                ),
            )
            self._connection.execute(
                """UPDATE operating_windows SET ended_at = ?, close_reason = 'manual-reset'
                   WHERE window_id = ? AND ended_at IS NULL""",
                (_timestamp(window_boundary), window_id),
            )
            self._connection.execute(
                "INSERT INTO accounts(account_id, created_at, active, state_version) VALUES (?, ?, 1, ?)",
                (replacement.account_id, _timestamp(now), replacement.state_version),
            )
            flat_start = self._connection.execute(
                """INSERT INTO events(event_id, account_id, event_type, occurred_at, payload)
                   VALUES (?, ?, 'FlatStart', ?, ?)""",
                (
                    str(uuid4()),
                    replacement.account_id,
                    _timestamp(now),
                    _json({"starting_equity": starting_equity, "tickers": state.tickers}),
                ),
            )
            reset_completed = self._connection.execute(
                """INSERT INTO events(event_id, account_id, event_type, occurred_at, payload)
                   VALUES (?, ?, 'ManualResetCompleted', ?, ?)""",
                (
                    str(uuid4()),
                    replacement.account_id,
                    _timestamp(now),
                    _json(
                        {
                            "archived_account_id": state.account_id,
                            "new_account_id": replacement.account_id,
                        }
                    ),
                ),
            )
            opened = self._connection.execute(
                """INSERT INTO events(event_id, account_id, event_type, occurred_at, payload)
                   VALUES (?, ?, 'OperatingWindowOpened', ?, ?)""",
                (
                    str(uuid4()),
                    replacement.account_id,
                    _timestamp(now),
                    _json({"window_id": replacement_window}),
                ),
            )
            if flat_start.lastrowid is None or reset_completed.lastrowid is None or opened.lastrowid is None:
                raise RuntimeError("SQLite did not return the reset event sequence")
            self._connection.execute(
                "INSERT INTO snapshots VALUES (?, ?, ?, ?)",
                (
                    replacement.account_id,
                    replacement.snapshot_version,
                    opened.lastrowid,
                    _json(_state_payload(replacement)),
                ),
            )
            self._connection.execute(
                "INSERT INTO operating_windows VALUES (?, ?, ?, NULL, NULL)",
                (replacement_window, replacement.account_id, _timestamp(now)),
            )
            notification_payload = {
                "archived_account_id": state.account_id,
                "new_account_id": replacement.account_id,
            }
            self._connection.execute(
                """INSERT OR IGNORE INTO notification_outbox(
                       notification_id, account_id, dedupe_key, kind, payload, created_at
                   ) VALUES (?, ?, ?, 'manual_reset_completed', ?, ?)""",
                (
                    str(uuid4()),
                    replacement.account_id,
                    f"reset:{state.account_id}",
                    _json(notification_payload),
                    _timestamp(now),
                ),
            )
            if idempotency is not None:
                key, action, request_hash, created_at = idempotency
                response = {
                    "accepted": True,
                    "message": "Manual Reset completed",
                    "account": {"id": replacement.account_id, "version": replacement.state_version},
                }
                self._connection.execute(
                    "INSERT INTO idempotency VALUES (?, ?, ?, ?, ?)",
                    (key, action, request_hash, _json(response), _timestamp(created_at)),
                )
            self._connection.execute("COMMIT")
        except BaseException:
            self._connection.execute("ROLLBACK")
            raise
        return replacement, replacement_window


def _state_payload(state: PaperAccountState) -> dict[str, Any]:
    return asdict(state)


def _state_from_payload(raw: dict[str, Any]) -> PaperAccountState:
    raw = dict(raw)
    raw["created_at"] = datetime.fromisoformat(raw["created_at"])
    raw["tickers"] = tuple(raw["tickers"])
    raw["lifecycle"] = LifecycleState(raw["lifecycle"])
    raw["data_status"] = DataStatus(raw["data_status"])
    for name in ("last_observation_at", "last_decision_at", "stale_since"):
        if raw.get(name) is not None:
            raw[name] = datetime.fromisoformat(raw[name])
    raw["simulation"] = SimulationState.from_payload(raw["simulation"])
    accounting = raw["accounting"]
    accounting["positions"] = {
        ticker: AverageCostPosition(**position) for ticker, position in accounting["positions"].items()
    }
    raw["accounting"] = AccountAccounting(**accounting)
    if raw.get("benchmarks") is not None:
        raw["benchmarks"] = PassiveBenchmarks(**raw["benchmarks"])
    pending = raw.get("pending_execution")
    if pending is not None:
        for name in ("signal_time", "eligible_at", "expires_at"):
            pending[name] = datetime.fromisoformat(pending[name])
        raw["pending_execution"] = PendingExecution(**pending)
    return PaperAccountState(**raw)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("database timestamps must be timezone-aware UTC")
    return value.astimezone(UTC).isoformat()


def _json(value: Any) -> str:
    def default(item: Any) -> Any:
        if isinstance(item, datetime):
            return _timestamp(item)
        if isinstance(item, date):
            return item.isoformat()
        if isinstance(item, Enum):
            return item.value
        if isinstance(item, tuple):
            return list(item)
        raise TypeError(f"cannot serialize {type(item)!r}")

    return json.dumps(value, default=default, separators=(",", ":"), sort_keys=True)
