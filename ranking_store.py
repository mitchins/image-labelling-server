"""SQLite persistence for the ranking MVP.

The module deliberately has no web-framework dependencies.  A database contains one
immutable ranking task, its supplied sets and candidates, and append-only revisions
for each set.  Revision zero is the implicit pending state and is never stored as a
row in ``ranking_revisions``.
"""

from __future__ import annotations

import json
import random
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union


class RankingStoreError(Exception):
    """Base class for errors raised by this repository."""


class RankingBadRequestError(RankingStoreError):
    """The request or supplied task data is malformed (HTTP 400)."""

    status_code = 400


class RankingNotFoundError(RankingStoreError):
    """The requested task or set does not exist (HTTP 404)."""

    status_code = 404


class RankingConflictError(RankingStoreError):
    """The operation conflicts with existing state (HTTP 409)."""

    status_code = 409


# Short aliases make mapping the domain errors in an API convenient.
InvalidRankingRequest = RankingBadRequestError
RankingValidationError = RankingBadRequestError
RankingNotFound = RankingNotFoundError
RankingConflict = RankingConflictError

Database = Union[str, Path, sqlite3.Connection, "RankingStore"]


SCHEMA = """
CREATE TABLE IF NOT EXISTS ranking_tasks (
    task_id INTEGER PRIMARY KEY CHECK (task_id = 1),
    mode TEXT NOT NULL CHECK (mode = 'ranking'),
    criterion_json TEXT NOT NULL,
    task_settings_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ranking_sets (
    set_id TEXT PRIMARY KEY,
    source_position INTEGER NOT NULL UNIQUE,
    display_position INTEGER NOT NULL UNIQUE,
    set_metadata_json TEXT NOT NULL DEFAULT '{}',
    current_revision INTEGER NOT NULL DEFAULT 0,
    current_outcome TEXT NOT NULL DEFAULT 'pending'
        CHECK (current_outcome IN ('pending', 'ranked', 'invalid')),
    current_order_json TEXT NOT NULL DEFAULT '[]',
    current_invalid_reason TEXT,
    current_session_id TEXT,
    current_request_id TEXT,
    current_timestamp TEXT
);

CREATE TABLE IF NOT EXISTS ranking_candidates (
    set_id TEXT NOT NULL REFERENCES ranking_sets(set_id),
    candidate_id TEXT NOT NULL,
    source_position INTEGER NOT NULL,
    display_position INTEGER NOT NULL,
    path TEXT NOT NULL,
    media_type TEXT,
    metadata_json TEXT NOT NULL,
    PRIMARY KEY (set_id, candidate_id),
    UNIQUE (set_id, source_position),
    UNIQUE (set_id, display_position)
);

CREATE TABLE IF NOT EXISTS ranking_revisions (
    set_id TEXT NOT NULL REFERENCES ranking_sets(set_id),
    revision INTEGER NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('submit', 'undo')),
    outcome TEXT NOT NULL CHECK (outcome IN ('pending', 'ranked', 'invalid')),
    ordered_candidate_ids_json TEXT NOT NULL,
    expected_revision INTEGER NOT NULL DEFAULT 0,
    invalid_reason TEXT,
    session_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    undo_of_revision INTEGER,
    request_payload_json TEXT,
    PRIMARY KEY (set_id, revision),
    UNIQUE (request_id),
    FOREIGN KEY (set_id, undo_of_revision)
        REFERENCES ranking_revisions(set_id, revision)
);

CREATE INDEX IF NOT EXISTS ranking_sets_pending_idx
    ON ranking_sets(current_outcome, display_position);
CREATE INDEX IF NOT EXISTS ranking_revisions_session_idx
    ON ranking_revisions(session_id, revision);
"""


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise RankingBadRequestError("metadata and settings must be JSON-serializable") from exc


def _read_json(raw: str) -> Any:
    return json.loads(raw)


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RankingBadRequestError(f"{name} must be a non-empty string")
    return value


def _position(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RankingBadRequestError(f"{name} must be a non-negative integer")
    return value


def _criterion(criterion: Mapping[str, Any]) -> Dict[str, str]:
    if not isinstance(criterion, Mapping):
        raise RankingBadRequestError("criterion must be an object")
    if set(criterion) != {"id", "version", "prompt", "direction"}:
        raise RankingBadRequestError(
            "criterion must contain exactly id, version, prompt, and direction"
        )
    result = {}
    for key in ("id", "version", "prompt"):
        result[key] = _text(criterion.get(key), f"criterion.{key}")
    if criterion.get("direction") != "most":
        raise RankingBadRequestError("criterion.direction must be 'most'")
    result["direction"] = "most"
    return result


def _database_value(database: Database) -> Union[str, Path, sqlite3.Connection]:
    if isinstance(database, RankingStore):
        return database.database
    return database


@contextmanager
def _connection(database: Database) -> Iterator[sqlite3.Connection]:
    value = _database_value(database)
    if isinstance(value, sqlite3.Connection):
        connection = value
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        yield connection
        return

    connection = sqlite3.connect(str(value), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
    finally:
        connection.close()


@contextmanager
def _write_transaction(connection: sqlite3.Connection) -> Iterator[None]:
    savepoint = "ranking_store_write"
    nested = connection.in_transaction
    if nested:
        connection.execute(f"SAVEPOINT {savepoint}")
    else:
        connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        if nested:
            connection.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        else:
            connection.rollback()
        raise
    else:
        if nested:
            connection.execute(f"RELEASE SAVEPOINT {savepoint}")
        else:
            connection.commit()


def initialize_schema(database: Database) -> None:
    """Create the ranking tables without creating a task."""
    with _connection(database) as connection:
        _ensure_schema(connection)
        connection.commit()


def _table_columns(connection: sqlite3.Connection, table: str) -> set:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}


_CURRENT_SCHEMA_COLUMNS = {
    "ranking_tasks": {
        "task_id", "mode", "criterion_json", "task_settings_json", "created_at",
    },
    "ranking_sets": {
        "set_id", "source_position", "display_position", "set_metadata_json",
        "current_revision", "current_outcome", "current_order_json", "current_invalid_reason",
        "current_session_id", "current_request_id", "current_timestamp",
    },
    "ranking_candidates": {
        "set_id", "candidate_id", "source_position", "display_position", "path", "media_type",
        "metadata_json",
    },
    "ranking_revisions": {
        "set_id", "revision", "action", "outcome", "ordered_candidate_ids_json",
        "expected_revision", "invalid_reason", "session_id", "request_id", "timestamp",
        "undo_of_revision", "request_payload_json",
    },
}


def _has_current_schema(connection: sqlite3.Connection) -> bool:
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    return all(
        table in tables and required <= _table_columns(connection, table)
        for table, required in _CURRENT_SCHEMA_COLUMNS.items()
    )


def _ensure_schema(connection: sqlite3.Connection) -> None:
    """Create or migrate the schema while serializing first-use upgrades."""
    # Reads of a complete native schema must not compete with an active writer
    # for SQLite's RESERVED lock.  An incomplete schema still takes the lock so
    # the migration below remains serialized across processes.
    if _has_current_schema(connection):
        return
    owns_transaction = not connection.in_transaction
    if owns_transaction:
        connection.execute("BEGIN IMMEDIATE")
    try:
        _ensure_schema_locked(connection)
    except Exception:
        if owns_transaction:
            connection.rollback()
        raise
    else:
        if owns_transaction:
            connection.commit()


def _ensure_schema_locked(connection: sqlite3.Connection) -> None:
    """Create the native schema, or upgrade the adjacent ingest schema in place."""
    tables = {
        row[0]
        for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if "ranking_sets" not in tables:
        connection.executescript(SCHEMA)
        return
    if "ranking_tasks" in tables:
        if "set_metadata_json" not in _table_columns(connection, "ranking_sets"):
            connection.execute(
                "ALTER TABLE ranking_sets ADD COLUMN set_metadata_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "expected_revision" not in _table_columns(connection, "ranking_revisions"):
            connection.execute(
                "ALTER TABLE ranking_revisions ADD COLUMN expected_revision INTEGER NOT NULL DEFAULT 0"
            )
        return

    # ingest_ranking.py predates this repository and stores criterion/settings in
    # different columns.  Normalize that database once so both entry points share
    # the same append-only behavior.
    set_columns = _table_columns(connection, "ranking_sets")
    if "criterion_json" not in set_columns:
        # This is a native schema created by initialize_schema(), waiting for its
        # immutable task snapshot to be inserted.
        connection.executescript(SCHEMA)
        return
    required_set_columns = {
        "source_position": "INTEGER",
        "display_position": "INTEGER",
        "set_metadata_json": "TEXT NOT NULL DEFAULT '{}'",
        "current_revision": "INTEGER NOT NULL DEFAULT 0",
        "current_outcome": "TEXT NOT NULL DEFAULT 'pending'",
        "current_order_json": "TEXT NOT NULL DEFAULT '[]'",
        "current_invalid_reason": "TEXT",
        "current_session_id": "TEXT",
        "current_request_id": "TEXT",
        "current_timestamp": "TEXT",
    }
    for column, definition in required_set_columns.items():
        if column not in set_columns:
            connection.execute(f"ALTER TABLE ranking_sets ADD COLUMN {column} {definition}")
    connection.execute(
        "UPDATE ranking_sets SET source_position = rowid WHERE source_position IS NULL"
    )
    connection.execute(
        "UPDATE ranking_sets SET display_position = source_position WHERE display_position IS NULL"
    )
    if "metadata_json" in set_columns:
        connection.execute(
            "UPDATE ranking_sets SET set_metadata_json = metadata_json "
            "WHERE set_metadata_json = '{}' AND metadata_json IS NOT NULL"
        )

    revision_columns = _table_columns(connection, "ranking_revisions")
    required_revision_columns = {
        "action": "TEXT NOT NULL DEFAULT 'submit'",
        "outcome": "TEXT NOT NULL DEFAULT 'ranked'",
        "ordered_candidate_ids_json": "TEXT NOT NULL DEFAULT '[]'",
        "expected_revision": "INTEGER NOT NULL DEFAULT 0",
        "session_id": "TEXT NOT NULL DEFAULT 'legacy'",
        # SQLite only permits constant defaults when adding a column to a
        # non-empty table. Backfill this nullable column below instead.
        "timestamp": "TEXT",
        "undo_of_revision": "INTEGER",
        "request_payload_json": "TEXT",
    }
    for column, definition in required_revision_columns.items():
        if column not in revision_columns:
            connection.execute(f"ALTER TABLE ranking_revisions ADD COLUMN {column} {definition}")
    connection.execute(
        "UPDATE ranking_revisions SET outcome = CASE WHEN is_invalid = 1 THEN 'invalid' ELSE 'ranked' END"
    )
    connection.execute(
        "UPDATE ranking_revisions SET ordered_candidate_ids_json = "
        "CASE WHEN is_invalid = 1 OR ranking_json IS NULL THEN '[]' ELSE ranking_json END"
    )
    for legacy_revision in connection.execute(
        "SELECT rowid AS legacy_rowid, timestamp, created_at FROM ranking_revisions"
    ).fetchall():
        timestamp = _canonical_timestamp(legacy_revision["timestamp"] or legacy_revision["created_at"])
        connection.execute(
            "UPDATE ranking_revisions SET timestamp = ? WHERE rowid = ?",
            (timestamp, legacy_revision["legacy_rowid"]),
        )

    settings = {}
    if "settings" in tables:
        for key, value in connection.execute("SELECT key, value FROM settings"):
            try:
                settings[key] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                settings[key] = value
    first_set = connection.execute(
        "SELECT criterion_json FROM ranking_sets ORDER BY source_position LIMIT 1"
    ).fetchone()
    if first_set is None:
        raise RankingNotFoundError("ranking database has no sets")
    criterion = _read_json(first_set[0])
    connection.execute(
        """
        CREATE TABLE ranking_tasks (
            task_id INTEGER PRIMARY KEY CHECK (task_id = 1),
            mode TEXT NOT NULL CHECK (mode = 'ranking'),
            criterion_json TEXT NOT NULL,
            task_settings_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    connection.execute(
        "INSERT INTO ranking_tasks VALUES (1, 'ranking', ?, ?, ?)",
        (_json(criterion), _json(settings), _utc_now()),
    )
    connection.execute(
        """
        UPDATE ranking_sets
        SET current_revision = COALESCE((SELECT MAX(revision) FROM ranking_revisions r
                                         WHERE r.set_id = ranking_sets.set_id), 0)
        """
    )
    for row in connection.execute("SELECT set_id, current_revision FROM ranking_sets").fetchall():
        if row["current_revision"] == 0:
            continue
        latest = connection.execute(
            "SELECT * FROM ranking_revisions WHERE set_id = ? AND revision = ?",
            (row["set_id"], row["current_revision"]),
        ).fetchone()
        connection.execute(
            """
            UPDATE ranking_sets
            SET current_outcome = ?, current_order_json = ?, current_invalid_reason = ?,
                current_session_id = ?, current_request_id = ?, current_timestamp = ?
            WHERE set_id = ?
            """,
            (
                latest["outcome"], latest["ordered_candidate_ids_json"], latest["invalid_reason"],
                latest["session_id"], latest["request_id"], latest["timestamp"], row["set_id"],
            ),
        )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ranking_sets_source_position_idx "
        "ON ranking_sets(source_position)"
    )
    connection.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ranking_sets_display_position_idx "
        "ON ranking_sets(display_position)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS ranking_revisions_session_idx "
        "ON ranking_revisions(session_id, revision)"
    )


def _validate_candidate(candidate: Mapping[str, Any], index: int) -> Dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise RankingBadRequestError(f"candidate {index} must be an object")
    candidate_id = _text(candidate.get("candidate_id"), "candidate_id")
    path = _text(candidate.get("path"), "path")
    media_type = candidate.get("media_type")
    if media_type is not None:
        media_type = _text(media_type, "media_type")
    metadata = candidate.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise RankingBadRequestError("candidate.metadata must be an object")
    source_position = candidate.get("source_position", index + 1)
    return {
        "candidate_id": candidate_id,
        "source_position": _position(source_position, "candidate.source_position"),
        "path": path,
        "media_type": media_type,
        "metadata": dict(metadata),
    }


def _validate_sets(supplied_sets: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(supplied_sets, Sequence) or isinstance(supplied_sets, (str, bytes)):
        raise RankingBadRequestError("sets must be a list")
    if not supplied_sets:
        raise RankingBadRequestError("at least one set is required")

    seen_sets = set()
    seen_source_positions = set()
    result = []
    for set_index, supplied in enumerate(supplied_sets):
        if not isinstance(supplied, Mapping):
            raise RankingBadRequestError(f"set {set_index} must be an object")
        set_id = _text(supplied.get("set_id"), "set_id")
        if set_id in seen_sets:
            raise RankingBadRequestError(f"duplicate set_id: {set_id}")
        seen_sets.add(set_id)
        source_position = supplied.get("source_position", set_index + 1)
        source_position = _position(source_position, "set.source_position")
        if source_position in seen_source_positions:
            raise RankingBadRequestError(f"duplicate set source_position: {source_position}")
        seen_source_positions.add(source_position)

        candidates = supplied.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            raise RankingBadRequestError(f"set {set_id} candidates must be a list")
        if not 2 <= len(candidates) <= 8:
            raise RankingBadRequestError(f"set {set_id} must contain 2 to 8 candidates")
        validated_candidates = [_validate_candidate(candidate, i) for i, candidate in enumerate(candidates)]
        candidate_ids = [candidate["candidate_id"] for candidate in validated_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise RankingBadRequestError(f"set {set_id} has duplicate candidate_id values")
        candidate_positions = [candidate["source_position"] for candidate in validated_candidates]
        if len(candidate_positions) != len(set(candidate_positions)):
            raise RankingBadRequestError(f"set {set_id} has duplicate candidate source positions")
        set_metadata = supplied.get("metadata", {})
        if not isinstance(set_metadata, Mapping):
            raise RankingBadRequestError(f"set {set_id} metadata must be an object")
        result.append({
            "set_id": set_id,
            "source_position": source_position,
            "metadata": dict(set_metadata),
            "candidates": validated_candidates,
        })
    return result


def initialize_ranking_store(
    database: Database,
    criterion: Mapping[str, Any],
    sets: Sequence[Mapping[str, Any]],
    task_settings: Optional[Mapping[str, Any]] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Create the immutable task snapshot and randomized display positions.

    ``random_seed`` is optional and exists for reproducible imports/tests.  The
    resulting positions are persisted, so reads never randomize again.
    """
    normalized_criterion = _criterion(criterion)
    normalized_sets = _validate_sets(sets)
    if task_settings is None:
        normalized_settings: Dict[str, Any] = {}
    elif isinstance(task_settings, Mapping):
        normalized_settings = dict(task_settings)
    else:
        raise RankingBadRequestError("task_settings must be an object")
    # Validate JSON now, before any transaction can partially create a task.
    criterion_json = _json(normalized_criterion)
    settings_json = _json(normalized_settings)
    rng = random.Random(random_seed) if random_seed is not None else random.SystemRandom()
    set_order = list(range(len(normalized_sets)))
    rng.shuffle(set_order)
    display_positions = {set_index: position for position, set_index in enumerate(set_order, start=1)}

    with _connection(database) as connection:
        _ensure_schema(connection)
        connection.commit()
        with _write_transaction(connection):
            if connection.execute("SELECT 1 FROM ranking_tasks WHERE task_id = 1").fetchone():
                raise RankingConflictError("ranking task already exists")
            connection.execute(
                "INSERT INTO ranking_tasks VALUES (1, 'ranking', ?, ?, ?)",
                (criterion_json, settings_json, _utc_now()),
            )
            for set_index, supplied in enumerate(normalized_sets):
                connection.execute(
                    """
                    INSERT INTO ranking_sets
                        (set_id, source_position, display_position, set_metadata_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        supplied["set_id"], supplied["source_position"], display_positions[set_index],
                        _json(supplied["metadata"]),
                    ),
                )
                candidate_order = list(range(len(supplied["candidates"])))
                rng.shuffle(candidate_order)
                candidate_display_positions = {
                    candidate_index: position
                    for position, candidate_index in enumerate(candidate_order, start=1)
                }
                for candidate_index, candidate in enumerate(supplied["candidates"]):
                    connection.execute(
                        """
                        INSERT INTO ranking_candidates
                            (set_id, candidate_id, source_position, display_position, path, media_type, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            supplied["set_id"], candidate["candidate_id"], candidate["source_position"],
                            candidate_display_positions[candidate_index], candidate["path"],
                            candidate["media_type"], _json(candidate["metadata"]),
                        ),
                    )
    return {"mode": "ranking", "criterion": normalized_criterion, "task_settings": normalized_settings}


# Public aliases use the same immutable-task creation behavior.
create_ranking_task = initialize_ranking_store
create_task = initialize_ranking_store


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_timestamp(value: Any) -> str:
    """Normalize legacy timestamps, whose naive values are SQLite UTC timestamps."""
    if not isinstance(value, str) or not value.strip():
        return _utc_now()
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        try:
            parsed = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return _utc_now()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _task_or_404(connection: sqlite3.Connection) -> sqlite3.Row:
    was_in_transaction = connection.in_transaction
    _ensure_schema(connection)
    if not was_in_transaction and connection.in_transaction:
        connection.commit()
    row = connection.execute("SELECT * FROM ranking_tasks WHERE task_id = 1").fetchone()
    if row is None:
        raise RankingNotFoundError("ranking task does not exist")
    return row


def _set_or_404(connection: sqlite3.Connection, set_id: str) -> sqlite3.Row:
    row = connection.execute("SELECT * FROM ranking_sets WHERE set_id = ?", (set_id,)).fetchone()
    if row is None:
        raise RankingNotFoundError(f"unknown set_id: {set_id}")
    return row


def _candidates(connection: sqlite3.Connection, set_id: str) -> List[sqlite3.Row]:
    return connection.execute(
        "SELECT * FROM ranking_candidates WHERE set_id = ? ORDER BY display_position", (set_id,)
    ).fetchall()


def _state_from_set(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "revision": row["current_revision"],
        "outcome": row["current_outcome"],
        "ordered_candidate_ids": _read_json(row["current_order_json"]),
        "invalid_reason": row["current_invalid_reason"],
        "session_id": row["current_session_id"],
        "request_id": row["current_request_id"],
        "timestamp": row["current_timestamp"],
    }


def _set_payload(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    candidates: Optional[Sequence[sqlite3.Row]] = None,
) -> Dict[str, Any]:
    if candidates is None:
        candidates = _candidates(connection, row["set_id"])
    candidate_payloads = []
    state = _state_from_set(row)
    rank_positions = {candidate_id: position for position, candidate_id in enumerate(
        state["ordered_candidate_ids"], start=1
    )}
    for candidate in candidates:
        item = {
            "candidate_id": candidate["candidate_id"],
            "path": candidate["path"],
            "media_type": candidate["media_type"],
            "metadata": _read_json(candidate["metadata_json"]),
            "source_position": candidate["source_position"],
            "display_position": candidate["display_position"],
            "rank_position": rank_positions.get(candidate["candidate_id"]),
        }
        candidate_payloads.append(item)
    result = {
        "set_id": row["set_id"],
        "source_position": row["source_position"],
        "display_position": row["display_position"],
        "metadata": _read_json(row["set_metadata_json"]),
        "candidates": candidate_payloads,
        "current": state,
    }
    # Keep the current state flat as well; this is convenient for API adapters.
    result.update(state)
    return result


def get_set(database: Database, set_id: str) -> Dict[str, Any]:
    """Return one set, candidates in persisted display order, and current state."""
    set_id = _text(set_id, "set_id")
    with _connection(database) as connection:
        _task_or_404(connection)
        return _set_payload(connection, _set_or_404(connection, set_id))


def get_next_set(database: Database, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return the first pending set in persisted display order, or ``None``."""
    if session_id is not None:
        _text(session_id, "session_id")
    with _connection(database) as connection:
        _task_or_404(connection)
        row = connection.execute(
            """
            SELECT * FROM ranking_sets
            WHERE current_outcome = 'pending'
            ORDER BY display_position
            LIMIT 1
            """
        ).fetchone()
        return None if row is None else _set_payload(connection, row)


def get_stats(database: Database) -> Dict[str, int]:
    """Return set-level progress; candidates are never counted as work items."""
    with _connection(database) as connection:
        _task_or_404(connection)
        counts = {
            row["current_outcome"]: row["count"]
            for row in connection.execute(
                "SELECT current_outcome, COUNT(*) AS count FROM ranking_sets GROUP BY current_outcome"
            )
        }
    total = sum(counts.values())
    ranked = counts.get("ranked", 0)
    invalid = counts.get("invalid", 0)
    pending = counts.get("pending", 0)
    return {
        "total_sets": total,
        "ranked_sets": ranked,
        "invalid_sets": invalid,
        "pending_sets": pending,
        "completed_sets": ranked + invalid,
        "remaining_sets": pending,
        "total": total,
        "labeled": ranked + invalid,
        "remaining": pending,
        "percent": round(100 * (ranked + invalid) / total, 1) if total else 0,
    }


def _revision_payload(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "set_id": row["set_id"],
        "revision": row["revision"],
        "action": row["action"],
        "outcome": row["outcome"],
        "ordered_candidate_ids": _read_json(row["ordered_candidate_ids_json"]),
        "expected_revision": row["expected_revision"],
        "invalid_reason": row["invalid_reason"],
        "session_id": row["session_id"],
        "request_id": row["request_id"],
        "timestamp": row["timestamp"],
        "undo_of_revision": row["undo_of_revision"],
    }


def _validate_pagination(limit: Optional[int], offset: int) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise RankingBadRequestError("offset must be a non-negative integer")
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise RankingBadRequestError("limit must be a non-negative integer")


def _joined_revision_payload(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "set_id": row["set_id"],
        "revision": row["history_revision"],
        "action": row["history_action"],
        "outcome": row["history_outcome"],
        "ordered_candidate_ids": _read_json(row["history_ordered_candidate_ids_json"]),
        "expected_revision": row["history_expected_revision"],
        "invalid_reason": row["history_invalid_reason"],
        "session_id": row["history_session_id"],
        "request_id": row["history_request_id"],
        "timestamp": row["history_timestamp"],
        "undo_of_revision": row["history_undo_of_revision"],
    }


def get_history(
    database: Database,
    set_id: Optional[str] = None,
    session_id: Optional[str] = None,
    limit: Optional[int] = None,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Return append-only revisions, newest first unless a limit is supplied."""
    if set_id is not None:
        _text(set_id, "set_id")
    if session_id is not None:
        _text(session_id, "session_id")
    _validate_pagination(limit, offset)
    with _connection(database) as connection:
        _task_or_404(connection)
        if set_id is not None:
            _set_or_404(connection, set_id)
        clauses = []
        values: List[Any] = []
        if set_id is not None:
            clauses.append("set_id = ?")
            values.append(set_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            values.append(session_id)
        query = "SELECT * FROM ranking_revisions"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY revision DESC"
        if set_id is None:
            query = query.replace("ORDER BY revision DESC", "ORDER BY timestamp DESC, set_id, revision DESC")
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            values.extend([limit, offset])
        elif offset:
            query += " LIMIT -1 OFFSET ?"
            values.append(offset)
        return [_revision_payload(row) for row in connection.execute(query, values).fetchall()]


def get_latest_completed_sets(
    database: Database,
    limit: Optional[int] = None,
    offset: int = 0,
) -> Dict[str, Any]:
    """Return the newest completed set state page and its total count.

    The page is selected in SQLite and candidates for all returned sets are
    hydrated with one query, matching the payload assembled by the ranking API
    without loading every revision or issuing one candidate query per set.
    """
    _validate_pagination(limit, offset)
    with _connection(database) as connection:
        _task_or_404(connection)
        total = connection.execute(
            "SELECT COUNT(*) FROM ranking_sets WHERE current_outcome IN ('ranked', 'invalid')"
        ).fetchone()[0]
        query = """
            SELECT s.*,
                   r.revision AS history_revision,
                   r.action AS history_action,
                   r.outcome AS history_outcome,
                   r.ordered_candidate_ids_json AS history_ordered_candidate_ids_json,
                   r.expected_revision AS history_expected_revision,
                   r.invalid_reason AS history_invalid_reason,
                   r.session_id AS history_session_id,
                   r.request_id AS history_request_id,
                   r.timestamp AS history_timestamp,
                   r.undo_of_revision AS history_undo_of_revision
            FROM ranking_sets AS s
            JOIN ranking_revisions AS r
              ON r.set_id = s.set_id AND r.revision = s.current_revision
            WHERE s.current_outcome IN ('ranked', 'invalid')
            ORDER BY s.current_timestamp DESC, s.set_id, s.current_revision DESC
        """
        values: List[Any] = []
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            values.extend([limit, offset])
        elif offset:
            query += " LIMIT -1 OFFSET ?"
            values.append(offset)
        page_rows = connection.execute(query, values).fetchall()

        candidates_by_set: Dict[str, List[sqlite3.Row]] = {
            row["set_id"]: [] for row in page_rows
        }
        if candidates_by_set:
            set_ids = list(candidates_by_set)
            placeholders = ", ".join("?" for _ in set_ids)
            for candidate in connection.execute(
                "SELECT * FROM ranking_candidates "
                f"WHERE set_id IN ({placeholders}) ORDER BY set_id, display_position",
                set_ids,
            ):
                candidates_by_set[candidate["set_id"]].append(candidate)

        items = []
        for row in page_rows:
            item = _set_payload(connection, row, candidates_by_set[row["set_id"]])
            item.update(_joined_revision_payload(row))
            items.append(item)
        return {"items": items, "total": total}


def _normalize_submission(request: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(request, Mapping):
        raise RankingBadRequestError("submission must be an object")
    required = ("set_id", "request_id", "expected_revision", "session_id", "outcome")
    for key in required:
        if key not in request:
            raise RankingBadRequestError(f"missing request field: {key}")
    normalized = {
        "set_id": _text(request["set_id"], "set_id"),
        "request_id": _text(request["request_id"], "request_id"),
        "session_id": _text(request["session_id"], "session_id"),
        "outcome": request["outcome"],
    }
    expected_revision = request["expected_revision"]
    if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
        raise RankingBadRequestError("expected_revision must be a non-negative integer")
    normalized["expected_revision"] = expected_revision
    if normalized["outcome"] not in ("ranked", "invalid"):
        raise RankingBadRequestError("outcome must be 'ranked' or 'invalid'")
    ordered = request.get("ordered_candidate_ids", [])
    if ordered is None:
        ordered = []
    if not isinstance(ordered, list) or any(not isinstance(value, str) for value in ordered):
        raise RankingBadRequestError("ordered_candidate_ids must be a list of strings")
    normalized["ordered_candidate_ids"] = list(ordered)
    invalid_reason = request.get("invalid_reason")
    if invalid_reason is not None and not isinstance(invalid_reason, str):
        raise RankingBadRequestError("invalid_reason must be a string")
    normalized["invalid_reason"] = invalid_reason
    if normalized["outcome"] == "ranked":
        if invalid_reason is not None:
            raise RankingBadRequestError("invalid_reason is only valid for invalid outcomes")
    else:
        if ordered:
            raise RankingBadRequestError("ordered_candidate_ids is only valid for ranked outcomes")
    return normalized


def _result_from_revision(row: sqlite3.Row, idempotent: bool = False) -> Dict[str, Any]:
    result = _revision_payload(row)
    return result


def submit_revision(
    database: Database, request: Optional[Mapping[str, Any]] = None, **kwargs: Any
) -> Dict[str, Any]:
    """Validate and append a revision using request-id idempotency and CAS.

    ``request`` may be passed as keyword fields as a convenience, e.g.
    ``submit_revision(db, set_id='s', request_id='r', ...)``.
    """
    if request is None:
        request = kwargs
    elif kwargs:
        if not isinstance(request, Mapping):
            raise RankingBadRequestError("submission must be an object")
        merged = dict(request)
        merged.update(kwargs)
        request = merged
    normalized = _normalize_submission(request)
    with _connection(database) as connection:
        _task_or_404(connection)
        with _write_transaction(connection):
            existing = connection.execute(
                "SELECT * FROM ranking_revisions WHERE request_id = ?", (normalized["request_id"],)
            ).fetchone()
            payload_json = _json(normalized)
            if existing is not None:
                if existing["request_payload_json"] != payload_json:
                    raise RankingConflictError("request_id was already used with a different payload")
                return _result_from_revision(existing)

            set_row = _set_or_404(connection, normalized["set_id"])
            candidate_rows = _candidates(connection, normalized["set_id"])
            candidate_ids = {row["candidate_id"] for row in candidate_rows}
            ordered = normalized["ordered_candidate_ids"]
            if normalized["outcome"] == "ranked":
                if len(ordered) != len(candidate_rows):
                    raise RankingBadRequestError("ranked outcome must include every candidate exactly once")
                if len(set(ordered)) != len(ordered):
                    raise RankingBadRequestError("ranked outcome cannot contain duplicate candidates")
                if set(ordered) != candidate_ids:
                    raise RankingBadRequestError("ranked outcome contains unknown or missing candidates")
            if normalized["expected_revision"] != set_row["current_revision"]:
                raise RankingConflictError(
                    f"stale revision: expected {normalized['expected_revision']}, "
                    f"current is {set_row['current_revision']}"
                )

            timestamp = _utc_now()
            revision = set_row["current_revision"] + 1
            connection.execute(
                """
                INSERT INTO ranking_revisions
                    (set_id, revision, action, outcome, ordered_candidate_ids_json, expected_revision,
                     invalid_reason, session_id, request_id, timestamp, request_payload_json)
                VALUES (?, ?, 'submit', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized["set_id"], revision, normalized["outcome"], _json(ordered),
                    normalized["expected_revision"],
                    normalized["invalid_reason"], normalized["session_id"], normalized["request_id"],
                    timestamp, payload_json,
                ),
            )
            updated = connection.execute(
                """
                UPDATE ranking_sets
                SET current_revision = ?, current_outcome = ?, current_order_json = ?,
                    current_invalid_reason = ?, current_session_id = ?,
                    current_request_id = ?, current_timestamp = ?
                WHERE set_id = ? AND current_revision = ?
                """,
                (
                    revision, normalized["outcome"], _json(ordered), normalized["invalid_reason"],
                    normalized["session_id"], normalized["request_id"], timestamp,
                    normalized["set_id"], normalized["expected_revision"],
                ),
            )
            if updated.rowcount != 1:
                raise RankingConflictError("set changed while submitting revision")
            row = connection.execute(
                "SELECT * FROM ranking_revisions WHERE set_id = ? AND revision = ?",
                (normalized["set_id"], revision),
            ).fetchone()
            return _result_from_revision(row)


def undo_last_for_session(
    database: Database,
    session_id: str,
    expected_revision: Optional[int] = None,
    request_id: Optional[str] = None,
    expected_set_id: Optional[str] = None,
    target_revision: Optional[int] = None,
    target_set_id: Optional[str] = None,
    set_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Append an idempotent undo for an explicitly CAS-selected submission.

    Explicit requests provide ``request_id``, ``expected_set_id`` (or its aliases),
    and ``expected_revision``. The target must be that set's current submission;
    an undo revision is never selected as a substitute. The old session-only form
    remains for the existing API adapter.
    """
    session_id = _text(session_id, "session_id")
    if expected_revision is not None:
        expected_revision = _position(expected_revision, "expected_revision")
    if target_revision is not None:
        target_revision = _position(target_revision, "target_revision")
    if expected_set_id is not None:
        expected_set_id = _text(expected_set_id, "expected_set_id")
    if target_set_id is not None:
        target_set_id = _text(target_set_id, "target_set_id")
        if expected_set_id is not None and target_set_id != expected_set_id:
            raise RankingBadRequestError("expected_set_id and target_set_id must match")
        expected_set_id = target_set_id
    if set_id is not None:
        set_id = _text(set_id, "set_id")
        if expected_set_id is not None and set_id != expected_set_id:
            raise RankingBadRequestError("expected_set_id and set_id must match")
        expected_set_id = set_id

    explicit_request = request_id is not None
    if explicit_request:
        request_id = _text(request_id, "request_id")
        if expected_set_id is None:
            raise RankingBadRequestError("expected_set_id is required for an explicit undo")
        if expected_revision is None:
            raise RankingBadRequestError("expected_revision is required for an explicit undo")
    else:
        request_id = f"undo:{session_id}:{uuid.uuid4()}"
    undo_payload = _json({
        "session_id": session_id,
        "request_id": request_id,
        "expected_set_id": expected_set_id,
        "expected_revision": expected_revision,
        "target_revision": target_revision,
    })
    with _connection(database) as connection:
        _task_or_404(connection)
        with _write_transaction(connection):
            existing = connection.execute(
                "SELECT * FROM ranking_revisions WHERE request_id = ?", (request_id,)
            ).fetchone()
            if existing is not None:
                if existing["request_payload_json"] != undo_payload:
                    raise RankingConflictError("request_id was already used with a different payload")
                return _result_from_revision(existing)

            if explicit_request:
                current_candidates = connection.execute(
                    """
                    SELECT s.*, r.action AS latest_action, r.session_id AS latest_session_id,
                           r.timestamp AS latest_timestamp
                    FROM ranking_sets AS s
                    JOIN ranking_revisions AS r
                      ON r.set_id = s.set_id AND r.revision = s.current_revision
                    WHERE s.set_id = ? AND s.current_revision = ?
                    """,
                    (expected_set_id, expected_revision),
                ).fetchall()
            else:
                current_candidates = connection.execute(
                    """
                    SELECT s.*, r.action AS latest_action, r.session_id AS latest_session_id,
                           r.timestamp AS latest_timestamp
                    FROM ranking_sets AS s
                    JOIN ranking_revisions AS r
                      ON r.set_id = s.set_id AND r.revision = s.current_revision
                    WHERE r.session_id = ? AND s.current_revision > 0
                    """,
                    (session_id,),
                ).fetchall()
            if expected_set_id is not None:
                pinned_set = _set_or_404(connection, expected_set_id)
                if (
                    expected_revision is not None
                    and pinned_set["current_revision"] != expected_revision
                ):
                    raise RankingConflictError("stale revision for undo")
                current_candidates = [
                    row for row in current_candidates if row["set_id"] == expected_set_id
                ]
            if not current_candidates:
                raise RankingNotFoundError(f"no current revision for session_id: {session_id}")
            current_candidates.sort(
                key=lambda row: (row["latest_timestamp"], row["set_id"]), reverse=True
            )

            current = None
            target_rows = []
            for candidate in current_candidates:
                latest_row = connection.execute(
                    "SELECT * FROM ranking_revisions WHERE set_id = ? AND revision = ?",
                    (candidate["set_id"], candidate["current_revision"]),
                ).fetchone()
                if latest_row["action"] == "submit":
                    candidate_target_rows = (
                        [latest_row] if latest_row["session_id"] == session_id else []
                    )
                elif explicit_request and target_revision is not None:
                    # A caller may explicitly walk back another still-active
                    # submission, but never select an undo revision implicitly.
                    candidate_target_rows = connection.execute(
                        """
                        SELECT r.*
                        FROM ranking_revisions AS r
                        WHERE r.set_id = ? AND r.revision = ?
                          AND r.session_id = ? AND r.action = 'submit'
                          AND NOT EXISTS (
                              SELECT 1
                              FROM ranking_revisions AS u
                              WHERE u.set_id = r.set_id
                                AND u.action = 'undo'
                                AND u.undo_of_revision = r.revision
                          )
                        """,
                        (candidate["set_id"], target_revision, session_id),
                    ).fetchall()
                elif explicit_request:
                    # The current revision is an undo and no target was pinned.
                    candidate_target_rows = []
                else:
                    candidate_target_rows = connection.execute(
                        """
                        SELECT r.*
                        FROM ranking_revisions AS r
                        WHERE r.set_id = ? AND r.session_id = ? AND r.action = 'submit'
                          AND r.revision < ?
                          AND NOT EXISTS (
                              SELECT 1
                              FROM ranking_revisions AS u
                              WHERE u.set_id = r.set_id
                                AND u.action = 'undo'
                                AND u.undo_of_revision = r.revision
                          )
                        ORDER BY r.revision DESC
                        LIMIT 1
                        """,
                        (candidate["set_id"], session_id, candidate["current_revision"]),
                    ).fetchall()
                if candidate_target_rows:
                    current = candidate
                    target_rows = candidate_target_rows
                    break

            if current is None or not target_rows:
                raise RankingNotFoundError("session has no current submission to undo")
            target = target_rows[0]
            if expected_revision is not None and current["current_revision"] != expected_revision:
                raise RankingConflictError("stale revision for undo")
            if target_revision is not None and target["revision"] != target_revision:
                raise RankingConflictError("stale target revision for undo")
            previous = connection.execute(
                "SELECT * FROM ranking_revisions WHERE set_id = ? AND revision = ?",
                (target["set_id"], target["revision"] - 1),
            ).fetchone()
            previous_state = {
                "outcome": "pending",
                "ordered_candidate_ids": [],
                "invalid_reason": None,
            } if previous is None else {
                "outcome": previous["outcome"],
                "ordered_candidate_ids": _read_json(previous["ordered_candidate_ids_json"]),
                "invalid_reason": previous["invalid_reason"],
            }
            revision = current["current_revision"] + 1
            timestamp = _utc_now()
            connection.execute(
                """
                INSERT INTO ranking_revisions
                    (set_id, revision, action, outcome, ordered_candidate_ids_json, expected_revision,
                     invalid_reason, session_id, request_id, timestamp, undo_of_revision,
                     request_payload_json)
                VALUES (?, ?, 'undo', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    current["set_id"], revision, previous_state["outcome"],
                    _json(previous_state["ordered_candidate_ids"]), current["current_revision"],
                    previous_state["invalid_reason"], session_id, request_id, timestamp,
                    target["revision"], undo_payload,
                ),
            )
            updated = connection.execute(
                """
                UPDATE ranking_sets
                SET current_revision = ?, current_outcome = ?, current_order_json = ?,
                    current_invalid_reason = ?, current_session_id = ?,
                    current_request_id = ?, current_timestamp = ?
                WHERE set_id = ? AND current_revision = ?
                """,
                (
                    revision, previous_state["outcome"], _json(previous_state["ordered_candidate_ids"]),
                    previous_state["invalid_reason"], session_id, request_id, timestamp,
                    current["set_id"], current["current_revision"],
                ),
            )
            if updated.rowcount != 1:
                raise RankingConflictError("set changed while undoing revision")
            row = connection.execute(
                "SELECT * FROM ranking_revisions WHERE set_id = ? AND revision = ?",
                (current["set_id"], revision),
            ).fetchone()
            return _result_from_revision(row)


def build_ranking_export(database: Database) -> Dict[str, Any]:
    """Build a lossless JSON-compatible snapshot of task, sets, and revisions."""
    with _connection(database) as connection:
        task = _task_or_404(connection)
        export_sets = []
        for set_row in connection.execute("SELECT * FROM ranking_sets ORDER BY source_position"):
            set_payload = {
                "set_id": set_row["set_id"],
                "source_position": set_row["source_position"],
                "display_position": set_row["display_position"],
                "metadata": _read_json(set_row["set_metadata_json"]),
                "candidates": [],
                "current": _state_from_set(set_row),
                "revisions": [],
            }
            for candidate in connection.execute(
                "SELECT * FROM ranking_candidates WHERE set_id = ? ORDER BY source_position",
                (set_row["set_id"],),
            ):
                set_payload["candidates"].append({
                    "candidate_id": candidate["candidate_id"],
                    "path": candidate["path"],
                    "media_type": candidate["media_type"],
                    "metadata": _read_json(candidate["metadata_json"]),
                    "source_position": candidate["source_position"],
                    "display_position": candidate["display_position"],
                })
            set_payload["revisions"] = [
                _revision_payload(revision)
                for revision in connection.execute(
                    "SELECT * FROM ranking_revisions WHERE set_id = ? ORDER BY revision",
                    (set_row["set_id"],),
                )
            ]
            rank_positions = {
                candidate_id: position for position, candidate_id in enumerate(
                    set_payload["current"]["ordered_candidate_ids"], start=1
                )
            }
            for candidate in set_payload["candidates"]:
                candidate["rank_position"] = rank_positions.get(candidate["candidate_id"])
            export_sets.append(set_payload)
        return {
            "mode": task["mode"],
            "criterion": _read_json(task["criterion_json"]),
            "task_settings": _read_json(task["task_settings_json"]),
            "created_at": task["created_at"],
            "sets": export_sets,
        }


class RankingStore:
    """Small object facade over the module-level repository functions."""

    def __init__(self, database: Union[str, Path, sqlite3.Connection]):
        self.database = database

    def initialize(self, criterion, sets, task_settings=None, random_seed=None):
        return initialize_ranking_store(self.database, criterion, sets, task_settings, random_seed)

    def get_next_set(self, session_id=None):
        return get_next_set(self.database, session_id)

    def get_set(self, set_id):
        return get_set(self.database, set_id)

    def get_stats(self):
        return get_stats(self.database)

    def get_history(self, set_id=None, session_id=None, limit=None, offset=0):
        return get_history(self.database, set_id, session_id, limit, offset)

    def get_latest_completed_sets(self, limit=None, offset=0):
        return get_latest_completed_sets(self.database, limit, offset)

    def submit_revision(self, request=None, **kwargs):
        if request is None:
            request = kwargs
            kwargs = {}
        return submit_revision(self.database, request, **kwargs)

    def undo_last_for_session(self, session_id, expected_revision=None, request_id=None,
                              expected_set_id=None, target_revision=None, target_set_id=None,
                              set_id=None):
        return undo_last_for_session(
            self.database, session_id, expected_revision, request_id, expected_set_id,
            target_revision, target_set_id, set_id
        )

    def build_ranking_export(self):
        return build_ranking_export(self.database)


__all__ = [
    "RankingBadRequestError", "RankingConflictError", "RankingConflict", "RankingNotFoundError",
    "RankingNotFound", "RankingStore", "RankingStoreError", "InvalidRankingRequest",
    "RankingValidationError",
    "build_ranking_export", "create_ranking_task", "create_task", "get_history",
    "get_latest_completed_sets", "get_next_set", "get_set",
    "get_stats", "initialize_ranking_store", "initialize_schema", "submit_revision",
    "undo_last_for_session",
]
