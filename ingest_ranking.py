#!/usr/bin/env python3
"""Create an append-only ranking task database from JSONL sets.

Each non-empty JSONL line is one ranking set.  Validation happens before any
destination is changed, and the database is assembled in a temporary file in
the destination directory before it is atomically replaced.
"""

import argparse
import json
import os
import random
import sqlite3
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Optional, Union

CRITERION_KEYS = ("id", "version", "prompt", "direction")
VALID_MEDIA_TYPES = {"image", "audio"}


class RankingIngestError(ValueError):
    """Raised when a ranking JSONL input does not satisfy the MVP contract."""


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RankingIngestError(f"{field} must be a non-empty string")
    return value


def _json_text(value: Any, field: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RankingIngestError(f"{field} must contain JSON-compatible values") from exc


def validate_criterion(criterion: Any) -> dict:
    """Validate and return the canonical immutable ranking criterion."""
    if not isinstance(criterion, dict):
        raise RankingIngestError("criterion must be an object")
    if set(criterion) != set(CRITERION_KEYS):
        raise RankingIngestError(
            "criterion must contain exactly id, version, prompt, and direction"
        )
    normalized = {
        key: _nonempty_string(criterion[key], f"criterion.{key}")
        for key in ("id", "version", "prompt")
    }
    if criterion["direction"] != "most":
        raise RankingIngestError("criterion.direction must be 'most'")
    normalized["direction"] = "most"
    return normalized


def _resolve_path(raw_path: str, base_dir: Optional[Path], absolute_paths: bool) -> str:
    path = Path(raw_path)
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    if absolute_paths:
        path = path.resolve()
    return str(path)


def validate_ranking_set(
    record: Any,
    *,
    line_number: int,
    base_dir: Optional[Path] = None,
    absolute_paths: bool = True,
) -> dict:
    """Validate and normalize one JSONL ranking set."""
    if not isinstance(record, dict):
        raise RankingIngestError(f"line {line_number}: set must be a JSON object")

    set_id = _nonempty_string(record.get("set_id"), f"line {line_number}: set_id")
    criterion = validate_criterion(record.get("criterion"))

    metadata = record.get("metadata", {})
    if not isinstance(metadata, dict):
        raise RankingIngestError(f"line {line_number}: metadata must be an object")
    _json_text(metadata, f"line {line_number}: metadata")

    candidates = record.get("candidates")
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 8:
        raise RankingIngestError(f"line {line_number}: candidates must contain 2 to 8 items")

    normalized_candidates = []
    candidate_ids = set()
    for position, candidate in enumerate(candidates, start=1):
        if not isinstance(candidate, dict):
            raise RankingIngestError(
                f"line {line_number}, candidate {position}: candidate must be an object"
            )
        candidate_id = _nonempty_string(
            candidate.get("candidate_id"),
            f"line {line_number}, candidate {position}: candidate_id",
        )
        if candidate_id in candidate_ids:
            raise RankingIngestError(
                f"line {line_number}: duplicate candidate_id {candidate_id!r}"
            )
        candidate_ids.add(candidate_id)

        raw_path = _nonempty_string(
            candidate.get("path"),
            f"line {line_number}, candidate {position}: path",
        )
        media_type = candidate.get("media_type")
        if media_type is not None:
            media_type = _nonempty_string(
                media_type,
                f"line {line_number}, candidate {position}: media_type",
            ).lower()
            if media_type not in VALID_MEDIA_TYPES:
                raise RankingIngestError(
                    f"line {line_number}, candidate {position}: media_type must be image or audio"
                )

        candidate_metadata = candidate.get("metadata", {})
        if not isinstance(candidate_metadata, dict):
            raise RankingIngestError(
                f"line {line_number}, candidate {position}: metadata must be an object"
            )
        _json_text(
            candidate_metadata,
            f"line {line_number}, candidate {position}: metadata",
        )

        normalized_candidates.append(
            {
                "candidate_id": candidate_id,
                "path": _resolve_path(raw_path, base_dir, absolute_paths),
                "media_type": media_type,
                "metadata": candidate_metadata,
                "source_position": position,
            }
        )

    return {
        "set_id": set_id,
        "criterion": criterion,
        "metadata": metadata,
        "candidates": normalized_candidates,
    }


def load_ranking_jsonl(
    jsonl_path: Union[Path, str],
    *,
    base_dir: Optional[Union[Path, str]] = None,
    absolute_paths: bool = True,
) -> list[dict]:
    """Read, validate, and normalize all ranking sets from a JSONL file."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    resolved_base = Path(base_dir).resolve() if base_dir is not None else None
    sets = []
    set_ids = set()
    criterion = None

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"invalid JSON constant {value}")
                ))
            except (json.JSONDecodeError, ValueError) as exc:
                raise RankingIngestError(f"line {line_number}: invalid JSON: {exc}") from exc

            normalized = validate_ranking_set(
                record,
                line_number=line_number,
                base_dir=resolved_base,
                absolute_paths=absolute_paths,
            )
            if normalized["set_id"] in set_ids:
                raise RankingIngestError(
                    f"line {line_number}: duplicate set_id {normalized['set_id']!r}"
                )
            set_ids.add(normalized["set_id"])
            if criterion is None:
                criterion = normalized["criterion"]
            elif normalized["criterion"] != criterion:
                raise RankingIngestError(f"line {line_number}: criterion does not match prior sets")
            sets.append(normalized)

    if not sets:
        raise RankingIngestError(f"No ranking sets found in {path}")
    return sets


def _assign_display_positions(sets: list[dict], *, shuffle: bool, seed: Optional[int]) -> None:
    rng = random.Random(seed)
    for ranking_set in sets:
        positions = list(range(1, len(ranking_set["candidates"]) + 1))
        if shuffle:
            rng.shuffle(positions)
        for candidate, display_position in zip(ranking_set["candidates"], positions):
            candidate["display_position"] = display_position


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA foreign_keys = ON;

        CREATE TABLE ranking_sets (
            set_id TEXT PRIMARY KEY,
            criterion_id TEXT NOT NULL,
            criterion_version TEXT NOT NULL,
            criterion_prompt TEXT NOT NULL,
            criterion_direction TEXT NOT NULL CHECK (criterion_direction = 'most'),
            criterion_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            current_revision INTEGER NOT NULL DEFAULT 0 CHECK (current_revision >= 0),
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE ranking_candidates (
            set_id TEXT NOT NULL,
            candidate_id TEXT NOT NULL,
            path TEXT NOT NULL,
            media_type TEXT CHECK (media_type IS NULL OR media_type IN ('image', 'audio')),
            metadata_json TEXT NOT NULL,
            source_position INTEGER NOT NULL CHECK (source_position BETWEEN 1 AND 8),
            display_position INTEGER NOT NULL CHECK (display_position BETWEEN 1 AND 8),
            PRIMARY KEY (set_id, candidate_id),
            UNIQUE (set_id, source_position),
            UNIQUE (set_id, display_position),
            FOREIGN KEY (set_id) REFERENCES ranking_sets(set_id) ON DELETE CASCADE
        );

        CREATE TABLE ranking_revisions (
            revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
            set_id TEXT NOT NULL,
            revision INTEGER NOT NULL CHECK (revision > 0),
            request_id TEXT NOT NULL,
            expected_revision INTEGER NOT NULL CHECK (expected_revision >= 0),
            ranking_json TEXT,
            is_invalid INTEGER NOT NULL DEFAULT 0 CHECK (is_invalid IN (0, 1)),
            invalid_reason TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (set_id, revision),
            UNIQUE (set_id, request_id),
            FOREIGN KEY (set_id) REFERENCES ranking_sets(set_id) ON DELETE CASCADE
        );

        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_activity TEXT,
            labels_count INTEGER NOT NULL DEFAULT 0,
            revisions_count INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX idx_ranking_candidates_display
            ON ranking_candidates(set_id, display_position);
        CREATE INDEX idx_ranking_revisions_set
            ON ranking_revisions(set_id, revision);
        """
    )


def _write_database(
    db_path: Path,
    sets: list[dict],
    *,
    name: str,
    description: str,
    shuffle: bool,
    seed: Optional[int],
    absolute_paths: bool,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{db_path.name}.", suffix=".tmp", dir=db_path.parent, delete=False
        ) as temporary:
            temp_path = Path(temporary.name)
        with sqlite3.connect(temp_path) as conn:
            _create_schema(conn)
            criterion = sets[0]["criterion"]
            conn.executemany(
                """
                INSERT INTO ranking_sets (
                    set_id, criterion_id, criterion_version, criterion_prompt,
                    criterion_direction, criterion_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        ranking_set["set_id"],
                        ranking_set["criterion"]["id"],
                        ranking_set["criterion"]["version"],
                        ranking_set["criterion"]["prompt"],
                        ranking_set["criterion"]["direction"],
                        _json_text(ranking_set["criterion"], "criterion"),
                        _json_text(ranking_set["metadata"], "set metadata"),
                    )
                    for ranking_set in sets
                ],
            )
            conn.executemany(
                """
                INSERT INTO ranking_candidates (
                    set_id, candidate_id, path, media_type, metadata_json,
                    source_position, display_position
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        ranking_set["set_id"],
                        candidate["candidate_id"],
                        candidate["path"],
                        candidate["media_type"],
                        _json_text(candidate["metadata"], "candidate metadata"),
                        candidate["source_position"],
                        candidate["display_position"],
                    )
                    for ranking_set in sets
                    for candidate in ranking_set["candidates"]
                ],
            )
            settings = {
                "name": name,
                "description": description,
                "mode": "ranking",
                "media_type": "mixed",
                "ranking_criterion": criterion,
                "metadata_fields": [],
                "shuffle": shuffle,
                "seed": seed,
                "absolute_paths": absolute_paths,
            }
            conn.executemany(
                "INSERT INTO settings (key, value) VALUES (?, ?)",
                [(key, _json_text(value, f"setting {key}")) for key, value in settings.items()],
            )
            conn.commit()
        os.replace(temp_path, db_path)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def write_ranking_config(
    config_path: Union[Path, str],
    *,
    name: str,
    description: str,
    db_path: Union[Path, str],
    criterion: dict,
) -> None:
    """Write a future-compatible ranking LabelConfig JSON atomically."""
    destination = Path(config_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "description": description,
        "mode": "ranking",
        "ranking_criterion": criterion,
        "labels": [],
        "db_path": str(db_path),
        "media_type": "mixed",
        "hint_field": None,
        "hint_confidence_field": None,
        "cluster_field": None,
        "metadata_fields": [],
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def ingest_ranking(
    jsonl_path: Union[Path, str],
    db_path: Union[Path, str],
    *,
    config_path: Optional[Union[Path, str]] = None,
    name: str = "Ranking Task",
    description: str = "",
    base_dir: Optional[Union[Path, str]] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    absolute_paths: bool = True,
) -> list[dict]:
    """Validate JSONL, atomically replace the SQLite DB, and optionally write config."""
    sets = load_ranking_jsonl(
        jsonl_path,
        base_dir=base_dir,
        absolute_paths=absolute_paths,
    )
    _assign_display_positions(sets, shuffle=shuffle, seed=seed)
    _write_database(
        Path(db_path),
        sets,
        name=name,
        description=description,
        shuffle=shuffle,
        seed=seed,
        absolute_paths=absolute_paths,
    )
    if config_path is not None:
        write_ranking_config(
            config_path,
            name=name,
            description=description,
            db_path=db_path,
            criterion=sets[0]["criterion"],
        )
    return sets


# Small aliases keep the core operations convenient for callers that use the
# terminology "load" or "create" rather than the CLI-oriented function name.
load_jsonl = load_ranking_jsonl


def create_ranking_database(
    db_path: Union[Path, str],
    sets: list[dict],
    *,
    name: str = "Ranking Task",
    description: str = "",
    shuffle: bool = True,
    seed: Optional[int] = None,
    absolute_paths: bool = True,
) -> None:
    """Write already-normalized sets to an atomically replaced database."""
    _assign_display_positions(sets, shuffle=shuffle, seed=seed)
    _write_database(
        Path(db_path),
        sets,
        name=name,
        description=description,
        shuffle=shuffle,
        seed=seed,
        absolute_paths=absolute_paths,
    )


create_ranking_db = create_ranking_database


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a ranking task from JSONL sets")
    parser.add_argument("--jsonl", required=True, help="Path to ranking JSONL")
    parser.add_argument("--db", default="ranking_queue.db", help="Output SQLite database")
    parser.add_argument("--config", default="ranking_task.json", help="Output config JSON")
    parser.add_argument("--name", default="Ranking Task", help="Task name")
    parser.add_argument("--description", default="", help="Optional task description")
    parser.add_argument("--base-dir", default=None, help="Base directory for relative media paths")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomize persisted display positions once",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for display position shuffling"
    )
    parser.add_argument(
        "--absolute-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store resolved absolute paths",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        sets = ingest_ranking(
            args.jsonl,
            args.db,
            config_path=args.config,
            name=args.name,
            description=args.description,
            base_dir=args.base_dir,
            shuffle=args.shuffle,
            seed=args.seed,
            absolute_paths=args.absolute_paths,
        )
    except (RankingIngestError, FileNotFoundError, OSError, sqlite3.Error) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Ranking database created: {args.db} ({len(sets)} sets)")
    print(f"Config written: {args.config}")


if __name__ == "__main__":
    main()
