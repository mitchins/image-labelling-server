#!/usr/bin/env python3
"""
Create a labeling queue from a JSONL file.

Each line should be a JSON object with at least a path field.
Optional fields can include cluster_id, predicted_style, predicted_confidence,
and any metadata fields you want to display in the UI.
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path
from typing import Optional
import yaml

from config import LabelConfig, encode_metadata_value, quote_identifier, validate_metadata_fields
from ingest_folder import write_config
from media_utils import guess_media_type_from_path, normalize_media_type

OUTCOMES = ("STRONG", "LOOSE", "NONE", "INVALID")

def parse_labels(label_arg: str) -> list:
    labels = [label.strip() for label in label_arg.split(",") if label.strip()]
    return labels


def parse_metadata_fields(metadata_arg: str) -> list:
    if not metadata_arg:
        return []
    fields = [field.strip() for field in metadata_arg.split(",") if field.strip()]
    try:
        return validate_metadata_fields(fields)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def create_queue_db(db_path: Path, metadata_fields: list, mode: str = "classification"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    metadata_columns = ",\n            ".join(
        f"{quote_identifier(field)} TEXT" for field in metadata_fields
    )
    if metadata_columns:
        metadata_columns = f",\n            {metadata_columns}"

    cur.executescript(
        f"""
        DROP TABLE IF EXISTS queue;
        DROP TABLE IF EXISTS sessions;
        DROP TABLE IF EXISTS labels;
        DROP TABLE IF EXISTS settings;

        CREATE TABLE queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT,
            media_type TEXT NOT NULL DEFAULT 'image',
            cluster_id INTEGER,
            predicted_style TEXT,
            predicted_confidence REAL,
            human_label TEXT,
            labeled_at TIMESTAMP,
            quality_flag TEXT,
            session_id TEXT,
            indicative_value TEXT,
            confirmation TEXT,
            confirmation_at TIMESTAMP,
            confirmation_session_id TEXT{metadata_columns}
        );

        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP,
            labels_count INTEGER DEFAULT 0
        );

        CREATE TABLE labels (
            name TEXT PRIMARY KEY,
            color TEXT,
            sort_order INTEGER
        );

        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX idx_queue_unlabeled ON queue(human_label) WHERE human_label IS NULL;
        CREATE INDEX idx_queue_cluster ON queue(cluster_id);
        CREATE UNIQUE INDEX idx_queue_classification_path
            ON queue(path) WHERE indicative_value IS NULL;
        CREATE UNIQUE INDEX idx_queue_confirmation_candidate
            ON queue(path, indicative_value) WHERE indicative_value IS NOT NULL;
        """
    )

    conn.commit()
    conn.close()


def write_labels_and_settings(
    db_path: Path,
    labels: list,
    name: str,
    description: str,
    media_type: str,
    metadata_fields: list,
    hint_field: str,
    hint_confidence_field: str,
    cluster_field: str,
    mode: str = "classification",
    ontology_id: Optional[str] = None,
    ontology_version: Optional[str] = None,
    ontology: Optional[list] = None,
):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executemany(
        "INSERT OR REPLACE INTO labels (name, color, sort_order) VALUES (?, NULL, ?)",
        [(label, idx) for idx, label in enumerate(labels)],
    )

    settings = {
        "name": name,
        "description": description,
        "media_type": media_type,
        "hint_field": hint_field,
        "hint_confidence_field": hint_confidence_field,
        "cluster_field": cluster_field,
        "metadata_fields": metadata_fields,
        "mode": mode,
        "ontology_id": ontology_id,
        "ontology_version": ontology_version,
        "ontology": ontology or [],
    }

    cur.executemany(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
        [(key, json.dumps(value)) for key, value in settings.items()],
    )

    conn.commit()
    conn.close()


def load_jsonl(
    jsonl_path: Path,
    path_field: str,
    cluster_field: str,
    hint_field: str,
    hint_confidence_field: str,
    metadata_fields: list,
    base_dir: Path,
    absolute_paths: bool,
    media_type: Optional[str],
    indicative_field: str = "indicative_value",
    mode: str = "classification",
    ontology_ids: Optional[set] = None,
):
    items = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_num}: {exc}") from exc

            if path_field not in record:
                raise SystemExit(f"Missing '{path_field}' on line {line_num}")

            raw_path = record[path_field]
            if not isinstance(raw_path, str) or not raw_path:
                raise SystemExit(f"Invalid '{path_field}' on line {line_num}")

            indicative_value = record.get(indicative_field)
            if mode == "ontology_confirmation":
                if indicative_value is None or indicative_value == "":
                    raise SystemExit(f"Missing '{indicative_field}' on line {line_num}")
                if indicative_value not in ontology_ids:
                    raise SystemExit(f"Unknown '{indicative_field}' value {indicative_value!r} on line {line_num}")

            path = Path(raw_path)
            if base_dir and not path.is_absolute():
                path = base_dir / path
            if absolute_paths:
                path = path.resolve()

            items.append(
                {
                    "path": str(path),
                    "media_type": normalize_media_type(media_type) if media_type else guess_media_type_from_path(path),
                    "cluster_id": record.get(cluster_field),
                    "predicted_style": record.get(hint_field),
                    "predicted_confidence": record.get(hint_confidence_field),
                    "indicative_value": indicative_value,
                    "metadata": {field: record.get(field) for field in metadata_fields},
                }
            )

    return items


def insert_items(db_path: Path, items: list, metadata_fields: list):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    base_columns = [
        "path",
        "media_type",
        "cluster_id",
        "predicted_style",
        "predicted_confidence",
        "human_label",
        "labeled_at",
        "quality_flag",
        "session_id",
        "indicative_value",
        "confirmation",
        "confirmation_at",
        "confirmation_session_id",
    ]
    all_columns = base_columns + metadata_fields

    placeholders = ", ".join("?" for _ in all_columns)
    column_clause = ", ".join(quote_identifier(column) for column in all_columns)

    values = []
    for item in items:
        row = [
            item["path"],
            item["media_type"],
            item["cluster_id"],
            item["predicted_style"],
            item["predicted_confidence"],
            None,
            None,
            None,
            None,
            item.get("indicative_value"),
            None,
            None,
            None,
        ]
        for field in metadata_fields:
            value = item["metadata"].get(field)
            row.append(encode_metadata_value(value))
        values.append(tuple(row))

    cur.executemany(
        f"""
        INSERT OR IGNORE INTO queue ({column_clause})
        VALUES ({placeholders})
        """,
        values,
    )

    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Create a labeling queue from a JSONL file")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL file")
    parser.add_argument("--mode", choices=["classification", "ontology-confirmation"], default="classification")
    parser.add_argument("--ontology", help="JSON/YAML ontology definition")
    parser.add_argument("--indicative-field", default="indicative_value")
    parser.add_argument(
        "--labels",
        required=False,
        default="",
        help="Comma-separated list of class names, e.g. cat,dog,other",
    )
    parser.add_argument("--db", default="labeling_queue.db", help="Output SQLite database path")
    parser.add_argument("--config", default="labeling_task.json", help="Output config JSON path")
    parser.add_argument("--name", default="Media Labeling Task", help="Task name shown in UI")
    parser.add_argument("--description", default="", help="Optional task description")
    parser.add_argument(
        "--media-type",
        default="image",
        choices=["image", "audio"],
        help="Task media type for the reviewer UI",
    )
    parser.add_argument(
        "--metadata-fields",
        default="",
        help="Comma-separated JSONL fields to expose in the UI",
    )
    parser.add_argument(
        "--path-field",
        default="path",
        help="JSONL field name containing the image path",
    )
    parser.add_argument(
        "--cluster-field",
        default="cluster_id",
        help="JSONL field name containing the cluster id",
    )
    parser.add_argument(
        "--hint-field",
        default="predicted_style",
        help="JSONL field name containing the model hint label",
    )
    parser.add_argument(
        "--hint-confidence-field",
        default="predicted_confidence",
        help="JSONL field name containing the model hint confidence",
    )
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Optional base dir for relative paths in JSONL",
    )
    parser.add_argument(
        "--absolute-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store absolute paths in the database",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle items before inserting",
    )

    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise SystemExit(f"JSONL file not found: {jsonl_path}")

    labels = parse_labels(args.labels)
    if not labels:
        if args.mode == "classification":
            raise SystemExit("No labels provided. Example: --labels cat,dog,other")
        labels = list(OUTCOMES)

    metadata_fields = parse_metadata_fields(args.metadata_fields)
    ontology_id = ontology_version = None
    ontology = []
    ontology_ids = None
    if args.mode == "ontology-confirmation":
        if not args.ontology:
            raise SystemExit("--ontology is required for ontology-confirmation mode")
        ontology_path = Path(args.ontology)
        if not ontology_path.exists():
            raise SystemExit(f"Ontology file not found: {ontology_path}")
        with ontology_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) if ontology_path.suffix in (".yaml", ".yml") else json.load(f)
        if not isinstance(data, dict):
            raise SystemExit("Ontology must include id, version, and an ontology/values list")
        ontology_id = data.get("id") or data.get("ontology_id")
        ontology_version = data.get("version") or data.get("ontology_version")
        ontology = data.get("ontology", data.get("values", []))
        if not ontology_id or not ontology_version:
            raise SystemExit("Ontology must include stable id and version values")
        if not isinstance(ontology, list) or not all(
            isinstance(x, dict)
            and isinstance(x.get("id"), str)
            and x["id"].strip()
            and isinstance(x.get("display_name"), str)
            and x["display_name"].strip()
            for x in ontology
        ):
            raise SystemExit("Ontology must be a list of objects with id and display_name")
        ontology_ids = {x["id"] for x in ontology}
        if len(ontology_ids) != len(ontology):
            raise SystemExit("Ontology ids must be unique")
    base_dir = Path(args.base_dir).resolve() if args.base_dir else None

    items = load_jsonl(
        jsonl_path=jsonl_path,
        path_field=args.path_field,
        cluster_field=args.cluster_field,
        hint_field=args.hint_field,
        hint_confidence_field=args.hint_confidence_field,
        metadata_fields=metadata_fields,
        base_dir=base_dir,
        absolute_paths=args.absolute_paths,
        media_type=args.media_type,
        indicative_field=args.indicative_field,
        mode="ontology_confirmation" if args.mode == "ontology-confirmation" else args.mode,
        ontology_ids=ontology_ids,
    )

    if not items:
        raise SystemExit(f"No records found in {jsonl_path}")

    if args.shuffle:
        random.shuffle(items)

    if args.limit:
        items = items[: args.limit]

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    create_queue_db(db_path, metadata_fields, args.mode)
    write_labels_and_settings(
        db_path,
        labels,
        args.name,
        args.description,
        normalize_media_type(args.media_type),
        metadata_fields,
        args.hint_field,
        args.hint_confidence_field,
        args.cluster_field,
        "ontology_confirmation" if args.mode == "ontology-confirmation" else args.mode,
        ontology_id,
        ontology_version,
        ontology,
    )
    insert_items(db_path, items, metadata_fields)

    config_path = Path(args.config)
    write_config(
        config_path,
        LabelConfig(
            name=args.name,
            description=args.description,
            mode="ontology_confirmation" if args.mode == "ontology-confirmation" else args.mode,
            ontology_id=ontology_id,
            ontology_version=ontology_version,
            ontology=ontology,
            labels=labels,
            db_path=str(db_path),
            media_type=normalize_media_type(args.media_type),
            hint_field=args.hint_field,
            hint_confidence_field=args.hint_confidence_field,
            cluster_field=args.cluster_field,
            metadata_fields=metadata_fields,
        ),
    )

    print(f"✓ Queue created: {db_path} ({len(items)} items)")
    print(f"✓ Config written: {config_path}")
    print()
    print("Next:")
    print(f"  python -m smart_label serve --db {db_path} --config {config_path}")


if __name__ == "__main__":
    main()
