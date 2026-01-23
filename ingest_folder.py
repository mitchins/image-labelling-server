#!/usr/bin/env python3
"""
Create a labeling queue from a folder of images.

This is the fastest path for "I have images and class names" tasks.
It creates a SQLite queue and a task config file that the server can use.
"""

import argparse
import json
import random
import sqlite3
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
}


def parse_labels(label_arg: str) -> list:
    labels = [label.strip() for label in label_arg.split(",") if label.strip()]
    return labels


def collect_images(root: Path, recursive: bool, extensions: set) -> list:
    if recursive:
        paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in extensions]
    else:
        paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    return paths


def create_queue_db(db_path: Path, image_paths: list):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript(
        """
        DROP TABLE IF EXISTS queue;
        DROP TABLE IF EXISTS sessions;

        CREATE TABLE queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            cluster_id INTEGER,
            predicted_style TEXT,
            predicted_confidence REAL,
            human_label TEXT,
            labeled_at TIMESTAMP,
            quality_flag TEXT,
            session_id TEXT
        );

        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP,
            labels_count INTEGER DEFAULT 0
        );

        CREATE INDEX idx_queue_unlabeled ON queue(human_label) WHERE human_label IS NULL;
        CREATE INDEX idx_queue_cluster ON queue(cluster_id);
        """
    )

    cur.executemany(
        """
        INSERT OR IGNORE INTO queue (
            path,
            cluster_id,
            predicted_style,
            predicted_confidence,
            human_label,
            labeled_at,
            quality_flag,
            session_id
        )
        VALUES (?, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
        """,
        [(str(p),) for p in image_paths],
    )

    conn.commit()
    conn.close()


def write_config(config_path: Path, db_path: Path, labels: list, name: str, description: str):
    config = {
        "name": name,
        "description": description,
        "labels": labels,
        "label_colors": {},
        "db_path": str(db_path),
        "hint_field": None,
        "hint_confidence_field": None,
        "cluster_field": None,
        "metadata_fields": [],
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Create a labeling queue from a folder of images")
    parser.add_argument("--images", required=True, help="Folder containing images")
    parser.add_argument(
        "--labels",
        required=True,
        help="Comma-separated list of class names, e.g. cat,dog,other",
    )
    parser.add_argument("--db", default="labeling_queue.db", help="Output SQLite database path")
    parser.add_argument("--config", default="labeling_task.json", help="Output config JSON path")
    parser.add_argument("--name", default="Image Labeling Task", help="Task name shown in UI")
    parser.add_argument("--description", default="", help="Optional task description")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan subfolders for images",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS)),
        help="Comma-separated file extensions to include (no dots)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle images before inserting",
    )
    parser.add_argument(
        "--absolute-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store absolute paths in the database",
    )

    args = parser.parse_args()

    image_root = Path(args.images)
    if not image_root.exists():
        raise SystemExit(f"Image folder not found: {image_root}")

    labels = parse_labels(args.labels)
    if not labels:
        raise SystemExit("No labels provided. Example: --labels cat,dog,other")

    extensions = {f".{ext.strip().lower()}" for ext in args.extensions.split(",") if ext.strip()}

    image_paths = collect_images(image_root, args.recursive, extensions)
    if not image_paths:
        raise SystemExit(f"No images found in {image_root} with extensions: {sorted(extensions)}")

    if args.shuffle:
        random.shuffle(image_paths)

    if args.limit:
        image_paths = image_paths[: args.limit]

    if args.absolute_paths:
        image_paths = [p.resolve() for p in image_paths]

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    create_queue_db(db_path, image_paths)

    config_path = Path(args.config)
    write_config(config_path, db_path, labels, args.name, args.description)

    print(f"✓ Queue created: {db_path} ({len(image_paths)} images)")
    print(f"✓ Config written: {config_path}")
    print()
    print("Next:")
    print(f"  python -m smart_label.server --db {db_path} --config {config_path}")


if __name__ == "__main__":
    main()
