#!/usr/bin/env python3
"""
Create a labeling queue from a folder of audio files.
"""

import argparse
from pathlib import Path

from config import LabelConfig
from ingest_folder import create_queue_db, parse_labels, write_config, write_labels_and_settings
from media_utils import AUDIO_EXTENSIONS, collect_media


def main():
    parser = argparse.ArgumentParser(description="Create a labeling queue from a folder of audio files")
    parser.add_argument("--audio", required=True, help="Folder containing audio clips")
    parser.add_argument(
        "--labels",
        required=True,
        help="Comma-separated arbitrary labels, e.g. positive,negative,neutral",
    )
    parser.add_argument("--db", default="labeling_queue.db", help="Output SQLite database path")
    parser.add_argument("--config", default="labeling_task.json", help="Output config JSON path")
    parser.add_argument("--name", default="Audio Labeling Task", help="Task name shown in UI")
    parser.add_argument("--description", default="", help="Optional task description")
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scan subfolders for audio",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(sorted(ext.lstrip(".") for ext in AUDIO_EXTENSIONS)),
        help="Comma-separated file extensions to include (no dots)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of clips")
    parser.add_argument(
        "--shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle clips before inserting",
    )
    parser.add_argument(
        "--absolute-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store absolute paths in the database",
    )

    args = parser.parse_args()

    audio_root = Path(args.audio)
    if not audio_root.exists():
        raise SystemExit(f"Audio folder not found: {audio_root}")

    labels = parse_labels(args.labels)
    if not labels:
        raise SystemExit("No labels provided. Example: --labels positive,negative,neutral")

    extensions = {f".{ext.strip().lower()}" for ext in args.extensions.split(",") if ext.strip()}
    clip_paths = collect_media(audio_root, args.recursive, extensions)
    if not clip_paths:
        raise SystemExit(f"No audio found in {audio_root} with extensions: {sorted(extensions)}")

    if args.shuffle:
        import random
        random.shuffle(clip_paths)

    if args.limit:
        clip_paths = clip_paths[: args.limit]

    if args.absolute_paths:
        clip_paths = [p.resolve() for p in clip_paths]

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    create_queue_db(db_path, clip_paths, media_type="audio")
    write_labels_and_settings(db_path, labels, args.name, args.description, media_type="audio")

    config_path = Path(args.config)
    write_config(
        config_path,
        LabelConfig(
            name=args.name,
            description=args.description,
            labels=labels,
            db_path=str(db_path),
            media_type="audio",
            hint_field=None,
            hint_confidence_field=None,
            cluster_field=None,
            metadata_fields=[],
        ),
    )

    print(f"✓ Queue created: {db_path} ({len(clip_paths)} audio clips)")
    print(f"✓ Config written: {config_path}")
    print()
    print("Next:")
    print(f"  python -m smart_label serve --db {db_path} --config {config_path}")


if __name__ == "__main__":
    main()
