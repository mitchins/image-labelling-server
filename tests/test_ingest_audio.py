"""Tests for audio ingestion."""

import json
import sqlite3
from pathlib import Path

from config import LabelConfig
from ingest_audio import create_queue_db, write_config, write_labels_and_settings


def test_audio_queue_records_media_type(tmp_path):
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFFtest")
    db_path = tmp_path / "queue.db"

    create_queue_db(db_path, [clip], media_type="audio")
    write_labels_and_settings(db_path, ["pos", "neg"], "Audio Task", "", media_type="audio")

    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT path, media_type FROM queue").fetchone()
    setting = conn.execute("SELECT value FROM settings WHERE key = 'media_type'").fetchone()
    conn.close()

    assert row == (str(clip), "audio")
    assert json.loads(setting[0]) == "audio"


def test_audio_config_writes_media_type(tmp_path):
    config_path = tmp_path / "task.json"
    write_config(
        config_path,
        LabelConfig(
            name="Audio Task",
            labels=["pos", "neg"],
            db_path="queue.db",
            media_type="audio",
            hint_field=None,
            hint_confidence_field=None,
            cluster_field=None,
            metadata_fields=[],
        ),
    )

    data = json.loads(config_path.read_text())
    assert data["media_type"] == "audio"
