import json
import sqlite3
from pathlib import Path

import pytest

from config import LabelConfig
from ingest_jsonl import main


def test_ontology_confirmation_ingest(monkeypatch, tmp_path):
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text('\n'.join([
        json.dumps({"path": "a.jpg", "indicative_value": "cat"}),
        json.dumps({"path": "b.jpg", "indicative_value": "dog"}),
    ]))
    ontology = tmp_path / "ontology.yaml"
    ontology.write_text("id: animals\nversion: '1'\nontology:\n  - {id: cat, display_name: Cat}\n  - {id: dog, display_name: Dog}\n")
    db = tmp_path / "queue.db"
    config = tmp_path / "config.json"
    monkeypatch.setattr("sys.argv", ["ingest_jsonl", "--jsonl", str(jsonl), "--mode",
                                      "ontology-confirmation", "--ontology", str(ontology),
                                      "--db", str(db), "--config", str(config), "--no-shuffle"])
    main()

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM queue").fetchone()[0] == 2
    assert conn.execute("SELECT indicative_value, confirmation FROM queue ORDER BY id").fetchall() == [("cat", None), ("dog", None)]
    assert json.loads(conn.execute("SELECT value FROM settings WHERE key='mode'").fetchone()[0]) == "ontology_confirmation"
    conn.close()
    assert LabelConfig.from_file(config).ontology_id == "animals"


def test_same_media_can_be_confirmed_against_multiple_ontology_values(monkeypatch, tmp_path):
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text('\n'.join([
        json.dumps({"path": "same.wav", "indicative_value": "cat"}),
        json.dumps({"path": "same.wav", "indicative_value": "dog"}),
    ]))
    ontology = tmp_path / "ontology.json"
    ontology.write_text(json.dumps({
        "id": "animals",
        "version": "1",
        "ontology": [
            {"id": "cat", "display_name": "Cat"},
            {"id": "dog", "display_name": "Dog"},
        ],
    }))
    db = tmp_path / "queue.db"
    config = tmp_path / "config.json"
    monkeypatch.setattr("sys.argv", [
        "ingest_jsonl", "--jsonl", str(jsonl), "--mode", "ontology-confirmation",
        "--ontology", str(ontology), "--db", str(db), "--config", str(config),
        "--no-shuffle",
    ])
    main()
    with sqlite3.connect(db) as conn:
        rows = conn.execute("SELECT path, indicative_value FROM queue ORDER BY id").fetchall()
    assert len(rows) == 2
    assert rows[0][0] == rows[1][0]
    assert [row[1] for row in rows] == ["cat", "dog"]


@pytest.mark.parametrize("record", [{"path": "a.jpg"}, {"path": "a.jpg", "indicative_value": "bird"}])
def test_invalid_ontology_value_does_not_create_db(monkeypatch, tmp_path, record):
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(json.dumps(record))
    ontology = tmp_path / "ontology.json"
    ontology.write_text(json.dumps({
        "id": "animals",
        "version": "1",
        "ontology": [{"id": "cat", "display_name": "Cat"}],
    }))
    db = tmp_path / "queue.db"
    monkeypatch.setattr("sys.argv", ["ingest_jsonl", "--jsonl", str(jsonl), "--mode",
                                      "ontology-confirmation", "--ontology", str(ontology), "--db", str(db)])
    with pytest.raises(SystemExit):
        main()
    assert not db.exists()


def test_confirmation_ontology_requires_id_and_version(monkeypatch, tmp_path):
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(json.dumps({"path": "a.jpg", "indicative_value": "cat"}))
    ontology = tmp_path / "ontology.json"
    ontology.write_text(json.dumps({
        "ontology": [{"id": "cat", "display_name": "Cat"}],
    }))
    db = tmp_path / "queue.db"
    monkeypatch.setattr("sys.argv", [
        "ingest_jsonl", "--jsonl", str(jsonl), "--mode", "ontology-confirmation",
        "--ontology", str(ontology), "--db", str(db),
    ])
    with pytest.raises(SystemExit, match="stable id and version"):
        main()
    assert not db.exists()


def test_legacy_config_defaults_round_trip(tmp_path):
    config = LabelConfig()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config.to_dict()))
    loaded = LabelConfig.from_file(path)
    assert loaded.mode == "classification"
    assert loaded.ontology == []


def test_metadata_fields_reject_sql_and_schema_collisions():
    from ingest_jsonl import parse_metadata_fields

    with pytest.raises(SystemExit):
        parse_metadata_fields("safe,not-valid")
    with pytest.raises(SystemExit):
        parse_metadata_fields("path")


def test_confirmation_metadata_round_trips_json_types(monkeypatch, tmp_path):
    metadata = {
        "text": "spoken line",
        "score": 7,
        "enabled": True,
        "tags": ["a", "b"],
        "details": {"source": "synthetic"},
        "empty": None,
    }
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(json.dumps({
        "path": "clip.wav", "indicative_value": "DREAD", **metadata,
    }))
    ontology = tmp_path / "ontology.json"
    ontology.write_text(json.dumps({
        "id": "registers", "version": "1", "ontology": [
            {"id": "DREAD", "display_name": "Foreboding"},
        ],
    }))
    db = tmp_path / "queue.db"
    config = tmp_path / "config.json"
    monkeypatch.setattr("sys.argv", [
        "ingest_jsonl", "--jsonl", str(jsonl), "--mode", "ontology-confirmation",
        "--ontology", str(ontology), "--db", str(db), "--config", str(config),
        "--metadata-fields", ",".join(metadata), "--no-shuffle",
    ])
    main()
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE queue SET confirmation='STRONG', confirmation_at='2026-01-01T00:00:00'"
        )
        conn.commit()
        from export_utils import build_export_payload
        item = build_export_payload(conn)["items"][0]
    for key, expected in metadata.items():
        assert item[key] == expected
        assert type(item[key]) is type(expected)
