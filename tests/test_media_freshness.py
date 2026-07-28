"""Regression tests for media identity across queue replacement and file changes."""

import sqlite3

from fastapi.testclient import TestClient

from config import LabelConfig


def make_queue(path, media_path):
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE queue (
            id INTEGER PRIMARY KEY, path TEXT, media_type TEXT NOT NULL,
            cluster_id INTEGER, predicted_style TEXT, predicted_confidence REAL,
            human_label TEXT, labeled_at TEXT, quality_flag TEXT, session_id TEXT
        )"""
    )
    conn.execute("INSERT INTO queue (id, path, media_type) VALUES (1, ?, 'audio')", (str(media_path),))
    conn.commit()
    conn.close()


def configure(monkeypatch, db_path):
    import server

    monkeypatch.setattr(server, "DB_PATH", str(db_path))
    monkeypatch.setattr(server, "CONFIG", LabelConfig(db_path=str(db_path), media_type="audio"))
    return TestClient(server.app)


def test_queue_replacement_cannot_reuse_media_url_or_cached_bytes(tmp_path, monkeypatch):
    first_media = tmp_path / "first.wav"
    second_media = tmp_path / "second.wav"
    first_media.write_bytes(b"first queue bytes")
    second_media.write_bytes(b"second queue bytes")
    first_db = tmp_path / "first.db"
    second_db = tmp_path / "second.db"
    make_queue(first_db, first_media)
    make_queue(second_db, second_media)

    client = configure(monkeypatch, first_db)
    first_url = client.get("/api/next").json()["media_url"]
    assert client.get("/api/media/1").status_code == 400
    first_response = client.get(first_url)
    assert first_response.content == b"first queue bytes"
    assert first_response.headers["cache-control"].startswith("no-store")
    assert first_response.headers["surrogate-control"] == "no-store"

    client = configure(monkeypatch, second_db)
    second_url = client.get("/api/next").json()["media_url"]
    assert second_url != first_url
    assert client.get(first_url).status_code == 409
    assert client.get(second_url).content == b"second queue bytes"

    with sqlite3.connect(second_db) as conn:
        conn.execute("UPDATE queue SET human_label = 'accepted' WHERE id = 1")
        conn.commit()
    history_url = client.get("/api/history").json()["items"][0]["media_url"]
    assert history_url == second_url


def test_in_place_media_change_invalidates_issued_url(tmp_path, monkeypatch):
    media = tmp_path / "clip.wav"
    media.write_bytes(b"original")
    db = tmp_path / "queue.db"
    make_queue(db, media)
    client = configure(monkeypatch, db)

    issued_url = client.get("/api/next").json()["media_url"]
    media.write_bytes(b"replacement")
    assert client.get(issued_url).status_code == 409
    refreshed_url = client.get("/api/next").json()["media_url"]
    assert refreshed_url != issued_url
    assert client.get(refreshed_url).content == b"replacement"
