import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from config import LabelConfig
from ingest_ranking import ingest_ranking
from ranking_store import initialize_ranking_store


@pytest.fixture
def ranking_client(tmp_path, monkeypatch):
    clips = []
    for name in ("a.wav", "b.wav", "c.wav"):
        path = tmp_path / name
        path.write_bytes(b"RIFFtest")
        clips.append(path)
    criterion = {
        "id": "sentiment",
        "version": "v1",
        "prompt": "Which clip sounds most positive?",
        "direction": "most",
    }
    records = [
        {
            "set_id": "pair/one",
            "criterion": criterion,
            "metadata": {"source": "messy", "weight": 0.5},
            "candidates": [
                {"candidate_id": "left/a", "path": str(clips[0]), "media_type": "audio", "metadata": {"text": "fine"}},
                {"candidate_id": "right b", "path": str(clips[1]), "media_type": "audio", "metadata": {"text": "great"}},
            ],
        },
        {
            "set_id": "set-two",
            "criterion": criterion,
            "candidates": [
                {"candidate_id": "a", "path": str(clips[0]), "media_type": "audio"},
                {"candidate_id": "b", "path": str(clips[1]), "media_type": "audio"},
                {"candidate_id": "c", "path": str(clips[2]), "media_type": "audio"},
            ],
        },
    ]
    jsonl = tmp_path / "sets.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records))
    db = tmp_path / "ranking.db"
    ingest_ranking(jsonl, db, shuffle=False)

    config = LabelConfig(
        name="Messy sentiment ranking",
        mode="ranking",
        labels=[],
        db_path=str(db),
        media_type="mixed",
        ranking_criterion=criterion,
    )
    import server as server_module
    monkeypatch.setattr(server_module, "CONFIG", config)
    monkeypatch.setattr(server_module, "DB_PATH", str(db))
    return TestClient(server_module.app), db


def _rank_payload(ranking_set, request_id="request-1", session_id="reviewer"):
    return {
        "set_id": ranking_set["set_id"],
        "request_id": request_id,
        "expected_revision": ranking_set["revision"],
        "session_id": session_id,
        "outcome": "ranked",
        "ordered_candidate_ids": [item["candidate_id"] for item in ranking_set["candidates"]],
    }


def test_ranking_next_media_submit_idempotency_stats_and_export(ranking_client):
    client, _ = ranking_client
    response = client.get("/api/next?session_id=reviewer")
    assert response.status_code == 200
    ranking_set = response.json()["set"]
    assert ranking_set["criterion"]["direction"] == "most"
    assert [item["display_position"] for item in ranking_set["candidates"]] == [1, 2]
    assert all("path" not in item for item in ranking_set["candidates"])

    media = client.get(ranking_set["candidates"][0]["url"])
    assert media.status_code == 200
    assert media.content == b"RIFFtest"

    payload = _rank_payload(ranking_set)
    first = client.post("/api/rank", json=payload)
    assert first.status_code == 200
    assert first.json()["progress"]["completed_sets"] == 1
    retry = client.post("/api/rank", json=payload)
    assert retry.status_code == 200
    assert retry.json()["revision"]["revision"] == 1

    changed_retry = dict(payload, outcome="invalid", ordered_candidate_ids=[])
    assert client.post("/api/rank", json=changed_retry).status_code == 409
    assert client.post("/api/rank", json={**payload, "score": 0.9}).status_code == 422

    exported = client.get("/api/export").json()
    assert exported["mode"] == "ranking"
    exported_set = next(item for item in exported["sets"] if item["set_id"] == "pair/one")
    assert exported_set["metadata"]["weight"] == 0.5
    assert exported_set["revisions"][0]["request_id"] == "request-1"


def test_ranking_criterion_must_match_database(ranking_client):
    _, db = ranking_client
    import server as server_module

    matching = server_module.load_config_from_db(str(db))
    assert server_module.validate_ranking_criterion(str(db), matching)["id"] == "sentiment"

    matching.ranking_criterion = {
        **matching.ranking_criterion,
        "prompt": "Which clip sounds most negative?",
    }
    with pytest.raises(ValueError, match="does not match"):
        server_module.validate_ranking_criterion(str(db), matching)

    with sqlite3.connect(db) as connection:
        assert server_module.get_database_ranking_criterion(connection)["id"] == "sentiment"
        assert connection.execute("SELECT 1").fetchone()[0] == 1


def test_ranking_history_correction_and_exact_undo(ranking_client):
    client, _ = ranking_client
    ranking_set = client.get("/api/next?session_id=reviewer").json()["set"]
    assert client.post("/api/rank", json=_rank_payload(ranking_set)).status_code == 200

    history = client.get("/api/history?page=1&per_page=12").json()
    assert history["total"] == 1
    item = history["items"][0]
    assert all("path" not in candidate for candidate in item["candidates"])
    assert item["candidates"][0]["url"].startswith("/api/ranking/media?")
    assert client.get(item["candidates"][0]["url"]).status_code == 200
    reversed_order = list(reversed(item["ordered_candidate_ids"]))
    corrected = client.post("/api/history/rerank", json={
        "set_id": item["set_id"],
        "request_id": "correction-1",
        "expected_revision": item["revision"],
        "session_id": "reviewer",
        "outcome": "ranked",
        "ordered_candidate_ids": reversed_order,
    })
    assert corrected.status_code == 200
    assert corrected.json()["revision"]["revision"] == 2

    undo_payload = {
        "request_id": "undo-correction-1",
        "expected_set_id": item["set_id"],
        "expected_revision": 2,
        "target_revision": 2,
        "session_id": "reviewer",
    }
    undone = client.post("/api/undo", json=undo_payload)
    assert undone.status_code == 200
    assert undone.json()["revision"]["outcome"] == "ranked"
    retry = client.post("/api/undo", json=undo_payload)
    assert retry.status_code == 200
    assert retry.json()["revision"] == undone.json()["revision"]
    assert all("path" not in candidate for candidate in undone.json()["set"]["candidates"])
    current = client.get("/api/export").json()["sets"][0]["current"]
    assert current["ordered_candidate_ids"] == item["ordered_candidate_ids"]


def test_native_ranking_store_loads_config_without_settings_table(tmp_path):
    db = tmp_path / "native.db"
    criterion = {
        "id": "sentiment",
        "version": "v2",
        "prompt": "Which is most positive?",
        "direction": "most",
    }
    initialize_ranking_store(
        db,
        criterion,
        [{
            "set_id": "native-set",
            "candidates": [
                {"candidate_id": "a", "path": "a.wav", "media_type": "audio"},
                {"candidate_id": "b", "path": "b.wav", "media_type": "audio"},
            ],
        }],
        task_settings={"name": "Native ranking", "media_type": "audio"},
    )

    import server as server_module
    config = server_module.load_config_from_db(str(db))
    assert config.mode == "ranking"
    assert config.name == "Native ranking"
    assert config.media_type == "audio"
    assert config.ranking_criterion == criterion


def test_ranking_rejects_partial_foreign_stale_and_mixed_invalid(ranking_client):
    client, _ = ranking_client
    ranking_set = client.get("/api/next").json()["set"]
    base = _rank_payload(ranking_set)
    assert client.post("/api/rank", json={**base, "ordered_candidate_ids": ["left/a"]}).status_code == 400
    assert client.post("/api/rank", json={**base, "ordered_candidate_ids": ["left/a", "foreign"]}).status_code == 400
    invalid = {**base, "outcome": "invalid", "ordered_candidate_ids": ["left/a"]}
    assert client.post("/api/rank", json=invalid).status_code == 400
    assert client.post("/api/rank", json=base).status_code == 200
    stale = {**base, "request_id": "stale-2"}
    assert client.post("/api/rank", json=stale).status_code == 409
