import json
import sqlite3

import pytest

from ingest_ranking import RankingIngestError, ingest_ranking, main


def criterion(prompt="Which clip is most positive?"):
    return {"id": "sentiment", "version": "v1", "prompt": prompt, "direction": "most"}


def ranking_set(set_id="s1", candidate_count=2, *, criterion_value=None, metadata=None):
    return {
        "set_id": set_id,
        "criterion": criterion_value or criterion(),
        "metadata": metadata or {"source": "test", "weight": 1.5},
        "candidates": [
            {
                "candidate_id": chr(ord("a") + index),
                "path": f"clip-{index}.wav",
                "media_type": "audio",
                "metadata": {"score": index, "enabled": index == 0},
            }
            for index in range(candidate_count)
        ],
    }


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_candidate_boundaries_and_positions(tmp_path):
    jsonl = tmp_path / "sets.jsonl"
    write_jsonl(jsonl, [ranking_set(candidate_count=2), ranking_set("s2", 8)])
    db = tmp_path / "ranking.db"

    ingest_ranking(jsonl, db, shuffle=False, absolute_paths=False)

    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ranking_sets").fetchone()[0] == 2
        rows = conn.execute(
            "SELECT source_position, display_position FROM ranking_candidates "
            "WHERE set_id = 's2' ORDER BY source_position"
        ).fetchall()
    assert rows == [(position, position) for position in range(1, 9)]


@pytest.mark.parametrize(
    "record, message",
    [
        (ranking_set(candidate_count=1), "2 to 8"),
        (ranking_set(candidate_count=9), "2 to 8"),
    ],
)
def test_candidate_count_boundaries_reject_out_of_range(tmp_path, record, message):
    jsonl = tmp_path / "sets.jsonl"
    write_jsonl(jsonl, [record])
    with pytest.raises(RankingIngestError, match=message):
        ingest_ranking(jsonl, tmp_path / "ranking.db")


def test_duplicate_set_and_candidate_ids_reject(tmp_path):
    duplicate_set = ranking_set("same")
    jsonl = tmp_path / "sets.jsonl"
    write_jsonl(jsonl, [duplicate_set, ranking_set("same")])
    with pytest.raises(RankingIngestError, match="duplicate set_id"):
        ingest_ranking(jsonl, tmp_path / "ranking.db")

    duplicate_candidate = ranking_set()
    duplicate_candidate["candidates"][1]["candidate_id"] = "a"
    write_jsonl(jsonl, [duplicate_candidate])
    with pytest.raises(RankingIngestError, match="duplicate candidate_id"):
        ingest_ranking(jsonl, tmp_path / "ranking.db")


def test_criterion_mismatch_rejects(tmp_path):
    jsonl = tmp_path / "sets.jsonl"
    write_jsonl(jsonl, [ranking_set(), ranking_set("s2", criterion_value=criterion("Other?"))])
    with pytest.raises(RankingIngestError, match="criterion does not match"):
        ingest_ranking(jsonl, tmp_path / "ranking.db")


def test_repeated_media_across_sets_and_typed_metadata_persist(tmp_path):
    first = ranking_set(metadata={"count": 2, "flag": True, "nested": {"ok": None}})
    second = ranking_set("s2")
    second["candidates"][0]["path"] = first["candidates"][0]["path"]
    jsonl = tmp_path / "sets.jsonl"
    write_jsonl(jsonl, [first, second])
    db = tmp_path / "ranking.db"

    ingest_ranking(jsonl, db, shuffle=False, absolute_paths=False)

    with sqlite3.connect(db) as conn:
        paths = conn.execute(
            "SELECT path FROM ranking_candidates WHERE candidate_id = 'a' ORDER BY set_id"
        ).fetchall()
        set_metadata = conn.execute(
            "SELECT metadata_json FROM ranking_sets WHERE set_id = 's1'"
        ).fetchone()[0]
        candidate_metadata = conn.execute(
            "SELECT metadata_json FROM ranking_candidates "
            "WHERE set_id = 's1' AND candidate_id = 'a'"
        ).fetchone()[0]
    assert paths == [("clip-0.wav",), ("clip-0.wav",)]
    assert json.loads(set_metadata) == {"count": 2, "flag": True, "nested": {"ok": None}}
    assert json.loads(candidate_metadata) == {"score": 0, "enabled": True}


def test_base_dir_seed_and_config_are_persisted(tmp_path):
    jsonl = tmp_path / "sets.jsonl"
    write_jsonl(jsonl, [ranking_set(candidate_count=4)])
    db = tmp_path / "nested" / "ranking.db"
    config = tmp_path / "nested" / "task.json"

    ingest_ranking(
        jsonl,
        db,
        config_path=config,
        base_dir=tmp_path / "media",
        seed=17,
        absolute_paths=True,
    )

    with sqlite3.connect(db) as conn:
        paths = [row[0] for row in conn.execute(
            "SELECT path FROM ranking_candidates ORDER BY source_position"
        )]
        settings = dict(conn.execute("SELECT key, value FROM settings"))
    assert paths[0] == str((tmp_path / "media" / "clip-0.wav").resolve())
    assert json.loads(settings["mode"]) == "ranking"
    assert json.loads(config.read_text())["ranking_criterion"] == criterion()


def test_seed_repeats_display_order_and_no_shuffle_is_identity(tmp_path):
    jsonl = tmp_path / "sets.jsonl"
    write_jsonl(jsonl, [ranking_set(candidate_count=6)])
    first_db = tmp_path / "first.db"
    second_db = tmp_path / "second.db"
    ingest_ranking(jsonl, first_db, seed=23)
    ingest_ranking(jsonl, second_db, seed=23)

    with sqlite3.connect(first_db) as first, sqlite3.connect(second_db) as second:
        first_positions = first.execute(
            "SELECT display_position FROM ranking_candidates ORDER BY source_position"
        ).fetchall()
        second_positions = second.execute(
            "SELECT display_position FROM ranking_candidates ORDER BY source_position"
        ).fetchall()
    assert first_positions == second_positions
    assert sorted(position[0] for position in first_positions) == list(range(1, 7))


def test_invalid_input_does_not_damage_existing_database(tmp_path):
    valid_jsonl = tmp_path / "valid.jsonl"
    invalid_jsonl = tmp_path / "invalid.jsonl"
    db = tmp_path / "ranking.db"
    write_jsonl(valid_jsonl, [ranking_set()])
    ingest_ranking(valid_jsonl, db, shuffle=False)
    before = db.read_bytes()

    invalid = ranking_set()
    invalid["candidates"][0]["media_type"] = "video"
    write_jsonl(invalid_jsonl, [invalid])
    with pytest.raises(RankingIngestError):
        ingest_ranking(invalid_jsonl, db)

    assert db.read_bytes() == before
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ranking_sets").fetchone()[0] == 1


def test_cli_reports_validation_errors_as_system_exit(tmp_path):
    jsonl = tmp_path / "sets.jsonl"
    write_jsonl(jsonl, [ranking_set(candidate_count=1)])
    with pytest.raises(SystemExit, match="2 to 8"):
        main(["--jsonl", str(jsonl), "--db", str(tmp_path / "ranking.db")])
