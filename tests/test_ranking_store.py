import concurrent.futures
import json
import sqlite3

import pytest

from ingest_ranking import ingest_ranking

from ranking_store import (
    RankingBadRequestError,
    RankingConflictError,
    RankingNotFoundError,
    build_ranking_export,
    create_task,
    get_history,
    get_next_set,
    get_set,
    get_stats,
    initialize_ranking_store,
    submit_revision,
    undo_last_for_session,
)


CRITERION = {"id": "visual-quality", "version": "v3", "prompt": "Which is most convincing?", "direction": "most"}


def supplied_sets():
    return [
        {
            "set_id": "set-a",
            "source_position": 10,
            "candidates": [
                {"candidate_id": "a-1", "source_position": 101, "path": "/a/1.jpg", "media_type": "image", "metadata": {"score": 0.1, "tags": ["x"]}},
                {"candidate_id": "a-2", "source_position": 102, "path": "/a/2.jpg", "metadata": {"score": 0.2, "keep": False}},
            ],
        },
        {
            "set_id": "set-b",
            "source_position": 20,
            "candidates": [
                {"candidate_id": "b-1", "source_position": 201, "path": "/b/1.wav", "media_type": "audio", "metadata": {"seconds": 1.5}},
                {"candidate_id": "b-2", "source_position": 202, "path": "/b/2.wav", "media_type": "audio", "metadata": {"seconds": 2}},
                {"candidate_id": "b-3", "source_position": 203, "path": "/b/3.wav", "media_type": "audio", "metadata": {"seconds": None}},
            ],
        },
    ]


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "ranking.sqlite"
    initialize_ranking_store(path, CRITERION, supplied_sets(), {"name": "Golden task", "threshold": 0.75}, random_seed=9)
    return path


def request(set_id, request_id, revision, session, outcome="ranked", ordered=None, invalid_reason=None):
    payload = {
        "set_id": set_id,
        "request_id": request_id,
        "expected_revision": revision,
        "session_id": session,
        "outcome": outcome,
    }
    if ordered is not None:
        payload["ordered_candidate_ids"] = ordered
    if invalid_reason is not None:
        payload["invalid_reason"] = invalid_reason
    return payload


def test_size_two_and_eight_permutations_are_accepted(tmp_path):
    sets = [{"set_id": "two", "candidates": [{"candidate_id": "0", "path": "0"}, {"candidate_id": "1", "path": "1"}]}]
    sets.append({
        "set_id": "eight",
        "candidates": [{"candidate_id": str(i), "path": str(i)} for i in range(8)],
    })
    db = tmp_path / "sizes.sqlite"
    initialize_ranking_store(db, CRITERION, sets, random_seed=1)
    two = submit_revision(db, request("two", "r-two", 0, "s", ordered=["1", "0"]))
    eight = submit_revision(db, request("eight", "r-eight", 0, "s", ordered=[str(i) for i in reversed(range(8))]))
    assert two["revision"] == 1
    assert eight["revision"] == 1


@pytest.mark.parametrize("ordered", [["a-1"], ["a-1", "a-1"], ["a-1", "foreign"], ["a-1", "b-1"]])
def test_malformed_rankings_are_rejected_without_a_revision(db, ordered):
    with pytest.raises(RankingBadRequestError):
        submit_revision(db, request("set-a", "bad", 0, "s", ordered=ordered))
    assert get_set(db, "set-a")["revision"] == 0


def test_request_idempotency_and_reuse_conflict(db):
    first = submit_revision(db, request("set-a", "same", 0, "s", ordered=["a-2", "a-1"]))
    duplicate = submit_revision(db, request("set-a", "same", 0, "s", ordered=["a-2", "a-1"]))
    assert duplicate["revision"] == first["revision"]
    assert duplicate["timestamp"] == first["timestamp"]
    assert duplicate == first
    with pytest.raises(RankingConflictError):
        submit_revision(db, request("set-a", "same", 1, "s", outcome="invalid", invalid_reason="changed"))
    assert len(get_history(db, set_id="set-a")) == 1


def test_stale_revision_is_conflict(db):
    submit_revision(db, request("set-a", "r1", 0, "s1", ordered=["a-1", "a-2"]))
    with pytest.raises(RankingConflictError):
        submit_revision(db, request("set-a", "r2", 0, "s2", ordered=["a-2", "a-1"]))


def test_ranked_invalid_transition_and_exact_undo_to_ranked_then_pending(db):
    ranked = submit_revision(db, request("set-a", "rank", 0, "reviewer", ordered=["a-2", "a-1"]))
    invalid = submit_revision(db, request("set-a", "invalidate", 1, "reviewer", outcome="invalid", invalid_reason="not comparable"))
    assert get_set(db, "set-a")["outcome"] == "invalid"
    restored_ranked = undo_last_for_session(db, "reviewer", expected_revision=invalid["revision"])
    assert restored_ranked["outcome"] == "ranked"
    assert restored_ranked["ordered_candidate_ids"] == ranked["ordered_candidate_ids"]
    assert get_set(db, "set-a")["revision"] == 3

    # A fresh set demonstrates that undo restores the implicit pending baseline exactly.
    submit_revision(db, request("set-b", "rank-b", 0, "other", ordered=["b-3", "b-1", "b-2"]))
    undone = undo_last_for_session(db, "other")
    assert undone["outcome"] == "pending"
    assert undone["ordered_candidate_ids"] == []
    assert undone["invalid_reason"] is None
    assert get_set(db, "set-b")["current"]["revision"] == 2


def test_stats_count_sets_not_candidates(db):
    assert get_stats(db) == {
        "total_sets": 2,
        "ranked_sets": 0,
        "invalid_sets": 0,
        "pending_sets": 2,
        "completed_sets": 0,
        "remaining_sets": 2,
        "total": 2,
        "labeled": 0,
        "remaining": 2,
        "percent": 0,
    }
    submit_revision(db, request("set-b", "invalid-b", 0, "s", outcome="invalid", invalid_reason="bad source"))
    stats = get_stats(db)
    assert stats["total_sets"] == 2
    assert stats["invalid_sets"] == 1
    assert stats["labeled"] == 1


def test_display_positions_are_persisted_and_distinct_from_source_and_rank(db):
    before = get_set(db, "set-a")
    again = get_set(db, "set-a")
    assert [(c["candidate_id"], c["display_position"]) for c in before["candidates"]] == [
        (c["candidate_id"], c["display_position"]) for c in again["candidates"]
    ]
    assert [c["source_position"] for c in before["candidates"]] != [c["rank_position"] for c in before["candidates"]]
    assert [c["display_position"] for c in before["candidates"]] == [1, 2]
    assert {c["candidate_id"]: c["source_position"] for c in before["candidates"]} == {
        "a-1": 101,
        "a-2": 102,
    }
    submit_revision(db, request("set-a", "rank", 0, "s", ordered=["a-2", "a-1"]))
    ranked = get_set(db, "set-a")
    positions = {c["candidate_id"]: c["rank_position"] for c in ranked["candidates"]}
    assert positions == {"a-2": 1, "a-1": 2}
    assert [c["display_position"] for c in ranked["candidates"]] == [c["display_position"] for c in before["candidates"]]


def test_next_set_uses_persisted_display_order_and_ends_after_completion(db):
    first = get_next_set(db)
    second = get_next_set(db)
    assert first["set_id"] == second["set_id"]
    submit_revision(db, request(first["set_id"], "next", 0, "s", ordered=[c["candidate_id"] for c in first["candidates"]]))
    assert get_next_set(db)["set_id"] != first["set_id"]


def test_golden_export_preserves_settings_typed_metadata_orders_and_revisions(db):
    submit_revision(db, request("set-a", "gold-rank", 0, "gold", ordered=["a-2", "a-1"]))
    submit_revision(db, request("set-a", "gold-invalid", 1, "gold", outcome="invalid", invalid_reason="unclear"))
    undo_last_for_session(db, "gold")
    export = build_ranking_export(db)
    assert export["mode"] == "ranking"
    assert export["criterion"] == CRITERION
    assert export["task_settings"] == {"name": "Golden task", "threshold": 0.75}
    exported = next(item for item in export["sets"] if item["set_id"] == "set-a")
    assert exported["source_position"] == 10
    assert len(exported["candidates"]) == 2
    candidate = next(item for item in exported["candidates"] if item["candidate_id"] == "a-1")
    assert candidate["metadata"] == {"score": 0.1, "tags": ["x"]}
    assert candidate["rank_position"] == 2
    assert [revision["outcome"] for revision in exported["revisions"]] == ["ranked", "invalid", "ranked"]
    assert exported["revisions"][2]["undo_of_revision"] == 2
    assert all("+00:00" in revision["timestamp"] for revision in exported["revisions"])


def test_unknown_set_and_undo_without_current_session_revision_are_not_found(db):
    with pytest.raises(RankingNotFoundError):
        get_set(db, "missing")
    with pytest.raises(RankingNotFoundError):
        undo_last_for_session(db, "nobody")


def test_undo_uses_latest_session_timestamp_not_largest_set_local_revision(db):
    submit_revision(db, request("set-a", "a-1", 0, "same", ordered=["a-1", "a-2"]))
    submit_revision(db, request("set-a", "a-2", 1, "same", ordered=["a-2", "a-1"]))
    submit_revision(db, request("set-b", "b-1", 0, "same", ordered=["b-1", "b-2", "b-3"]))
    undone = undo_last_for_session(db, "same")
    assert undone["set_id"] == "set-b"
    assert undone["outcome"] == "pending"


def test_two_sets_and_two_undos_walk_back_active_submissions(db):
    first_a = submit_revision(db, request("set-a", "a-1", 0, "same", ordered=["a-1", "a-2"]))
    second_a = submit_revision(db, request("set-a", "a-2", 1, "same", ordered=["a-2", "a-1"]))
    first_b = submit_revision(db, request("set-b", "b-1", 0, "same", ordered=["b-1", "b-2", "b-3"]))

    undo_b = undo_last_for_session(
        db,
        "same",
        request_id="undo-b",
        expected_set_id="set-b",
        expected_revision=first_b["revision"],
        target_revision=first_b["revision"],
    )
    undo_a = undo_last_for_session(
        db,
        "same",
        request_id="undo-a",
        expected_set_id="set-a",
        expected_revision=second_a["revision"],
        target_revision=second_a["revision"],
    )

    assert undo_b["undo_of_revision"] == first_b["revision"]
    assert undo_a["undo_of_revision"] == second_a["revision"]
    assert undo_a["ordered_candidate_ids"] == first_a["ordered_candidate_ids"]
    assert [item["action"] for item in reversed(get_history(db, session_id="same"))] == [
        "submit", "submit", "submit", "undo", "undo"
    ]


def test_duplicate_undo_request_is_idempotent_after_lost_response(db):
    submitted = submit_revision(db, request("set-a", "rank", 0, "reviewer", ordered=["a-2", "a-1"]))
    undo_request = {
        "request_id": "undo-lost-response",
        "expected_set_id": "set-a",
        "expected_revision": submitted["revision"],
        "target_revision": submitted["revision"],
    }
    first = undo_last_for_session(db, "reviewer", **undo_request)
    retry = undo_last_for_session(db, "reviewer", **undo_request)
    with pytest.raises(RankingConflictError):
        undo_last_for_session(
            db,
            "reviewer",
            request_id="undo-stale",
            expected_set_id="set-a",
            expected_revision=submitted["revision"],
        )

    assert retry == first
    assert len(get_history(db, set_id="set-a")) == 2
    assert get_set(db, "set-a")["revision"] == first["revision"]


def test_explicit_undo_requires_the_target_submit_session(db):
    submitted = submit_revision(db, request("set-a", "owned-submit", 0, "owner", ordered=["a-2", "a-1"]))

    with pytest.raises(RankingNotFoundError):
        undo_last_for_session(
            db,
            "different-session",
            request_id="foreign-undo",
            expected_set_id="set-a",
            expected_revision=submitted["revision"],
            target_revision=submitted["revision"],
        )

    assert get_set(db, "set-a")["revision"] == submitted["revision"]
    undone = undo_last_for_session(
        db,
        "owner",
        request_id="owner-undo",
        expected_set_id="set-a",
        expected_revision=submitted["revision"],
        target_revision=submitted["revision"],
    )
    assert undone["undo_of_revision"] == submitted["revision"]


def test_adjacent_ingest_schema_migration_is_safe_concurrently(tmp_path):
    jsonl = tmp_path / "sets.jsonl"
    jsonl.write_text(json.dumps({
        "set_id": "legacy-set",
        "criterion": CRITERION,
        "metadata": {"source": "legacy"},
        "candidates": [
            {"candidate_id": "first", "path": "first.jpg"},
            {"candidate_id": "second", "path": "second.jpg"},
        ],
    }) + "\n")
    database = tmp_path / "legacy.sqlite"
    ingest_ranking(jsonl, database, shuffle=False, absolute_paths=False)

    def migrate(_):
        from ranking_store import initialize_schema

        initialize_schema(database)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(migrate, range(8))) == [True] * 8

    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM ranking_tasks").fetchone()[0] == 1
        set_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(ranking_sets)")
        }
        revision_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(ranking_revisions)")
        }
    assert {"set_metadata_json", "current_session_id", "current_request_id"} <= set_columns
    assert {"action", "expected_revision", "session_id", "request_payload_json"} <= revision_columns


def test_non_empty_adjacent_ingest_schema_migration_preserves_revisions(tmp_path):
    jsonl = tmp_path / "sets.jsonl"
    jsonl.write_text(json.dumps({
        "set_id": "legacy-set",
        "criterion": CRITERION,
        "metadata": {"source": "legacy"},
        "candidates": [
            {"candidate_id": "first", "path": "first.jpg"},
            {"candidate_id": "second", "path": "second.jpg"},
        ],
    }) + "\n")
    database = tmp_path / "legacy-with-revision.sqlite"
    ingest_ranking(jsonl, database, shuffle=False, absolute_paths=False)
    legacy_timestamp = "2026-07-18 01:02:03"
    legacy_ranking_json = json.dumps(["second", "first"])
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO ranking_revisions
                (set_id, revision, request_id, expected_revision, ranking_json,
                 is_invalid, invalid_reason, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-set", 1, "legacy-request", 0,
                legacy_ranking_json, 0, None, legacy_timestamp,
            ),
        )

    def migrate(_):
        from ranking_store import initialize_schema

        initialize_schema(database)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(migrate, range(8))) == [True] * 8

    with sqlite3.connect(database) as connection:
        revision = connection.execute(
            """
            SELECT set_id, revision, request_id, timestamp, ordered_candidate_ids_json
            FROM ranking_revisions
            """
        ).fetchone()
    assert revision == (
        "legacy-set", 1, "legacy-request", legacy_timestamp, legacy_ranking_json
    )
    assert get_set(database, "legacy-set")["current"] == {
        "revision": 1,
        "outcome": "ranked",
        "ordered_candidate_ids": ["second", "first"],
        "invalid_reason": None,
        "session_id": "legacy",
        "request_id": "legacy-request",
        "timestamp": legacy_timestamp,
    }


def test_repeated_undo_skips_prior_undo_revision_and_reaches_pending(db):
    first = submit_revision(db, request("set-a", "first", 0, "reviewer", ordered=["a-1", "a-2"]))
    second = submit_revision(db, request("set-a", "second", 1, "reviewer", ordered=["a-2", "a-1"]))

    restored_first = undo_last_for_session(
        db,
        "reviewer",
        request_id="undo-second",
        expected_set_id="set-a",
        expected_revision=second["revision"],
        target_revision=second["revision"],
    )
    restored_pending = undo_last_for_session(
        db,
        "reviewer",
        request_id="undo-first",
        expected_set_id="set-a",
        expected_revision=restored_first["revision"],
        target_revision=first["revision"],
    )

    assert restored_first["undo_of_revision"] == second["revision"]
    assert restored_first["ordered_candidate_ids"] == first["ordered_candidate_ids"]
    assert restored_pending["undo_of_revision"] == first["revision"]
    assert restored_pending["outcome"] == "pending"
    assert [revision["action"] for revision in get_history(db, set_id="set-a")] == [
        "undo", "undo", "submit", "submit"
    ]


def test_public_create_task_uses_one_based_candidate_display_positions(tmp_path):
    db = tmp_path / "public-create.sqlite"
    create_task(
        db,
        CRITERION,
        [{
            "set_id": "explicit-source",
            "source_position": 17,
            "candidates": [
                {"candidate_id": "first", "source_position": 41, "path": "/first"},
                {"candidate_id": "second", "source_position": 99, "path": "/second"},
            ],
        }],
        random_seed=4,
    )

    ranking_set = get_set(db, "explicit-source")
    assert ranking_set["source_position"] == 17
    assert {item["candidate_id"]: item["source_position"] for item in ranking_set["candidates"]} == {
        "first": 41,
        "second": 99,
    }
    assert sorted(item["display_position"] for item in ranking_set["candidates"]) == [1, 2]
