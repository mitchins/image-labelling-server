"""Tests for smart_label server API endpoints."""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from fastapi.testclient import TestClient
from config import LabelConfig


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Create test database schema
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE queue (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            media_type TEXT NOT NULL DEFAULT 'image',
            cluster_id INTEGER,
            predicted_style TEXT,
            predicted_confidence REAL,
            human_label TEXT,
            labeled_at TIMESTAMP,
            quality_flag TEXT,
            session_id TEXT
        )
    """)
    
    # Insert test data
    conn.execute("""
        INSERT INTO queue (path, media_type, cluster_id, predicted_style, predicted_confidence)
        VALUES (?, ?, ?, ?, ?)
    """, ("/test/image1.jpg", "image", 1, "modern", 0.8))
    
    conn.execute("""
        INSERT INTO queue (path, media_type, cluster_id, predicted_style, predicted_confidence, 
                          human_label, labeled_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, ("/test/image2.jpg", "image", 2, "moe", 0.9, "moe", "2024-01-01T12:00:00"))
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def test_config(test_db):
    """Create a test configuration."""
    return LabelConfig(
        name="Test Task",
        labels=["flat", "grim", "modern", "moe"],
        db_path=test_db,
        media_type="image",
        hint_field="predicted_style",
        cluster_field="cluster_id",
        garbage_classifier_path=None  # Disable for tests
    )


@pytest.fixture
def client(test_config, monkeypatch):
    """Create a test client with test configuration."""
    # Import here to avoid circular imports
    import server as server_module
    
    # Override the global CONFIG
    monkeypatch.setattr(server_module, 'CONFIG', test_config)
    monkeypatch.setattr(server_module, 'DB_PATH', test_config.db_path)
    
    from server import app
    return TestClient(app)


class TestConfigEndpoint:
    """Test /api/config endpoint."""
    
    def test_get_config(self, client):
        """Test GET /api/config returns configuration."""
        response = client.get("/api/config")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Test Task"
        assert data["media_type"] == "image"
        assert len(data["labels"]) == 4
        assert "flat" in data["labels"]
        assert data["hint_field"] == "predicted_style"


class TestStatsEndpoint:
    """Test /api/stats endpoint."""
    
    def test_get_stats(self, client):
        """Test GET /api/stats returns statistics."""
        response = client.get("/api/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert data["total"] == 2
        assert data["labeled"] == 1
        assert data["remaining"] == 1
        assert "by_label" in data


class TestHistoryEndpoint:
    """Test /api/history endpoints."""
    
    def test_get_history(self, client):
        """Test GET /api/history returns labeled items."""
        response = client.get("/api/history?page=1&per_page=10")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] >= 1
        assert len(data["items"]) >= 1
    
    def test_get_history_with_filter(self, client):
        """Test GET /api/history with label filter."""
        response = client.get("/api/history?label_filter=moe")
        assert response.status_code == 200
        
        data = response.json()
        assert all(item["label"] == "moe" for item in data["items"])
    
    def test_get_history_pagination(self, client):
        """Test history pagination."""
        response = client.get("/api/history?page=1&per_page=1")
        assert response.status_code == 200
        
        data = response.json()
        assert data["per_page"] == 1
        assert data["page"] == 1


class TestLabelValidation:
    """Test label validation."""
    
    def test_valid_labels(self, client):
        """Test that only configured labels are accepted."""
        # This would require mocking the label endpoint properly
        # For now, just verify config loads correctly
        response = client.get("/api/config")
        assert response.status_code == 200
        
        data = response.json()
        assert set(data["labels"]) == {"flat", "grim", "modern", "moe"}


@pytest.fixture
def confirmation_client(tmp_path, monkeypatch):
    db_path = tmp_path / "confirmation.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE queue (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            media_type TEXT NOT NULL DEFAULT 'audio',
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
            confirmation_session_id TEXT,
            narration_text TEXT
        );
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP,
            labels_count INTEGER DEFAULT 0
        );
        CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT);
    """)
    conn.executemany(
        "INSERT INTO queue (id, path, media_type, indicative_value, narration_text) VALUES (?, ?, 'audio', ?, ?)",
        [
            (1, "/test/a.wav", "DREAD", "Something moved."),
            (2, "/test/b.wav", "BRIGHT", "Morning opened."),
        ],
    )
    conn.executemany(
        "INSERT INTO settings (key, value) VALUES (?, ?)",
        [
            ("mode", '"ontology_confirmation"'),
            ("ontology_id", '"prosody-register"'),
            ("ontology_version", '"v1"'),
            ("ontology", '[{"id":"DREAD","display_name":"Foreboding"},{"id":"BRIGHT","display_name":"Bright"}]'),
            ("metadata_fields", '["narration_text"]'),
        ],
    )
    conn.commit()
    conn.close()

    config = LabelConfig(
        name="Register confirmation",
        mode="ontology_confirmation",
        ontology_id="prosody-register",
        ontology_version="v1",
        ontology=[
            {"id": "DREAD", "display_name": "Foreboding"},
            {"id": "BRIGHT", "display_name": "Bright"},
        ],
        labels=["STRONG", "LOOSE", "NONE", "INVALID"],
        db_path=str(db_path),
        media_type="audio",
        metadata_fields=["narration_text"],
    )
    import server as server_module
    monkeypatch.setattr(server_module, "CONFIG", config)
    monkeypatch.setattr(server_module, "DB_PATH", str(db_path))
    return TestClient(server_module.app), db_path


class TestOntologyConfirmation:
    def test_next_submit_stats_history_export_and_duplicate(self, confirmation_client):
        client, _ = confirmation_client
        next_response = client.get("/api/next?session_id=reviewer-a")
        assert next_response.status_code == 200
        item = next_response.json()
        assert item["indicative_value"] in {"DREAD", "BRIGHT"}

        submitted = client.post("/api/label", json={
            "image_id": item["id"],
            "confirmation": "STRONG",
            "session_id": "reviewer-a",
        })
        assert submitted.status_code == 200
        assert submitted.json()["progress"]["labeled"] == 1

        duplicate = client.post("/api/label", json={
            "image_id": item["id"],
            "confirmation": "NONE",
            "session_id": "reviewer-b",
        })
        assert duplicate.status_code == 409

        stats = client.get("/api/stats").json()
        assert stats["by_label"] == {"STRONG": 1}
        history = client.get("/api/history").json()
        assert history["items"][0]["indicative_value"] == item["indicative_value"]
        assert history["items"][0]["confirmation"] == "STRONG"
        exported = client.get("/api/export").json()
        assert exported["ontology"]["id"] == "prosody-register"
        assert exported["ontology"]["version"] == "v1"
        assert exported["ontology"]["values"][0]["display_name"] == "Foreboding"
        assert exported["items"][0]["confirmation_session_id"] == "reviewer-a"
        assert exported["items"][0]["narration_text"]

    def test_invalid_outcome_and_unknown_ontology_value(self, confirmation_client):
        client, db_path = confirmation_client
        invalid = client.post("/api/label", json={"image_id": 1, "confirmation": "YES"})
        assert invalid.status_code == 400
        with sqlite3.connect(db_path) as conn:
            conn.execute("UPDATE queue SET indicative_value='UNKNOWN' WHERE id=1")
            conn.commit()
        unknown = client.post("/api/label", json={"image_id": 1, "confirmation": "NONE"})
        assert unknown.status_code == 400

    def test_relabel_changes_undo_owner_and_undo_returns_item(self, confirmation_client):
        client, _ = confirmation_client
        assert client.post("/api/label", json={
            "image_id": 1, "confirmation": "LOOSE", "session_id": "reviewer-a"
        }).status_code == 200
        corrected = client.post("/api/history/1/relabel", json={
            "confirmation": "NONE", "session_id": "reviewer-b"
        })
        assert corrected.status_code == 200

        old_owner = client.post("/api/undo?session_id=reviewer-a").json()
        assert old_owner["success"] is False
        new_owner = client.post("/api/undo?session_id=reviewer-b").json()
        assert new_owner["success"] is True
        assert new_owner["item"]["id"] == 1
        assert new_owner["item"]["indicative_value"] == "DREAD"

    def test_history_orders_by_confirmation_timestamp(self, confirmation_client):
        client, _ = confirmation_client
        for item_id, outcome in [(1, "STRONG"), (2, "LOOSE")]:
            assert client.post("/api/label", json={
                "image_id": item_id, "confirmation": outcome, "session_id": "reviewer-a"
            }).status_code == 200
        assert client.post("/api/history/1/relabel", json={
            "confirmation": "NONE", "session_id": "reviewer-a"
        }).status_code == 200
        history = client.get("/api/history").json()["items"]
        assert [item["id"] for item in history] == [1, 2]

    def test_cli_export_uses_confirmation_envelope(self, confirmation_client, tmp_path):
        client, db_path = confirmation_client
        assert client.post("/api/label", json={
            "image_id": 1, "confirmation": "STRONG", "session_id": "reviewer-a"
        }).status_code == 200
        output = tmp_path / "export.json"
        from smart_label.cli import export_labels
        export_labels(str(db_path), str(output))
        data = __import__("json").loads(output.read_text())
        assert data["mode"] == "ontology_confirmation"
        assert data["ontology"]["version"] == "v1"
        assert data["items"][0]["indicative_value"] == "DREAD"

    def test_api_export_honors_explicit_config_without_db_settings(self, confirmation_client):
        client, db_path = confirmation_client
        assert client.post("/api/label", json={
            "image_id": 1, "confirmation": "STRONG", "session_id": "reviewer-a"
        }).status_code == 200
        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP TABLE settings")
            conn.commit()
        data = client.get("/api/export").json()
        assert data["mode"] == "ontology_confirmation"
        assert data["ontology"]["id"] == "prosody-register"
        assert data["items"][0]["confirmation"] == "STRONG"

    def test_cli_stats_uses_confirmation_for_cluster_counts(self, confirmation_client, capsys):
        client, db_path = confirmation_client
        assert client.post("/api/label", json={
            "image_id": 1, "confirmation": "STRONG", "session_id": "reviewer-a"
        }).status_code == 200
        from smart_label.cli import show_stats
        show_stats(str(db_path))
        output = capsys.readouterr().out
        assert "Progress: 1 / 2" in output
        assert "Cluster None: 1/2 labeled" in output

    def test_legacy_schema_is_migrated_without_changing_classification(self, client, test_db):
        response = client.get("/api/next")
        assert response.status_code == 200
        with sqlite3.connect(test_db) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(queue)")}
        assert "confirmation" in columns
        assert response.json().get("indicative_value") is None
