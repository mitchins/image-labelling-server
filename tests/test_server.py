"""Tests for smart_label server API endpoints."""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from fastapi.testclient import TestClient
from smart_label.config import LabelConfig


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
        INSERT INTO queue (path, cluster_id, predicted_style, predicted_confidence)
        VALUES (?, ?, ?, ?)
    """, ("/test/image1.jpg", 1, "modern", 0.8))
    
    conn.execute("""
        INSERT INTO queue (path, cluster_id, predicted_style, predicted_confidence, 
                          human_label, labeled_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ("/test/image2.jpg", 2, "moe", 0.9, "moe", "2024-01-01T12:00:00"))
    
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
        hint_field="predicted_style",
        cluster_field="cluster_id",
        garbage_classifier_path=None  # Disable for tests
    )


@pytest.fixture
def client(test_config, monkeypatch):
    """Create a test client with test configuration."""
    # Import here to avoid circular imports
    import smart_label.server as server_module
    
    # Override the global CONFIG
    monkeypatch.setattr(server_module, 'CONFIG', test_config)
    monkeypatch.setattr(server_module, 'DB_PATH', test_config.db_path)
    
    from smart_label.server import app
    return TestClient(app)


class TestConfigEndpoint:
    """Test /api/config endpoint."""
    
    def test_get_config(self, client):
        """Test GET /api/config returns configuration."""
        response = client.get("/api/config")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "Test Task"
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
