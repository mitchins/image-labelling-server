"""Tests for smart_label configuration system."""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from smart_label.config import LabelConfig, ANIME_STYLE_CONFIG


class TestLabelConfig:
    """Test LabelConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LabelConfig()
        assert config.name == "Labeling Task"
        assert len(config.labels) == 3
        assert config.db_path == "queue.db"
        assert config.host == "0.0.0.0"
        assert config.port == 8765
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LabelConfig(
            name="Test Task",
            labels=["cat", "dog", "bird"],
            label_colors={"cat": "#FF0000", "dog": "#00FF00"},
            db_path="test.db"
        )
        assert config.name == "Test Task"
        assert config.labels == ["cat", "dog", "bird"]
        assert config.label_colors["cat"] == "#FF0000"
        assert config.db_path == "test.db"
    
    def test_from_json(self):
        """Test loading config from JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "name": "JSON Task",
                "labels": ["red", "green", "blue"],
                "db_path": "json.db"
            }, f)
            temp_path = f.name
        
        try:
            config = LabelConfig.from_file(temp_path)
            assert config.name == "JSON Task"
            assert config.labels == ["red", "green", "blue"]
            assert config.db_path == "json.db"
        finally:
            Path(temp_path).unlink()
    
    def test_from_yaml(self):
        """Test loading config from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "name": "YAML Task",
                "labels": ["one", "two", "three"],
                "db_path": "yaml.db"
            }, f)
            temp_path = f.name
        
        try:
            config = LabelConfig.from_file(temp_path)
            assert config.name == "YAML Task"
            assert config.labels == ["one", "two", "three"]
            assert config.db_path == "yaml.db"
        finally:
            Path(temp_path).unlink()
    
    def test_from_file_not_found(self):
        """Test loading from non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            LabelConfig.from_file("/nonexistent/config.yaml")
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = LabelConfig(
            name="Dict Test",
            labels=["a", "b"],
            metadata_fields=["field1", "field2"]
        )
        d = config.to_dict()
        
        assert d["name"] == "Dict Test"
        assert d["labels"] == ["a", "b"]
        assert d["metadata_fields"] == ["field1", "field2"]
        assert "db_path" not in d  # Internal config not exposed
    
    def test_get_label_color(self):
        """Test getting label colors with defaults."""
        config = LabelConfig(
            labels=["red", "green", "blue", "yellow"],
            label_colors={"red": "#FF0000"}
        )
        
        # Explicit color
        assert config.get_label_color("red") == "#FF0000"
        
        # Default color (index-based)
        green_color = config.get_label_color("green")
        assert green_color.startswith("#")
        assert len(green_color) == 7
        
        # Unknown label gets fallback
        assert config.get_label_color("unknown") == "#757575"


class TestAnimeStyleConfig:
    """Test the default anime style configuration."""
    
    def test_anime_config_values(self):
        """Test anime style config has expected values."""
        assert ANIME_STYLE_CONFIG.name == "Anime Style Classification"
        assert len(ANIME_STYLE_CONFIG.labels) == 6
        assert "flat" in ANIME_STYLE_CONFIG.labels
        assert "moe" in ANIME_STYLE_CONFIG.labels
        assert ANIME_STYLE_CONFIG.hint_field == "predicted_style"
        assert ANIME_STYLE_CONFIG.cluster_field == "cluster_id"
    
    def test_anime_config_colors(self):
        """Test anime style colors are defined."""
        for label in ANIME_STYLE_CONFIG.labels:
            color = ANIME_STYLE_CONFIG.get_label_color(label)
            assert color.startswith("#")
            assert len(color) == 7
    
    def test_anime_config_metadata(self):
        """Test anime config metadata fields."""
        assert "series_name" in ANIME_STYLE_CONFIG.metadata_fields
        assert "production_year" in ANIME_STYLE_CONFIG.metadata_fields
