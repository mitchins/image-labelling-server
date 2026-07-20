"""Tests for smart_label configuration system."""

import pytest
import tempfile
import json
import yaml
from pathlib import Path
from config import (
    ANIME_STYLE_CONFIG,
    LabelConfig,
    decode_metadata_value,
    encode_metadata_value,
)


class TestLabelConfig:
    """Test LabelConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LabelConfig()
        assert config.name == "Labeling Task"
        assert len(config.labels) == 3
        assert config.db_path == "queue.db"
        assert config.media_type == "image"
        assert config.audio_autoplay_default is False
        assert config.audio_autoplay_persistence == "session"
        assert config.host == "0.0.0.0"
        assert config.port == 8765
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = LabelConfig(
            name="Test Task",
            labels=["cat", "dog", "bird"],
            label_colors={"cat": "#FF0000", "dog": "#00FF00"},
            db_path="test.db",
            media_type="audio",
        )
        assert config.name == "Test Task"
        assert config.labels == ["cat", "dog", "bird"]
        assert config.label_colors["cat"] == "#FF0000"
        assert config.db_path == "test.db"
        assert config.media_type == "audio"

    def test_audio_autoplay_config(self):
        config = LabelConfig(
            media_type="audio",
            audio_autoplay_default=True,
            audio_autoplay_persistence="cookie",
        )
        assert config.to_dict()["audio_autoplay_default"] is True
        assert config.to_dict()["audio_autoplay_persistence"] == "cookie"

        with pytest.raises(ValueError, match="audio_autoplay_persistence"):
            LabelConfig(audio_autoplay_persistence="local")

        with pytest.raises(ValueError, match="audio_autoplay_default"):
            LabelConfig(audio_autoplay_default="yes")

    def test_confirmation_config_contract(self):
        config = LabelConfig(
            mode="ontology_confirmation",
            ontology_id="prosody-register",
            ontology_version="v1",
            ontology=[{"id": "DREAD", "display_name": "Foreboding"}],
            labels=["STRONG", "LOOSE", "NONE", "INVALID"],
        )
        assert config.to_dict()["ontology"][0]["id"] == "DREAD"

    def test_confirmation_config_requires_versioned_ontology(self):
        with pytest.raises(ValueError):
            LabelConfig(mode="ontology_confirmation")

    def test_ranking_config_requires_versioned_most_criterion(self):
        valid_criterion = {
            "id": "sentiment",
            "version": "v1",
            "prompt": "Which clip is most positive?",
            "direction": "most",
        }
        config = LabelConfig(
            mode="ranking",
            labels=[],
            ranking_criterion=valid_criterion,
        )
        assert config.to_dict()["ranking_criterion"]["id"] == "sentiment"

        with pytest.raises(ValueError, match="exactly"):
            LabelConfig(
                mode="ranking",
                labels=[],
                ranking_criterion={**valid_criterion, "unexpected": True},
            )
        with pytest.raises(ValueError, match="exactly"):
            LabelConfig(
                mode="ranking",
                labels=[],
                ranking_criterion={key: value for key, value in valid_criterion.items()
                                   if key != "direction"},
            )

        with pytest.raises(ValueError, match="ranking_criterion"):
            LabelConfig(mode="ranking", labels=[])
        with pytest.raises(ValueError, match="direction"):
            LabelConfig(
                mode="ranking",
                labels=[],
                ranking_criterion={
                    "id": "sentiment",
                    "version": "v1",
                    "prompt": "Which clip is least positive?",
                    "direction": "least",
                },
            )

    @pytest.mark.parametrize("legacy", ["123", "true", "null", "[1,2]"])
    def test_legacy_metadata_that_looks_like_json_stays_text(self, legacy):
        assert decode_metadata_value(legacy) == legacy

    @pytest.mark.parametrize("value", [123, True, [1, 2], {"a": 1}, "text"])
    def test_marked_metadata_round_trips_type(self, value):
        decoded = decode_metadata_value(encode_metadata_value(value))
        assert decoded == value
        assert type(decoded) is type(value)
    
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
        assert d["media_type"] == "image"
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
