"""
Configuration system for Smart Label.

Allows customizing labels, media sources, and optional features per task.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import yaml


METADATA_JSON_PREFIX = "smart-label-json:"


QUEUE_COLUMNS = {
    "id", "path", "media_type", "cluster_id", "predicted_style",
    "predicted_confidence", "human_label", "labeled_at", "quality_flag",
    "session_id", "indicative_value", "confirmation", "confirmation_at",
    "confirmation_session_id",
}


def validate_metadata_fields(fields: list) -> list:
    """Reject duplicate, malformed, or schema-colliding metadata columns."""
    if not isinstance(fields, list):
        raise ValueError("metadata_fields must be a list")
    if any(not isinstance(field, str) or not field.isidentifier() for field in fields):
        raise ValueError("Metadata fields must be valid identifier names")
    if len(fields) != len(set(fields)):
        raise ValueError("Metadata fields must be unique")
    collisions = QUEUE_COLUMNS.intersection(fields)
    if collisions:
        raise ValueError(f"Metadata fields collide with queue columns: {sorted(collisions)}")
    return fields


def quote_identifier(value: str) -> str:
    """Quote a previously validated SQLite identifier."""
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"Invalid SQL identifier: {value!r}")
    return '"' + value.replace('"', '""') + '"'


def encode_metadata_value(value):
    """Encode typed metadata without confusing it with legacy plain text."""
    if value is None:
        return None
    return METADATA_JSON_PREFIX + json.dumps(value, separators=(",", ":"))


def decode_metadata_value(value):
    """Decode marked metadata and preserve all unmarked legacy strings."""
    if not isinstance(value, str) or not value.startswith(METADATA_JSON_PREFIX):
        return value
    try:
        return json.loads(value[len(METADATA_JSON_PREFIX):])
    except json.JSONDecodeError:
        return value


@dataclass
class LabelConfig:
    """Configuration for a labeling task."""
    
    # Task identity
    name: str = "Labeling Task"
    description: str = ""
    mode: str = "classification"
    ontology_id: Optional[str] = None
    ontology_version: Optional[str] = None
    ontology: list = field(default_factory=list)
    ranking_criterion: Optional[dict] = None
    
    # Labels: list of valid label names (order determines keyboard shortcuts 1-9)
    labels: list = field(default_factory=lambda: ["label_1", "label_2", "label_3"])
    
    # Label display colors (optional, uses defaults if not provided)
    label_colors: dict = field(default_factory=dict)
    
    # Database path
    db_path: str = "queue.db"

    # Task media type
    media_type: str = "image"
    
    # Optional hint/prediction field name in database (e.g., "predicted_style")
    # Set to None to disable hints
    hint_field: Optional[str] = "predicted_style"
    hint_confidence_field: Optional[str] = "predicted_confidence"
    
    # Optional clustering support
    # Set to None to disable cluster-based replacement on refuse
    cluster_field: Optional[str] = "cluster_id"
    
    # Optional metadata fields to display in UI
    metadata_fields: list = field(default_factory=list)
    
    # Optional garbage classifier
    garbage_classifier_path: Optional[str] = None
    garbage_threshold: float = 0.7
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8765

    def __post_init__(self):
        validate_metadata_fields(self.metadata_fields)
        if self.mode not in {"classification", "ontology_confirmation", "ranking"}:
            raise ValueError(f"Unsupported labeling mode: {self.mode}")
        if self.mode == "ontology_confirmation":
            if not self.ontology_id or not self.ontology_version:
                raise ValueError("Confirmation mode requires ontology_id and ontology_version")
            if not self.ontology:
                raise ValueError("Confirmation mode requires at least one ontology value")
            ids = []
            for entry in self.ontology:
                if (
                    not isinstance(entry, dict)
                    or not isinstance(entry.get("id"), str)
                    or not entry["id"].strip()
                    or not isinstance(entry.get("display_name"), str)
                    or not entry["display_name"].strip()
                ):
                    raise ValueError("Ontology entries require stable id and display_name values")
                ids.append(entry["id"])
            if len(ids) != len(set(ids)):
                raise ValueError("Ontology entry ids must be unique")
        if self.mode == "ranking":
            criterion = self.ranking_criterion
            if not isinstance(criterion, dict):
                raise ValueError("Ranking mode requires ranking_criterion")
            required = ("id", "version", "prompt")
            if any(not isinstance(criterion.get(key), str) or not criterion[key].strip() for key in required):
                raise ValueError("Ranking criterion requires non-empty id, version, and prompt")
            if criterion.get("direction") != "most":
                raise ValueError("Ranking criterion direction must be 'most'")
    
    @classmethod
    def from_file(cls, path: str) -> "LabelConfig":
        """Load config from YAML or JSON file."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path) as f:
            if path.suffix in ('.yaml', '.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls(**data)
    
    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "mode": self.mode,
            "ontology_id": self.ontology_id,
            "ontology_version": self.ontology_version,
            "ontology": self.ontology,
            "ranking_criterion": self.ranking_criterion,
            "labels": self.labels,
            "label_colors": self.label_colors,
            "media_type": self.media_type,
            "hint_field": self.hint_field,
            "hint_confidence_field": self.hint_confidence_field,
            "cluster_field": self.cluster_field,
            "metadata_fields": self.metadata_fields,
        }
    
    def get_label_color(self, label: str) -> str:
        """Get color for a label, with sensible defaults."""
        if label in self.label_colors:
            return self.label_colors[label]
        
        # Default color palette
        default_colors = [
            "#607D8B", "#37474F", "#1976D2", "#C2185B", 
            "#7B1FA2", "#F57C00", "#00897B", "#5D4037"
        ]
        
        try:
            idx = self.labels.index(label)
            return default_colors[idx % len(default_colors)]
        except ValueError:
            return "#757575"


# Default config for anime style labeling (backward compatible)
ANIME_STYLE_CONFIG = LabelConfig(
    name="Anime Style Classification",
    description="Classify anime frames by visual style",
    media_type="image",
    labels=["flat", "grim", "modern", "moe", "painterly", "retro"],
    label_colors={
        "flat": "#607D8B",
        "grim": "#37474F", 
        "modern": "#1976D2",
        "moe": "#C2185B",
        "painterly": "#7B1FA2",
        "retro": "#F57C00",
    },
    hint_field="predicted_style",
    hint_confidence_field="predicted_confidence",
    cluster_field="cluster_id",
    metadata_fields=["series_name", "production_year", "demographic"],
    garbage_classifier_path="/data/anime-scene-extraction/mobilevit_s_garbage.pth",
    garbage_threshold=0.7115,
)
