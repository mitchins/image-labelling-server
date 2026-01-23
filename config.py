"""
Configuration system for Smart Label.

Allows customizing labels, image sources, and optional features per task.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import yaml


@dataclass
class LabelConfig:
    """Configuration for a labeling task."""
    
    # Task identity
    name: str = "Labeling Task"
    description: str = ""
    
    # Labels: list of valid label names (order determines keyboard shortcuts 1-9)
    labels: list = field(default_factory=lambda: ["label_1", "label_2", "label_3"])
    
    # Label display colors (optional, uses defaults if not provided)
    label_colors: dict = field(default_factory=dict)
    
    # Database path
    db_path: str = "queue.db"
    
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
            "labels": self.labels,
            "label_colors": self.label_colors,
            "db_path": self.db_path,
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
