# Image Labeling Server (smart_label)

Fast start:
```bash
python -m smart_label ingest-folder --images /path/to/images --labels cat,dog,other
python -m smart_label serve --db labeling_queue.db --config labeling_task.json
# Open http://localhost:8765
```

A reusable image labeling server with:
- **Config-driven labels** - Use for any classification task via YAML/JSON config
- **Embedding-based clustering** - Diverse sample selection
- **Zero-friction web UI** - Keyboard shortcuts, auto-advance, image preloading
- **History review** - Browse and relabel previous decisions

## Fast Start (Folder + Class Names)

If you have a folder of images and class names, this is the fastest path:

```bash
# 1. Build a queue + config from a folder
python -m smart_label ingest-folder \
    --images /path/to/images \
    --labels cat,dog,other

# 2. Start the server
python -m smart_label serve --db labeling_queue.db --config labeling_task.json

# 3. Open http://localhost:8765 and start labeling
```

This creates:
- `labeling_queue.db` with all image paths
- `labeling_task.json` with your label list and task settings

## Quick Start (Clustering + Hints)

```bash
# 1. Prepare queue (select diverse samples with clustering)
python -m smart_label prepare --source ensemble_dataset_gold.json \
    --clusters 80 --samples-per-cluster 13 --output smart_label/queue.db

# 2. Launch server
python -m smart_label serve --db smart_label/queue.db

# 3. Open http://localhost:8765 and start labeling
```

## Custom Tasks

Create a config file to use Smart Label for any classification task:

```yaml
# my_task.yaml
name: "Object Classification"
description: "Label objects in images"
labels: ["car", "bike", "person", "building", "other"]
label_colors:
  car: "#1976D2"
  bike: "#4CAF50"
  person: "#C2185B"
  building: "#607D8B"
  other: "#757575"
db_path: "my_queue.db"
cluster_field: null  # Disable clustering if not needed
hint_field: null     # Disable predictions if not available
```

```bash
python -m smart_label serve --config my_task.yaml
```

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| 1-9 | Assign label (in order) |
| X / Space | Refuse (ambiguous) |
| Q | Bad quality |
| Z | Undo last |
| H | History |
| Esc | Close modal |

## Database Schema

```sql
CREATE TABLE queue (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,              -- Image file path
    cluster_id INTEGER,            -- Optional clustering
    predicted_style TEXT,          -- Optional model hint
    predicted_confidence REAL,
    human_label TEXT,              -- Set after labeling
    labeled_at TIMESTAMP,
    quality_flag TEXT,             -- 'BAD_QUALITY' if marked
    session_id TEXT
);
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config` | GET | Current labeling configuration |
| `/api/next` | GET | Next unlabeled image |
| `/api/label` | POST | Submit label for current image |
| `/api/stats` | GET | Label distribution stats |
| `/api/history` | GET | Paginated history (filter by label) |
| `/api/history/{id}/relabel` | POST | Change existing label |
| `/api/export` | GET | Download all labels as JSON |

## Configuration Options

```python
@dataclass
class LabelConfig:
    name: str                      # Task title
    labels: list                   # Valid labels (1-9 map to these)
    label_colors: dict             # Optional colors for UI
    db_path: str                   # SQLite queue database
    hint_field: Optional[str]      # Column with model prediction
    cluster_field: Optional[str]   # Column for cluster-based replacement
    metadata_fields: list          # Extra fields to show in UI
    garbage_classifier_path: str   # Optional quality classifier
```

## Output

```bash
# Export labels
python -m smart_label export --db smart_label/queue.db --output labels.json

# Format: [{"path": "/path/to/img.jpg", "label": "modern"}, ...]
```
