# Image Labeling Server (smart_label)

Fast, configurable web UI for labeling images with keyboard shortcuts.

## Read This First: Repo Layout and How to Run

This repo root **is** the Python package. There is no nested `smart_label/` folder.
That means:

- If you run commands from the **parent directory**, `python -m smart_label ...` works.
- If you run commands from **inside this repo**, install it first (`pip install -e .`).

### Option A: Run from the parent directory (no install)

```bash
cd /data/anime-scene-extraction
python -m smart_label --help
python -m smart_label serve --db /path/to/queue.db
```

### Option B: Install editable (recommended for most users)

```bash
cd /data/anime-scene-extraction/smart_label
python -m pip install -e .
python -m smart_label serve --db /path/to/queue.db

# Optional console script if installed:
# image-labeling-server --db /path/to/queue.db
```

If you see `ModuleNotFoundError: No module named smart_label`, you are either:
- in the wrong folder (run from the parent directory), or
- missing the editable install (`pip install -e .`).

## Labels and Config (Single Source of Truth)

Labels live in the **database** (`labels` table). Ingest commands populate them.
`serve` will read labels from the DB by default. A config file is optional and
only needed to override labels/metadata or customize UI colors.

If you pass `--config`, those labels override the DB.

## Quick Start (Pick One Ingestion Path)

### 1) Folder of images (fastest path)

```bash
python -m smart_label ingest-folder \
  --images /path/to/images \
  --labels cat,dog,other

python -m smart_label serve --db labeling_queue.db --config labeling_task.json
# Open http://localhost:8765
```

### 2) JSONL list (paths + metadata + hints)

Example JSONL line:

```json
{"path": "/data/images/img_001.jpg", "cluster_id": 12, "predicted_style": "modern", "predicted_confidence": 0.87, "series_name": "Show A", "production_year": 2018}
```

Ingest:

```bash
python -m smart_label ingest-jsonl \
  --jsonl /path/to/data.jsonl \
  --labels cat,dog,other \
  --metadata-fields series_name,production_year

python -m smart_label serve --db labeling_queue.db --config labeling_task.json
```

Notes:
- `path` is required (use `--path-field` if your JSONL uses another key).
- Relative paths can be resolved with `--base-dir /path/to/images`.
- Optional hint fields: `predicted_style`, `predicted_confidence`.
- Optional cluster field: `cluster_id` (enables replacement on refuse).

### 3) Already have a queue DB

```bash
python -m smart_label serve --db /path/to/queue.db --config /path/to/task.json
```

`serve` uses `--db` when provided, otherwise it falls back to `db_path` in the config.

## Configure Any Task

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
cluster_field: "cluster_id"        # Set null to disable clustering
hint_field: "predicted_style"      # Set null to disable hints
hint_confidence_field: "predicted_confidence"
metadata_fields: ["series_name", "production_year"]
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
    -- plus any metadata columns you configured
);

CREATE TABLE labels (
    name TEXT PRIMARY KEY,
    color TEXT,
    sort_order INTEGER
);

CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT
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

## Export

```bash
python -m smart_label export --db /path/to/queue.db --output labels.json
# Format: [{"path": "/path/to/img.jpg", "label": "modern"}, ...]
```
