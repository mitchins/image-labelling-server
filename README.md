# Media Labeling Server (smart_label)

Fast, configurable web UI for labeling images or short audio clips with keyboard shortcuts.

## Read This First: Repo Layout and How to Run

The repository includes the installable `smart_label` package and can be used from
any working directory after installation. A source checkout can also be run directly
from its parent directory.

### Option A: Run from the parent directory (no install)

```bash
cd /data/anime-scene-extraction
python -m smart_label --help
python -m smart_label serve --db /path/to/queue.db
```

### Option B: Install the package (recommended for Linux and production use)

```bash
cd /data/anime-scene-extraction/smart_label
python -m pip install .
python -m smart_label serve --db /path/to/queue.db

# Optional console script if installed:
# image-labeling-server --db /path/to/queue.db
```

If you see `ModuleNotFoundError: No module named smart_label`, install the package
from the repository root with `python -m pip install .`.

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

### 1b) Folder of audio clips

```bash
python -m smart_label ingest-audio \
  --audio /path/to/clips \
  --labels positive,negative,neutral

python -m smart_label serve --db labeling_queue.db --config labeling_task.json
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
- Use `--media-type audio` when the JSONL rows point to audio clips.
- Optional hint fields: `predicted_style`, `predicted_confidence`.
- Optional cluster field: `cluster_id` (enables replacement on refuse).

### 2b) Confirm candidates across an ontology

Use one queue when every item already has an indicative ontology value and the
reviewer should judge how strongly it matches.

```json
{"path":"/audio/001.wav","indicative_value":"DREAD","narration_text":"Something moved beyond the door."}
```

```yaml
# ontology.yaml
id: prosody-register
version: v1
ontology:
  - {id: SUSPENSE, display_name: Suspense}
  - {id: DREAD, display_name: Foreboding}
  - {id: URGENCY, display_name: Urgency}
  - {id: SOMBRE, display_name: Sombre}
  - {id: WRY, display_name: Wry}
  - {id: BRIGHT, display_name: Bright}
```

```bash
python -m smart_label ingest-jsonl \
  --jsonl candidates.jsonl \
  --mode ontology-confirmation \
  --ontology ontology.yaml \
  --media-type audio \
  --metadata-fields narration_text
```

Confirmation mode uses fixed outcomes: `1` strong, `2` loose, `3` no match,
and `4` invalid item. Exports retain `ontology_id`, `ontology_version`,
`indicative_value`, and `confirmation`; downstream code decides whether to map
these to values such as `YES`, `KINDA`, and `NO`.

### 3) Already have a queue DB

```bash
python -m smart_label serve --db /path/to/queue.db --config /path/to/task.json
```

`serve` uses `--db` when provided, otherwise it falls back to `db_path` in the config.

### 4) Rank supplied candidate sets for one criterion

Ranking mode records an ordinal best-first order, not scores. Each JSONL line is
one atomic pairwise or setwise comparison with 2–8 candidates:

```json
{"set_id":"sentiment-001","criterion":{"id":"positive-sentiment","version":"v1","prompt":"Which clip sounds most positive?","direction":"most"},"metadata":{"split":"train"},"candidates":[{"candidate_id":"a","path":"/audio/a.wav","media_type":"audio","metadata":{"text":"Fine."}},{"candidate_id":"b","path":"/audio/b.wav","media_type":"audio","metadata":{"text":"Wonderful!"}}]}
```

```bash
python -m smart_label ingest-ranking \
  --jsonl sentiment-sets.jsonl \
  --db sentiment-ranking.db \
  --config sentiment-ranking.json

python -m smart_label serve --db sentiment-ranking.db --config sentiment-ranking.json
```

Card numbers are fixed display positions. For pairs, `1` or `2` selects the
winner and submits immediately. For larger sets, press card numbers in best-first
order, `Backspace` to remove the last draft choice, and `Enter` to submit. `X`
marks the whole set invalid, `R` replays audio, and `Z` appends an auditable undo.
Ties, partial orders, scores, active pair generation, and model fitting are
deliberately outside this mode.

## Configure Any Task

Create a config file to use Smart Label for any classification task:

```yaml
# my_task.yaml
name: "Object Classification"
description: "Label objects in images"
media_type: "image"               # or "audio"
audio_autoplay_default: false      # Initial value for single-item audio review
audio_autoplay_persistence: session # "session" or persistent "cookie"
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
| Q | Bad quality (image tasks) |
| R | Replay audio |
| Z | Undo last |
| H | History |
| Esc | Close modal |

In ontology confirmation mode, `1–4` are always `STRONG`, `LOOSE`, `NONE`, and
`INVALID`; `X` and Space are deliberately disabled.

## Database Schema

```sql
CREATE TABLE queue (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE,              -- Media file path
    media_type TEXT NOT NULL,      -- 'image' or 'audio'
    cluster_id INTEGER,            -- Optional clustering
    predicted_style TEXT,          -- Optional model hint
    predicted_confidence REAL,
    human_label TEXT,              -- Set after labeling
    labeled_at TIMESTAMP,
    quality_flag TEXT,             -- 'BAD_QUALITY' if marked
    session_id TEXT,
    indicative_value TEXT,         -- Proposed ontology ID (confirmation mode)
    confirmation TEXT,             -- STRONG/LOOSE/NONE/INVALID
    confirmation_at TIMESTAMP,
    confirmation_session_id TEXT
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
| `/api/next` | GET | Next unlabeled item |
| `/api/label` | POST | Submit label for current item |
| `/api/media/{id}` | GET | Fetch image/audio file |
| `/api/stats` | GET | Label distribution stats |
| `/api/history` | GET | Paginated history (filter by label) |
| `/api/history/{id}/relabel` | POST | Change existing label |
| `/api/export` | GET | Download all labels as JSON |

Ranking mode also exposes `/api/rank`, `/api/ranking/media`, and
`/api/history/rerank`. Ranking submissions use a request ID and expected revision
so retries are idempotent and stale tabs cannot overwrite newer work. Exports
retain typed metadata, source/display/rank positions, current state, and every
append-only revision; downstream ML code chooses how to fit a ranking model.

## Export

```bash
python -m smart_label export --db /path/to/queue.db --output labels.json
# Format: [{"path": "/path/to/img.jpg", "label": "modern"}, ...]
```

Confirmation exports use a versioned envelope with `ontology` and `items`
members. Each item retains its source metadata, indicative value, confirmation,
timestamp, and review session; no `YES`/`KINDA`/`NO` conversion is applied.
