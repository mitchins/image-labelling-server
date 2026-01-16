#!/usr/bin/env python3
"""
Smart Label - Configurable image labeling server.

A reusable FastAPI server for rapid human labeling of images.
Configure labels, hints, clustering, and metadata per task.

Usage:
    python -m smart_label.server --db queue.db [--config config.yaml]
"""

import argparse
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Lazy imports for optional heavy dependencies
torch = None
timm = None

# Global state (configured at startup)
DB_PATH: str = None
GARBAGE_CLASSIFIER = None
CONFIG = None  # LabelConfig instance

app = FastAPI(title="Smart Label", description="Configurable image labeling")

# Mount static files directory
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@contextmanager
def get_db():
    """Database connection context manager."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_progress_stats(cur):
    """Get progress stats - only counts actual labels (1-6), not REFUSE."""
    cur.execute("SELECT COUNT(*) FROM queue")
    total = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) FROM queue 
        WHERE human_label IS NOT NULL AND human_label != 'REFUSE'
    """)
    labeled = cur.fetchone()[0]
    return {
        "labeled": labeled,
        "total": total,
        "percent": round(100 * labeled / total, 1) if total > 0 else 0
    }


class LabelRequest(BaseModel):
    """Request to label an image."""
    image_id: int
    label: str  # One of the configured labels or REFUSE
    quality_flag: Optional[str] = None  # Optional flag (e.g., BAD_QUALITY)
    session_id: Optional[str] = None


class StatsResponse(BaseModel):
    """Labeling statistics."""
    total: int
    labeled: int
    remaining: int
    by_label: dict
    by_cluster: dict


def get_valid_labels() -> list:
    """Get list of valid labels including REFUSE."""
    if CONFIG:
        return CONFIG.labels + ['REFUSE']
    return ['REFUSE']


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def index():
    """Serve the main labeling UI."""
    static_dir = Path(__file__).parent / "static"
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    else:
        # Fallback inline HTML if static files don't exist
        return HTMLResponse(get_inline_html())


@app.get("/api/next")
async def get_next_image(session_id: str = Query(default=None)):
    """Get the next unlabeled image."""
    with get_db() as conn:
        cur = conn.cursor()
        
        # Get next unlabeled image (not REFUSE)
        cur.execute("""
            SELECT id, path, cluster_id, predicted_style, predicted_confidence, 
                   series_name, production_year, demographic
            FROM queue
            WHERE human_label IS NULL
            ORDER BY RANDOM()
            LIMIT 1
        """)
        row = cur.fetchone()
        
        if not row:
            return JSONResponse({"done": True, "message": "All images labeled!"})
        
        # Get stats (only count 1-6 labels, not REFUSE)
        progress = get_progress_stats(cur)
        
        return {
            "done": False,
            "id": row["id"],
            "path": row["path"],
            "cluster_id": row["cluster_id"],
            "predicted_style": row["predicted_style"],
            "predicted_confidence": row["predicted_confidence"],
            "series_name": row["series_name"],
            "production_year": row["production_year"],
            "demographic": row["demographic"],
            "progress": progress
        }


@app.get("/api/batch")
async def get_batch(count: int = Query(default=5, le=20)):
    """Get a batch of unlabeled images for preloading."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, path, cluster_id, predicted_style, predicted_confidence
            FROM queue
            WHERE human_label IS NULL
            ORDER BY RANDOM()
            LIMIT ?
        """, (count,))
        rows = cur.fetchall()
        
        return {
            "images": [
                {
                    "id": row["id"],
                    "path": row["path"],
                    "cluster_id": row["cluster_id"],
                    "predicted_style": row["predicted_style"],
                    "predicted_confidence": row["predicted_confidence"]
                }
                for row in rows
            ]
        }


@app.get("/api/replacement/{cluster_id}")
async def get_replacement(cluster_id: int):
    """Get a replacement frame from the same cluster (for rejections)."""
    with get_db() as conn:
        cur = conn.cursor()
        
        # Get an unlabeled frame from the same cluster
        cur.execute("""
            SELECT id, path, cluster_id, predicted_style, predicted_confidence, 
                   series_name, production_year, demographic
            FROM queue
            WHERE cluster_id = ? AND human_label IS NULL
            ORDER BY RANDOM()
            LIMIT 1
        """, (cluster_id,))
        row = cur.fetchone()
        
        if not row:
            # Cluster exhausted, return any unlabeled frame
            cur.execute("""
                SELECT id, path, cluster_id, predicted_style, predicted_confidence,
                       series_name, production_year, demographic
                FROM queue
                WHERE human_label IS NULL
                ORDER BY RANDOM()
                LIMIT 1
            """)
            row = cur.fetchone()
            
            if not row:
                return JSONResponse({"done": True, "message": "All images labeled!"})
        
        # Get stats (only count 1-6 labels, not REFUSE)
        progress = get_progress_stats(cur)
        
        return {
            "done": False,
            "id": row["id"],
            "path": row["path"],
            "cluster_id": row["cluster_id"],
            "predicted_style": row["predicted_style"],
            "predicted_confidence": row["predicted_confidence"],
            "series_name": row["series_name"],
            "production_year": row["production_year"],
            "demographic": row["demographic"],
            "progress": progress
        }



@app.post("/api/label")
async def label_image(request: LabelRequest):
    """Label an image."""
    valid_labels = get_valid_labels()
    if request.label not in valid_labels:
        raise HTTPException(status_code=400, detail=f"Invalid label: {request.label}. Valid: {valid_labels}")
    
    session_id = request.session_id or str(uuid.uuid4())
    
    with get_db() as conn:
        cur = conn.cursor()
        
        # Update label and quality flag
        cur.execute("""
            UPDATE queue 
            SET human_label = ?, quality_flag = ?, labeled_at = ?, session_id = ?
            WHERE id = ?
        """, (request.label, request.quality_flag, datetime.now().isoformat(), session_id, request.image_id))
        
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Update session
        cur.execute("""
            INSERT INTO sessions (id, last_activity, labels_count)
            VALUES (?, ?, 1)
            ON CONFLICT(id) DO UPDATE SET
                last_activity = excluded.last_activity,
                labels_count = labels_count + 1
        """, (session_id, datetime.now().isoformat()))
        
        conn.commit()
        
        # Get updated stats (only count 1-6 labels, not REFUSE)
        progress = get_progress_stats(cur)
        
        return {
            "success": True,
            "session_id": session_id,
            "progress": progress
        }


@app.get("/api/stats")
async def get_stats():
    """Get labeling statistics."""
    with get_db() as conn:
        cur = conn.cursor()
        
        cur.execute("SELECT COUNT(*) FROM queue")
        total = cur.fetchone()[0]
        
        cur.execute("SELECT COUNT(*) FROM queue WHERE human_label IS NOT NULL")
        labeled = cur.fetchone()[0]
        
        # By label
        cur.execute("""
            SELECT human_label, COUNT(*) as count
            FROM queue
            WHERE human_label IS NOT NULL
            GROUP BY human_label
        """)
        by_label = {row["human_label"]: row["count"] for row in cur.fetchall()}
        
        # By cluster (labeled count)
        cur.execute("""
            SELECT cluster_id, COUNT(*) as count
            FROM queue
            WHERE human_label IS NOT NULL
            GROUP BY cluster_id
            ORDER BY cluster_id
        """)
        by_cluster = {row["cluster_id"]: row["count"] for row in cur.fetchall()}
        
        return StatsResponse(
            total=total,
            labeled=labeled,
            remaining=total - labeled,
            by_label=by_label,
            by_cluster=by_cluster
        )


@app.get("/api/image/{image_id}")
async def get_image(image_id: int):
    """Get image file by database ID."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT path FROM queue WHERE id = ?", (image_id,))
        row = cur.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Image not found")
        
        path = Path(row["path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Image file not found: {path}")
        
        return FileResponse(path, media_type="image/jpeg")


@app.get("/api/export")
async def export_labels():
    """Export all labeled data as JSON."""
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT path, human_label, cluster_id, predicted_style, labeled_at
            FROM queue
            WHERE human_label IS NOT NULL
            ORDER BY labeled_at
        """)
        
        labels = [
            {
                "path": row["path"],
                "label": row["human_label"],
                "cluster_id": row["cluster_id"],
                "predicted_style": row["predicted_style"],
                "labeled_at": row["labeled_at"]
            }
            for row in cur.fetchall()
        ]
        
        return JSONResponse(labels)


@app.get("/api/config")
async def get_config():
    """Get current labeling configuration."""
    if CONFIG:
        return JSONResponse(CONFIG.to_dict())
    # Fallback for backward compatibility
    return JSONResponse({
        "name": "Smart Label",
        "labels": ["flat", "grim", "modern", "moe", "painterly", "retro"],
        "hint_field": "predicted_style",
        "cluster_field": "cluster_id",
    })


@app.get("/api/history")
async def get_history(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    label_filter: Optional[str] = Query(default=None)
):
    """Get labeling history for review, newest first."""
    with get_db() as conn:
        cur = conn.cursor()
        
        offset = (page - 1) * per_page
        
        # Build query with optional label filter
        where_clause = "WHERE human_label IS NOT NULL"
        params = []
        
        if label_filter:
            where_clause += " AND human_label = ?"
            params.append(label_filter)
        
        # Get total count
        cur.execute(f"SELECT COUNT(*) FROM queue {where_clause}", params)
        total = cur.fetchone()[0]
        
        # Get page of results
        cur.execute(f"""
            SELECT id, path, human_label, cluster_id, predicted_style, 
                   predicted_confidence, labeled_at, quality_flag
            FROM queue
            {where_clause}
            ORDER BY labeled_at DESC
            LIMIT ? OFFSET ?
        """, params + [per_page, offset])
        
        items = [
            {
                "id": row["id"],
                "path": row["path"],
                "label": row["human_label"],
                "cluster_id": row["cluster_id"],
                "predicted_style": row["predicted_style"],
                "predicted_confidence": row["predicted_confidence"],
                "labeled_at": row["labeled_at"],
                "quality_flag": row["quality_flag"],
            }
            for row in cur.fetchall()
        ]
        
        return {
            "items": items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page if total > 0 else 1
        }


@app.post("/api/history/{image_id}/relabel")
async def relabel_from_history(image_id: int, request: LabelRequest):
    """Change label for a previously labeled image."""
    valid_labels = get_valid_labels()
    if request.label not in valid_labels:
        raise HTTPException(status_code=400, detail=f"Invalid label: {request.label}")
    
    with get_db() as conn:
        cur = conn.cursor()
        
        # Check image exists and was labeled
        cur.execute("SELECT human_label FROM queue WHERE id = ?", (image_id,))
        row = cur.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Image not found")
        
        if row["human_label"] is None:
            raise HTTPException(status_code=400, detail="Image was not previously labeled")
        
        # Update label
        cur.execute("""
            UPDATE queue 
            SET human_label = ?, quality_flag = ?, labeled_at = ?
            WHERE id = ?
        """, (request.label, request.quality_flag, datetime.now().isoformat(), image_id))
        
        conn.commit()
        
        return {"success": True, "image_id": image_id, "new_label": request.label}


@app.get("/api/garbage-rating/{image_id}")
async def get_garbage_rating(image_id: int):
    """Get garbage quality rating for an image (0=good, 1=garbage)."""
    global GARBAGE_CLASSIFIER
    
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT path FROM queue WHERE id = ?", (image_id,))
        row = cur.fetchone()
        
        if not row:
            raise HTTPException(status_code=404, detail="Image not found")
        
        path = Path(row["path"])
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Image file not found: {path}")
    
    try:
        # Lazy import torch dependencies
        global torch
        if torch is None:
            import torch as _torch
            torch = _torch
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Load and classify image
        if GARBAGE_CLASSIFIER is None:
            raise RuntimeError("Garbage classifier not initialized")
        
        image = Image.open(path).convert('RGB')
        
        # Prepare image (MobileViT-S expects 256x256)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        # Classify
        with torch.no_grad():
            device = next(GARBAGE_CLASSIFIER.parameters()).device
            image_tensor = image_tensor.to(device)
            logits = GARBAGE_CLASSIFIER(image_tensor)
            probs = torch.softmax(logits, dim=1)
            garbage_prob = probs[0, 0].item()  # Class 0 = garbage, Class 1 = quality
            quality_prob = probs[0, 1].item()
        
        # Use config threshold or default
        threshold = CONFIG.garbage_threshold if CONFIG else 0.7115
        is_garbage = garbage_prob > threshold
        
        return {
            "image_id": image_id,
            "garbage_score": round(garbage_prob, 3),
            "quality": "garbage" if is_garbage else "good",
            "confidence": round(quality_prob * 100, 2) if not is_garbage else round(garbage_prob * 100, 2)
        }
    
    except Exception as e:
        # If classification fails, return neutral
        return {
            "image_id": image_id,
            "garbage_score": 0.5,
            "quality": "unknown",
            "confidence": 0.0,
            "error": str(e)
        }

@app.post("/api/undo")
async def undo_last():
    """Undo the most recent label."""
    with get_db() as conn:
        cur = conn.cursor()
        
        # Find most recent
        cur.execute("""
            SELECT id, path FROM queue
            WHERE human_label IS NOT NULL
            ORDER BY labeled_at DESC
            LIMIT 1
        """)
        row = cur.fetchone()
        
        if not row:
            return {"success": False, "message": "Nothing to undo"}
        
        # Clear it
        cur.execute("""
            UPDATE queue
            SET human_label = NULL, labeled_at = NULL, session_id = NULL
            WHERE id = ?
        """, (row["id"],))
        conn.commit()
        
        return {"success": True, "undone_id": row["id"], "path": row["path"]}


# ============================================================================
# Inline HTML (fallback if static files don't exist)
# ============================================================================

def get_inline_html():
    """Return inline HTML for the labeling UI."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Label - Anime Style Classifier</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            padding: 10px 20px;
            background: #16213e;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .progress-bar {
            width: 300px;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s;
        }
        .progress-text {
            font-size: 14px;
            color: #aaa;
        }
        .main {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .image-container {
            max-width: 90vw;
            max-height: 60vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .image-container img {
            max-width: 100%;
            max-height: 60vh;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .prediction {
            margin: 10px 0;
            font-size: 14px;
            color: #888;
        }
        .buttons {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 20px;
        }
        .btn {
            padding: 15px 25px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.1s, box-shadow 0.2s;
            min-width: 100px;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }
        .btn:active {
            transform: translateY(0);
        }
        .btn-flat { background: #607D8B; color: white; }
        .btn-grim { background: #455A64; color: white; }
        .btn-modern { background: #2196F3; color: white; }
        .btn-moe { background: #E91E63; color: white; }
        .btn-painterly { background: #9C27B0; color: white; }
        .btn-retro { background: #FF9800; color: white; }
        .btn-refuse { background: #f44336; color: white; }
        .shortcut {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            margin-left: 5px;
        }
        .help {
            position: fixed;
            bottom: 20px;
            left: 20px;
            font-size: 12px;
            color: #666;
        }
        .done {
            text-align: center;
            padding: 50px;
        }
        .done h1 { color: #4CAF50; margin-bottom: 20px; }
        .loading {
            color: #888;
            font-size: 18px;
        }
        .undo-btn {
            position: fixed;
            top: 10px;
            right: 20px;
            background: #333;
            color: #aaa;
            border: 1px solid #555;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
        }
        .undo-btn:hover { background: #444; }
    </style>
</head>
<body>
    <div class="header">
        <h2>🎨 Smart Label</h2>
        <div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill" style="width: 0%"></div>
            </div>
            <div class="progress-text" id="progressText">Loading...</div>
        </div>
    </div>
    
    <button class="undo-btn" onclick="undoLast()">↩ Undo (Z)</button>
    
    <div class="main" id="main">
        <div class="loading">Loading...</div>
    </div>
    
    <div class="help">
        Keyboard: 1=flat 2=grim 3=modern 4=moe 5=painterly 6=retro X/Space=refuse Z=undo
    </div>

    <script>
        let currentImage = null;
        let preloadedImages = [];
        let sessionId = localStorage.getItem('sessionId') || crypto.randomUUID();
        localStorage.setItem('sessionId', sessionId);
        
        const STYLES = ['flat', 'grim', 'modern', 'moe', 'painterly', 'retro'];
        const KEY_MAP = {
            '1': 'flat', '2': 'grim', '3': 'modern',
            '4': 'moe', '5': 'painterly', '6': 'retro',
            'x': 'REFUSE', ' ': 'REFUSE'
        };
        
        async function loadNext() {
            const res = await fetch('/api/next?session_id=' + sessionId);
            const data = await res.json();
            
            if (data.done) {
                showDone();
                return;
            }
            
            currentImage = data;
            updateProgress(data.progress);
            renderImage(data);
            preloadNext();
        }
        
        async function preloadNext() {
            const res = await fetch('/api/batch?count=3');
            const data = await res.json();
            preloadedImages = data.images;
            
            // Preload images into browser cache
            preloadedImages.forEach(img => {
                const preload = new Image();
                preload.src = '/api/image/' + img.id;
            });
        }
        
        function renderImage(data) {
            const main = document.getElementById('main');
            main.innerHTML = `
                <div class="image-container">
                    <img src="/api/image/${data.id}" alt="Frame to label">
                </div>
                <div class="prediction">
                    Predicted: <strong>${data.predicted_style}</strong> 
                    (${Math.round(data.predicted_confidence * 100)}% confidence)
                    | Cluster ${data.cluster_id}
                </div>
                <div class="buttons">
                    ${STYLES.map((s, i) => `
                        <button class="btn btn-${s}" onclick="label('${s}')">
                            ${s} <span class="shortcut">${i+1}</span>
                        </button>
                    `).join('')}
                    <button class="btn btn-refuse" onclick="label('REFUSE')">
                        Refuse <span class="shortcut">X</span>
                    </button>
                </div>
            `;
        }
        
        function updateProgress(progress) {
            document.getElementById('progressFill').style.width = progress.percent + '%';
            document.getElementById('progressText').textContent = 
                `${progress.labeled} / ${progress.total} (${progress.percent}%)`;
        }
        
        function showDone() {
            document.getElementById('main').innerHTML = `
                <div class="done">
                    <h1>🎉 All Done!</h1>
                    <p>You've labeled all ${currentImage?.progress?.total || 'the'} images.</p>
                    <p style="margin-top: 20px;">
                        <a href="/api/export" style="color: #4CAF50;">Download Labels (JSON)</a>
                    </p>
                </div>
            `;
        }
        
        async function label(style) {
            if (!currentImage) return;
            
            const res = await fetch('/api/label', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    image_id: currentImage.id,
                    label: style,
                    session_id: sessionId
                })
            });
            
            const data = await res.json();
            if (data.success) {
                updateProgress(data.progress);
                loadNext();
            }
        }
        
        async function undoLast() {
            const res = await fetch('/api/undo', {method: 'POST'});
            const data = await res.json();
            if (data.success) {
                loadNext();
            }
        }
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;
            
            const key = e.key.toLowerCase();
            if (key === 'z') {
                undoLast();
            } else if (KEY_MAP[key]) {
                label(KEY_MAP[key]);
            }
        });
        
        // Start
        loadNext();
    </script>
</body>
</html>
"""


# ============================================================================
# CLI Entry Point
# ============================================================================

def load_garbage_classifier_model(classifier_path: str):
    """Load garbage classifier if available."""
    global torch, GARBAGE_CLASSIFIER
    
    if not classifier_path or not Path(classifier_path).exists():
        print("⚠ Garbage classifier not configured or not found")
        return
    
    try:
        import torch as _torch
        import timm
        torch = _torch
        
        print(f"Loading garbage classifier from {classifier_path}...")
        GARBAGE_CLASSIFIER = timm.create_model('mobilevit_s', num_classes=2, pretrained=False)
        GARBAGE_CLASSIFIER.load_state_dict(torch.load(classifier_path, map_location='cpu'))
        GARBAGE_CLASSIFIER.eval()
        
        if torch.cuda.is_available():
            GARBAGE_CLASSIFIER = GARBAGE_CLASSIFIER.cuda()
        
        print("✓ Garbage classifier loaded")
    except Exception as e:
        print(f"⚠ Failed to load garbage classifier: {e}")


def main():
    parser = argparse.ArgumentParser(description='Smart Label - Configurable image labeling server')
    parser.add_argument('--db', required=True, help='Path to queue database')
    parser.add_argument('--config', default=None, help='Path to config YAML/JSON (optional)')
    parser.add_argument('--port', type=int, default=8765, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--garbage-classifier', default=None, 
                        help='Path to garbage classifier model (optional)')
    
    args = parser.parse_args()
    
    global DB_PATH, CONFIG
    DB_PATH = args.db
    
    if not Path(DB_PATH).exists():
        print(f"Error: Database not found: {DB_PATH}")
        print("Run prepare.py first to create the queue database")
        return 1
    
    # Load configuration
    if args.config:
        from .config import LabelConfig
        CONFIG = LabelConfig.from_file(args.config)
        print(f"✓ Loaded config: {CONFIG.name}")
    else:
        # Default anime style config for backward compatibility
        from .config import ANIME_STYLE_CONFIG
        CONFIG = ANIME_STYLE_CONFIG
        print("✓ Using default anime style config")
    
    # Override garbage classifier from CLI if provided
    classifier_path = args.garbage_classifier or CONFIG.garbage_classifier_path
    load_garbage_classifier_model(classifier_path)
    
    # Print startup info
    print(f"\n{'='*50}")
    print(f"Smart Label Server")
    print(f"{'='*50}")
    print(f"Task: {CONFIG.name}")
    print(f"Database: {DB_PATH}")
    print(f"URL: http://localhost:{args.port}")
    print(f"\nLabels: {', '.join(CONFIG.labels)}")
    print(f"Shortcuts: {' '.join(f'{i+1}={l}' for i, l in enumerate(CONFIG.labels[:9]))}")
    print(f"           X/Space=refuse  Z=undo  Q=bad quality")
    print(f"{'='*50}\n")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
