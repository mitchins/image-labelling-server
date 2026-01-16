#!/usr/bin/env python3
"""
Prepare diverse sample queue for human labeling.

1. Embed all frames using CLIP
2. Cluster into ~80 semantic groups
3. Balance sample selection with pre-classification
4. Store queue in SQLite database
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
from tqdm import tqdm

# Lazy imports for heavy dependencies
torch = None
clip_model = None
clip_processor = None
classifier_model = None


def lazy_load_clip():
    """Lazy load CLIP model to avoid startup delay."""
    global torch, clip_model, clip_processor
    if torch is None:
        import torch as _torch
        torch = _torch
    if clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        print("Loading CLIP model (ViT-L/14)...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        if torch.cuda.is_available():
            clip_model = clip_model.cuda()
        clip_model.eval()
    return clip_model, clip_processor


def lazy_load_classifier(model_path: str, backbone: str = 'efficientnet_b0'):
    """Lazy load classifier for pre-classification.
    
    Args:
        model_path: Either a local file path (.pth) or HuggingFace model ID (org/repo)
        backbone: Model architecture (efficientnet_b0, efficientnetv2_s)
    """
    global torch, classifier_model
    if torch is None:
        import torch as _torch
        torch = _torch
    if classifier_model is None:
        import timm
        print(f"Loading classifier: {backbone}...")
        if backbone == 'efficientnet_b0':
            classifier_model = timm.create_model('tf_efficientnet_b0', pretrained=False, num_classes=6)
        elif backbone == 'efficientnetv2_s':
            classifier_model = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=6)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Check if model_path is a HuggingFace model ID or local file
        if '/' in model_path and not model_path.endswith('.pth'):
            # Load from HuggingFace
            print(f"Loading from HuggingFace: {model_path}...")
            from safetensors.torch import load_file as load_safetensors
            try:
                # Try loading safetensors format first (preferred)
                from huggingface_hub import hf_hub_download
                model_file = hf_hub_download(repo_id=model_path, filename='model.safetensors')
                state_dict = load_safetensors(model_file)
            except Exception:
                # Fallback to pytorch_model.bin
                from huggingface_hub import hf_hub_download
                model_file = hf_hub_download(repo_id=model_path, filename='pytorch_model.bin')
                state_dict = torch.load(model_file, map_location='cpu')
        else:
            # Load from local file
            print(f"Loading from local path: {model_path}...")
            state_dict = torch.load(model_path, map_location='cpu')
        
        classifier_model.load_state_dict(state_dict)
        if torch.cuda.is_available():
            classifier_model = classifier_model.cuda()
        classifier_model.eval()
    return classifier_model


def embed_frames(frame_paths: list, batch_size: int = 32) -> np.ndarray:
    """Generate CLIP embeddings for all frames."""
    from PIL import Image
    
    model, processor = lazy_load_clip()
    device = next(model.parameters()).device
    
    embeddings = []
    
    for i in tqdm(range(0, len(frame_paths), batch_size), desc="Embedding frames"):
        batch_paths = frame_paths[i:i+batch_size]
        images = []
        valid_indices = []
        
        for j, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
                valid_indices.append(i + j)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
        
        if not images:
            continue
            
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            # Normalize embeddings
            outputs = outputs / outputs.norm(dim=-1, keepdim=True)
            embeddings.append(outputs.cpu().numpy())
    
    return np.vstack(embeddings)


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 80) -> np.ndarray:
    """Cluster embeddings using K-means."""
    from sklearn.cluster import MiniBatchKMeans
    
    print(f"Clustering {len(embeddings)} embeddings into {n_clusters} clusters...")
    
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=42,
        batch_size=1024,
        n_init=3
    )
    cluster_ids = kmeans.fit_predict(embeddings)
    
    # Report cluster sizes
    unique, counts = np.unique(cluster_ids, return_counts=True)
    print(f"Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}")
    
    return cluster_ids


def classify_frames(frame_paths: list, model_path: str, backbone: str = 'efficientnet_b0', 
                    batch_size: int = 32) -> list:
    """Pre-classify frames with weak classifier."""
    from PIL import Image
    from torchvision import transforms
    
    model = lazy_load_classifier(model_path, backbone)
    device = next(model.parameters()).device
    
    # Get appropriate image size
    if 'b0' in backbone:
        size = 224
    elif 'v2_s' in backbone:
        size = 300
    else:
        size = 224
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    styles = ['flat', 'grim', 'modern', 'moe', 'painterly', 'retro']
    predictions = []
    
    for i in tqdm(range(0, len(frame_paths), batch_size), desc="Pre-classifying"):
        batch_paths = frame_paths[i:i+batch_size]
        batch_tensors = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert('RGB')
                tensor = transform(img)
                batch_tensors.append(tensor)
            except Exception:
                # Use zeros for failed loads
                batch_tensors.append(torch.zeros(3, size, size))
        
        batch = torch.stack(batch_tensors).to(device)
        
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(dim=1)
            
            for pred, conf in zip(preds.cpu().numpy(), confs.cpu().numpy()):
                predictions.append({
                    'style': styles[pred],
                    'confidence': float(conf)
                })
    
    return predictions


def select_diverse_samples(
    frame_paths: list,
    cluster_ids: np.ndarray,
    predictions: list,
    samples_per_cluster: int = 13,
    max_total: int = 1024
) -> list:
    """
    Select diverse samples balancing clusters and predicted styles.
    
    Strategy:
    - From each cluster, try to get samples from each predicted style
    - This ensures we don't just get "obvious" examples of each style
    - We want diverse contexts for each style
    """
    styles = ['flat', 'grim', 'modern', 'moe', 'painterly', 'retro']
    
    # Group by cluster
    clusters = defaultdict(list)
    for idx, (path, cid, pred) in enumerate(zip(frame_paths, cluster_ids, predictions)):
        clusters[cid].append({
            'idx': idx,
            'path': path,
            'cluster_id': int(cid),
            'predicted_style': pred['style'],
            'predicted_confidence': pred['confidence']
        })
    
    selected = []
    
    # For each cluster, try to balance styles
    for cid in sorted(clusters.keys()):
        cluster_items = clusters[cid]
        
        # Group by predicted style within cluster
        by_style = defaultdict(list)
        for item in cluster_items:
            by_style[item['predicted_style']].append(item)
        
        # Try to get ~2 samples per style (12 total), plus 1 random
        samples_per_style = max(1, samples_per_cluster // len(styles))
        cluster_selected = []
        
        for style in styles:
            if style in by_style:
                # Sort by confidence, take diverse confidence levels
                items = sorted(by_style[style], key=lambda x: x['predicted_confidence'])
                # Take from different confidence levels
                n = min(samples_per_style, len(items))
                if n > 0:
                    indices = np.linspace(0, len(items)-1, n, dtype=int)
                    for i in indices:
                        cluster_selected.append(items[i])
        
        # If we haven't hit our target, add random samples
        remaining = set(range(len(cluster_items))) - {item['idx'] for item in cluster_selected if 'idx' in item}
        remaining_items = [cluster_items[i] for i in remaining if i < len(cluster_items)]
        
        while len(cluster_selected) < samples_per_cluster and remaining_items:
            item = random.choice(remaining_items)
            remaining_items.remove(item)
            cluster_selected.append(item)
        
        selected.extend(cluster_selected[:samples_per_cluster])
    
    # Shuffle and limit
    random.shuffle(selected)
    selected = selected[:max_total]
    
    print(f"Selected {len(selected)} samples from {len(clusters)} clusters")
    
    # Report predicted style distribution
    style_counts = defaultdict(int)
    for item in selected:
        style_counts[item['predicted_style']] += 1
    print("Predicted style distribution:")
    for style in styles:
        print(f"  {style}: {style_counts[style]}")
    
    return selected


def create_database(db_path: str, samples: list, embeddings: np.ndarray = None):
    """Create SQLite database with labeling queue."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Create tables
    cur.executescript("""
        DROP TABLE IF EXISTS queue;
        DROP TABLE IF EXISTS sessions;
        
        CREATE TABLE queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            cluster_id INTEGER,
            predicted_style TEXT,
            predicted_confidence REAL,
            embedding BLOB,
            human_label TEXT,
            labeled_at TIMESTAMP,
            session_id TEXT
        );
        
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_activity TIMESTAMP,
            labels_count INTEGER DEFAULT 0
        );
        
        CREATE INDEX idx_queue_unlabeled ON queue(human_label) WHERE human_label IS NULL;
        CREATE INDEX idx_queue_cluster ON queue(cluster_id);
    """)
    
    # Insert samples
    for sample in tqdm(samples, desc="Creating database"):
        embedding_blob = None
        if embeddings is not None and 'idx' in sample:
            embedding_blob = embeddings[sample['idx']].tobytes()
        
        cur.execute("""
            INSERT OR IGNORE INTO queue (path, cluster_id, predicted_style, predicted_confidence, embedding)
            VALUES (?, ?, ?, ?, ?)
        """, (
            sample['path'],
            sample['cluster_id'],
            sample['predicted_style'],
            sample['predicted_confidence'],
            embedding_blob
        ))
    
    conn.commit()
    
    # Report
    cur.execute("SELECT COUNT(*) FROM queue")
    total = cur.fetchone()[0]
    print(f"Created database with {total} samples at {db_path}")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description='Prepare diverse labeling queue')
    
    parser.add_argument('--source', required=True,
                        help='Source dataset JSON (e.g., ensemble_dataset_gold.json)')
    parser.add_argument('--output', default='smart_label/queue.db',
                        help='Output SQLite database path')
    parser.add_argument('--clusters', type=int, default=80,
                        help='Number of clusters for diversity sampling')
    parser.add_argument('--samples-per-cluster', type=int, default=13,
                        help='Samples per cluster (13 × 80 = 1040)')
    parser.add_argument('--max-samples', type=int, default=1024,
                        help='Maximum total samples')
    parser.add_argument('--classifier', default=None,
                        help='Path to classifier model for pre-classification (local .pth file or HF model ID like org/repo)')
    parser.add_argument('--backbone', default='efficientnet_b0',
                        choices=['efficientnet_b0', 'efficientnetv2_s'],
                        help='Classifier backbone')
    parser.add_argument('--embeddings-cache', default=None,
                        help='Path to cache/load embeddings (.npz)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for embedding/classification')
    parser.add_argument('--limit-source', type=int, default=None,
                        help='Limit source frames (for testing)')
    
    args = parser.parse_args()
    
    # Load source dataset
    print(f"Loading source dataset: {args.source}")
    with open(args.source) as f:
        data = json.load(f)
    
    frames = data['frames']
    if args.limit_source:
        random.shuffle(frames)
        frames = frames[:args.limit_source]
    
    frame_paths = [f['path'] for f in frames]
    print(f"Loaded {len(frame_paths)} frames")
    
    # Generate or load embeddings
    if args.embeddings_cache and Path(args.embeddings_cache).exists():
        print(f"Loading cached embeddings from {args.embeddings_cache}")
        cached = np.load(args.embeddings_cache, allow_pickle=True)
        
        # Handle different cache formats
        if 'embeddings_768d' in cached.files:
            embeddings = cached['embeddings_768d']
        elif 'embeddings' in cached.files:
            embeddings = cached['embeddings']
        else:
            raise ValueError(f"Unknown embedding key in cache. Available keys: {list(cached.files)}")
        
        cached_paths = cached['paths'].tolist()
        
        # Match paths
        path_to_idx = {p: i for i, p in enumerate(cached_paths)}
        valid_indices = []
        valid_paths = []
        for p in frame_paths:
            if p in path_to_idx:
                valid_indices.append(path_to_idx[p])
                valid_paths.append(p)
        
        embeddings = embeddings[valid_indices]
        frame_paths = valid_paths
        print(f"Matched {len(frame_paths)} paths from cache")
    else:
        embeddings = embed_frames(frame_paths, batch_size=args.batch_size)
        
        if args.embeddings_cache:
            print(f"Saving embeddings to {args.embeddings_cache}")
            Path(args.embeddings_cache).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                args.embeddings_cache,
                embeddings=embeddings,
                paths=np.array(frame_paths)
            )
    
    # Cluster
    cluster_ids = cluster_embeddings(embeddings, n_clusters=args.clusters)
    
    # Pre-classify (optional but recommended)
    if args.classifier:
        predictions = classify_frames(
            frame_paths, args.classifier, args.backbone, batch_size=args.batch_size
        )
    else:
        # Use uniform random if no classifier
        print("Warning: No classifier provided, using random style assignment")
        styles = ['flat', 'grim', 'modern', 'moe', 'painterly', 'retro']
        predictions = [{'style': random.choice(styles), 'confidence': 0.5} for _ in frame_paths]
    
    # Select diverse samples
    samples = select_diverse_samples(
        frame_paths, cluster_ids, predictions,
        samples_per_cluster=args.samples_per_cluster,
        max_total=args.max_samples
    )
    
    # Create database
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    create_database(args.output, samples, embeddings)
    
    print(f"\n✓ Queue ready at {args.output}")
    print(f"  Run: python -m smart_label.server --db {args.output}")


if __name__ == '__main__':
    main()
