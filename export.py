#!/usr/bin/env python3
"""
Export utilities for smart_label.

Convert labeled data to formats compatible with train_unified.py
"""

import argparse
import json
from pathlib import Path
from collections import Counter


def to_training_dataset(labels_path: str, output_path: str, exclude_refuse: bool = True):
    """
    Convert smart_label exports to training dataset format.
    
    Output format matches ensemble_dataset_*.json:
    {
        "dataset": "human_labeled",
        "description": "Human-labeled diverse sample set",
        "frames": [
            {"path": "...", "label": "modern"},
            ...
        ]
    }
    """
    with open(labels_path) as f:
        labels = json.load(f)
    
    frames = []
    refused = 0
    
    for item in labels:
        if item['label'] == 'REFUSE':
            refused += 1
            if exclude_refuse:
                continue
        
        frames.append({
            'path': item['path'],
            'label': item['label'],
            'cluster_id': item.get('cluster_id'),
            'source': 'human_labeled'
        })
    
    dataset = {
        'dataset': 'human_labeled',
        'description': f'Human-labeled diverse sample set from smart_label ({len(frames)} frames)',
        'source_file': labels_path,
        'frames': frames
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Report
    print(f"Created training dataset: {output_path}")
    print(f"  Total frames: {len(frames)}")
    if refused > 0:
        print(f"  Refused (excluded): {refused}")
    
    dist = Counter(f['label'] for f in frames)
    print("\nLabel distribution:")
    for label in ['flat', 'grim', 'modern', 'moe', 'painterly', 'retro']:
        count = dist.get(label, 0)
        pct = 100 * count / len(frames) if frames else 0
        print(f"  {label}: {count} ({pct:.1f}%)")


def merge_with_existing(human_labels_path: str, existing_dataset_path: str, 
                        output_path: str, human_weight: float = 1.0):
    """
    Merge human labels with existing ensemble dataset.
    
    Human labels can be weighted (duplicated) to increase their influence.
    """
    with open(human_labels_path) as f:
        human = json.load(f)
    
    with open(existing_dataset_path) as f:
        existing = json.load(f)
    
    # Collect frames
    frames = []
    
    # Add existing frames
    for frame in existing.get('frames', []):
        frame['source'] = 'ensemble'
        frames.append(frame)
    
    # Add human frames (potentially multiple times for weighting)
    human_frames = [f for f in human if f.get('label') != 'REFUSE']
    repetitions = max(1, int(human_weight))
    
    for _ in range(repetitions):
        for item in human_frames:
            frames.append({
                'path': item['path'],
                'label': item['label'],
                'source': 'human_labeled'
            })
    
    # Create merged dataset
    dataset = {
        'dataset': 'merged_human_ensemble',
        'description': f'Merged human labels ({len(human_frames)}×{repetitions}) with ensemble ({len(existing.get("frames", []))})',
        'frames': frames
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Created merged dataset: {output_path}")
    print(f"  Ensemble frames: {len(existing.get('frames', []))}")
    print(f"  Human frames: {len(human_frames)} × {repetitions} = {len(human_frames) * repetitions}")
    print(f"  Total: {len(frames)}")


def analyze_disagreements(labels_path: str):
    """
    Analyze where human labels disagree with model predictions.
    """
    with open(labels_path) as f:
        labels = json.load(f)
    
    agreements = 0
    disagreements = []
    refused = 0
    
    for item in labels:
        if item['label'] == 'REFUSE':
            refused += 1
            continue
        
        if item['label'] == item.get('predicted_style'):
            agreements += 1
        else:
            disagreements.append({
                'path': item['path'],
                'human': item['label'],
                'predicted': item.get('predicted_style'),
                'cluster_id': item.get('cluster_id')
            })
    
    total = agreements + len(disagreements)
    
    print(f"Agreement Analysis")
    print(f"==================")
    print(f"Total labeled: {total + refused}")
    print(f"  Agreed: {agreements} ({100*agreements/total:.1f}%)")
    print(f"  Disagreed: {len(disagreements)} ({100*len(disagreements)/total:.1f}%)")
    print(f"  Refused: {refused}")
    print()
    
    if disagreements:
        print("Disagreement matrix (Human → Predicted):")
        matrix = Counter((d['human'], d['predicted']) for d in disagreements)
        
        styles = ['flat', 'grim', 'modern', 'moe', 'painterly', 'retro']
        print("         ", "  ".join(f"{s[:4]:>4}" for s in styles))
        for h in styles:
            row = [matrix.get((h, p), 0) for p in styles]
            print(f"{h:>8}:", "  ".join(f"{c:>4}" for c in row))


def main():
    parser = argparse.ArgumentParser(description='Smart Label export utilities')
    
    subparsers = parser.add_subparsers(dest='command')
    
    # To dataset
    to_ds = subparsers.add_parser('to-dataset', help='Convert to training dataset')
    to_ds.add_argument('--labels', required=True, help='Labels JSON from export')
    to_ds.add_argument('--output', required=True, help='Output dataset JSON')
    to_ds.add_argument('--include-refuse', action='store_true', help='Include REFUSE labels')
    
    # Merge
    merge = subparsers.add_parser('merge', help='Merge with existing dataset')
    merge.add_argument('--human', required=True, help='Human labels JSON')
    merge.add_argument('--existing', required=True, help='Existing dataset JSON')
    merge.add_argument('--output', required=True, help='Output merged dataset')
    merge.add_argument('--human-weight', type=float, default=1.0, 
                       help='Weight for human labels (>1 = duplicate)')
    
    # Analyze
    analyze = subparsers.add_parser('analyze', help='Analyze human vs model disagreements')
    analyze.add_argument('--labels', required=True, help='Labels JSON from export')
    
    args = parser.parse_args()
    
    if args.command == 'to-dataset':
        to_training_dataset(args.labels, args.output, not args.include_refuse)
    elif args.command == 'merge':
        merge_with_existing(args.human, args.existing, args.output, args.human_weight)
    elif args.command == 'analyze':
        analyze_disagreements(args.labels)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
