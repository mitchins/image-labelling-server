#!/usr/bin/env python3
"""
CLI entry point for smart_label package.

Usage:
    python -m smart_label prepare --source ... --output ...
    python -m smart_label serve --db ...
    python -m smart_label export --db ... --output ...
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Smart Label - Intelligent Human Labeling System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  prepare   Generate diverse sample queue using embedding clustering
  serve     Start the labeling web server
  export    Export labeled data to JSON
  stats     Show labeling statistics

Examples:
  # Prepare queue with 80 clusters, 1024 samples
  python -m smart_label prepare \\
      --source ensemble_dataset_gold.json \\
      --classifier efficientnetv2_s_ultra_gold.pth \\
      --backbone efficientnetv2_s \\
      --output smart_label/queue.db

  # Start server
  python -m smart_label serve --db smart_label/queue.db

  # Export labels
  python -m smart_label export --db smart_label/queue.db --output labels.json
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Prepare command
    prep = subparsers.add_parser('prepare', help='Prepare labeling queue')
    prep.add_argument('--source', required=True, help='Source dataset JSON')
    prep.add_argument('--output', default='smart_label/queue.db', help='Output database')
    prep.add_argument('--clusters', type=int, default=80, help='Number of clusters')
    prep.add_argument('--samples-per-cluster', type=int, default=13, help='Samples per cluster')
    prep.add_argument('--max-samples', type=int, default=1024, help='Max total samples')
    prep.add_argument('--classifier', help='Path to classifier model')
    prep.add_argument('--backbone', default='efficientnet_b0', 
                      choices=['efficientnet_b0', 'efficientnetv2_s'])
    prep.add_argument('--embeddings-cache', help='Path to cache embeddings')
    prep.add_argument('--batch-size', type=int, default=32, help='Batch size')
    prep.add_argument('--limit-source', type=int, help='Limit source frames (testing)')
    
    # Serve command
    serve = subparsers.add_parser('serve', help='Start labeling server')
    serve.add_argument('--db', required=True, help='Queue database path')
    serve.add_argument('--port', type=int, default=8765, help='Server port')
    serve.add_argument('--host', default='0.0.0.0', help='Server host')
    
    # Export command
    export = subparsers.add_parser('export', help='Export labeled data')
    export.add_argument('--db', required=True, help='Queue database path')
    export.add_argument('--output', default='labels.json', help='Output JSON file')
    export.add_argument('--exclude-refuse', action='store_true', help='Exclude REFUSE labels')
    
    # Stats command
    stats = subparsers.add_parser('stats', help='Show labeling statistics')
    stats.add_argument('--db', required=True, help='Queue database path')
    
    args = parser.parse_args()
    
    if args.command == 'prepare':
        from smart_label.prepare import main as prepare_main
        sys.argv = ['prepare'] + [f'--{k.replace("_", "-")}={v}' for k, v in vars(args).items() 
                                   if k != 'command' and v is not None]
        prepare_main()
        
    elif args.command == 'serve':
        from smart_label.server import main as serve_main
        sys.argv = ['serve', '--db', args.db, '--port', str(args.port), '--host', args.host]
        serve_main()
        
    elif args.command == 'export':
        export_labels(args.db, args.output, args.exclude_refuse)
        
    elif args.command == 'stats':
        show_stats(args.db)
        
    else:
        parser.print_help()


def export_labels(db_path: str, output_path: str, exclude_refuse: bool = False):
    """Export labeled data to JSON file."""
    import json
    import sqlite3
    from pathlib import Path
    
    if not Path(db_path).exists():
        print(f"Error: Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    query = """
        SELECT path, human_label, cluster_id, predicted_style, labeled_at
        FROM queue
        WHERE human_label IS NOT NULL
    """
    if exclude_refuse:
        query += " AND human_label != 'REFUSE'"
    
    cur.execute(query)
    
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
    
    conn.close()
    
    with open(output_path, 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"Exported {len(labels)} labels to {output_path}")
    
    # Show distribution
    from collections import Counter
    dist = Counter(l['label'] for l in labels)
    print("\nLabel distribution:")
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count}")


def show_stats(db_path: str):
    """Show labeling statistics."""
    import sqlite3
    from pathlib import Path
    
    if not Path(db_path).exists():
        print(f"Error: Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    # Total
    cur.execute("SELECT COUNT(*) FROM queue")
    total = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM queue WHERE human_label IS NOT NULL")
    labeled = cur.fetchone()[0]
    
    print(f"Progress: {labeled} / {total} ({100*labeled/total:.1f}%)")
    print()
    
    # By label
    cur.execute("""
        SELECT human_label, COUNT(*) as count
        FROM queue
        WHERE human_label IS NOT NULL
        GROUP BY human_label
        ORDER BY count DESC
    """)
    
    print("By label:")
    for row in cur.fetchall():
        print(f"  {row['human_label']}: {row['count']}")
    
    # By cluster (sampled)
    cur.execute("""
        SELECT cluster_id, 
               COUNT(*) as total,
               SUM(CASE WHEN human_label IS NOT NULL THEN 1 ELSE 0 END) as labeled
        FROM queue
        GROUP BY cluster_id
        ORDER BY cluster_id
    """)
    
    print()
    print("By cluster (first 10):")
    for i, row in enumerate(cur.fetchall()):
        if i >= 10:
            print("  ...")
            break
        print(f"  Cluster {row['cluster_id']}: {row['labeled']}/{row['total']} labeled")
    
    conn.close()


if __name__ == '__main__':
    main()
