#!/usr/bin/env python3
"""
Monitor Smart Label progress in real-time
Shows frame counts, distribution, and v3.0 agreement
"""

import sqlite3
import sys
from pathlib import Path
from datetime import datetime

def get_stats():
    db_path = Path("smart_label/queue.db")
    if not db_path.exists():
        print("❌ Queue database not found!")
        return None
    
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Get total and labeled counts
        c.execute("SELECT COUNT(*) as total FROM queue")
        total = c.fetchone()['total']
        
        c.execute("SELECT COUNT(*) as labeled FROM queue WHERE human_label IS NOT NULL")
        labeled = c.fetchone()['labeled']
        
        remaining = total - labeled
        percent = (labeled / total * 100) if total > 0 else 0
        
        # Get distribution
        c.execute("""
            SELECT human_label, COUNT(*) as count 
            FROM queue 
            WHERE human_label IS NOT NULL 
            GROUP BY human_label 
            ORDER BY count DESC
        """)
        distribution = {row['human_label']: row['count'] for row in c.fetchall()}
        
        # Get v3.0 agreement
        c.execute("""
            SELECT COUNT(*) as agree
            FROM queue
            WHERE human_label IS NOT NULL
            AND human_label = predicted_style
        """)
        agree = c.fetchone()['agree']
        
        c.execute("""
            SELECT COUNT(*) as disagree
            FROM queue
            WHERE human_label IS NOT NULL
            AND human_label != predicted_style
        """)
        disagree = c.fetchone()['disagree']
        
        conn.close()
        
        return {
            'total': total,
            'labeled': labeled,
            'remaining': remaining,
            'percent': percent,
            'distribution': distribution,
            'agree': agree,
            'disagree': disagree
        }
    except Exception as e:
        print(f"❌ Error reading database: {e}")
        return None

def print_stats(stats):
    if not stats:
        return
    
    print("\n" + "="*50)
    print("  Smart Label Progress Monitor")
    print("="*50)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Progress bar
    bar_length = 30
    filled = int(bar_length * stats['percent'] / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"Progress: [{bar}] {stats['percent']:.1f}%")
    print(f"  {stats['labeled']} / {stats['total']} frames labeled")
    print(f"  {stats['remaining']} frames remaining")
    
    # Distribution
    if stats['distribution']:
        print("\nLabel Distribution:")
        total_labeled = sum(stats['distribution'].values())
        for style in ['flat', 'grim', 'modern', 'moe', 'painterly', 'retro']:
            count = stats['distribution'].get(style, 0)
            pct = (count / total_labeled * 100) if total_labeled > 0 else 0
            print(f"  {style:12s}: {count:3d} ({pct:5.1f}%)")
    
    # Agreement
    total_labeled = stats['labeled']
    if total_labeled > 0:
        print(f"\nAgreement with v3.0:")
        agree_pct = (stats['agree'] / total_labeled * 100)
        print(f"  Agree:   {stats['agree']:3d} ({agree_pct:5.1f}%)")
        print(f"  Disagree: {stats['disagree']:3d} ({100-agree_pct:5.1f}%)")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    stats = get_stats()
    print_stats(stats)
