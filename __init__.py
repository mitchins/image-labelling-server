"""
Smart Label - Intelligent Human Labeling System

Collects high-quality human labels through:
1. Embedding-based clustering for statistically diverse sample selection
2. Pre-classification balancing to ensure style coverage  
3. Zero-friction web UI for rapid human labeling
"""

__version__ = "0.1.0"

STYLES = ['flat', 'grim', 'modern', 'moe', 'painterly', 'retro']
STYLE_KEYS = {
    '1': 'flat',
    '2': 'grim', 
    '3': 'modern',
    '4': 'moe',
    '5': 'painterly',
    '6': 'retro',
}
