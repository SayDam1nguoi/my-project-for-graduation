"""
Script để visualize merged datasets với AffectNet

Chạy: python scripts/visualize_merged_datasets.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_datasets():
    """Visualize emotion distribution của merged datasets"""
    
    processed_dir = Path("data/processed")
    
    # Tìm merged datasets
    merged_files = list(processed_dir.glob("*_with_affectnet.csv"))
    
    if not merged_files:
        print("No merged datasets found!")
        return
    
    print(f"Found {len(merged_files)} merged datasets")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Merged Datasets with AffectNet - Analysis', fontsize=16, fontweight='bold')
    
    # Load all datasets
    datasets = {}
    for f in merged_files:
        name = f.stem.replace('_with_affectnet', '')
        df = pd.read_csv(f)
        datasets[name] = df
        print(f"\nLoaded {name}: {len(df):,} samples")
    
    # Plot 1: Dataset sizes comparison
    ax1 = axes[0, 0]
    names = list(datasets.keys())
    sizes = [len(df) for df in datasets.values()]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax1.bar(names, sizes, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Dataset Sizes Comparison', fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    ax1.set_xlabel('Dataset')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')
    
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')
    
    # Plot 2: Source distribution
    ax2 = axes[0, 1]
    
    # Use largest dataset for source distribution
    largest_name = max(datasets.keys(), key=lambda k: len(datasets[k]))
    largest_df = datasets[largest_name]
    
    source_counts = largest_df['source_dataset'].value_counts()
    colors_pie = ['#FF6B6B', '#4ECDC4']
    
    wedges, texts, autotexts = ax2.pie(
        source_counts.values,
        labels=source_counts.index,
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90,
        explode=[0.05] * len(source_counts)
    )
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax2.set_title(f'Source Distribution\n({largest_name})', fontweight='bold')
    
    # Plot 3: Emotion distribution comparison
    ax3 = axes[1, 0]
    
    emotions = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust', 'Neutral']
    x = range(len(emotions))
    width = 0.25
    
    for i, (name, df) in enumerate(datasets.items()):
        emotion_counts = df['emotion'].value_counts()
        counts = [emotion_counts.get(e, 0) for e in emotions]
        offset = (i - 1) * width
        ax3.bar([xi + offset for xi in x], counts, width, 
               label=name, alpha=0.7, edgecolor='black')
    
    ax3.set_title('Emotion Distribution Comparison', fontweight='bold')
    ax3.set_ylabel('Number of Samples')
    ax3.set_xlabel('Emotion')
    ax3.set_xticks(x)
    ax3.set_xticklabels(emotions, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Emotion percentages (largest dataset)
    ax4 = axes[1, 1]
    
    emotion_counts = largest_df['emotion'].value_counts()
    emotion_pcts = (emotion_counts / len(largest_df) * 100).sort_values(ascending=True)
    
    colors_bar = plt.cm.Set3(range(len(emotion_pcts)))
    bars = ax4.barh(emotion_pcts.index, emotion_pcts.values, color=colors_bar, 
                    alpha=0.7, edgecolor='black')
    
    ax4.set_title(f'Emotion Distribution %\n({largest_name})', fontweight='bold')
    ax4.set_xlabel('Percentage (%)')
    ax4.set_ylabel('Emotion')
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, emotion_pcts.values)):
        ax4.text(pct + 0.5, i, f'{pct:.1f}%', 
                va='center', fontweight='bold')
    
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("data/reports/merged_datasets_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization to {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for name, df in datasets.items():
        print(f"\n{name}:")
        print(f"  Total samples: {len(df):,}")
        
        if 'source_dataset' in df.columns:
            print(f"  Sources:")
            for source, count in df['source_dataset'].value_counts().items():
                pct = count / len(df) * 100
                print(f"    - {source}: {count:,} ({pct:.1f}%)")
        
        print(f"  Emotions:")
        for emotion, count in df['emotion'].value_counts().items():
            pct = count / len(df) * 100
            print(f"    - {emotion}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    visualize_datasets()
