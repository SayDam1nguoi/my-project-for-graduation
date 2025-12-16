"""Script kiểm tra datasets đã merge"""
import pandas as pd
import os
from pathlib import Path

files = [
    'dataset_high_quality_balanced.csv',
    'dataset_large_diverse.csv', 
    'dataset_small_highest_quality.csv'
]

print('='*70)
print('DATASET SUMMARY')
print('='*70)

for f in files:
    path = Path(f'data/processed/{f}')
    
    if not path.exists():
        print(f'\n{f}: NOT FOUND')
        continue
    
    df = pd.read_csv(path)
    
    print(f'\n📊 {f}')
    print(f'   Total: {len(df):,} samples')
    print(f'   Size: {path.stat().st_size/1024/1024:.1f} MB')
    
    print(f'\n   Datasets:')
    for ds, count in df['source_dataset'].value_counts().items():
        pct = count/len(df)*100
        print(f'     {ds:12} {count:6,} ({pct:5.1f}%)')
    
    print(f'\n   Emotions:')
    for em, count in df['emotion'].value_counts().items():
        pct = count/len(df)*100
        print(f'     {em:12} {count:6,} ({pct:5.1f}%)')

print('\n' + '='*70)
