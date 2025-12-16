"""
Script kiểm tra cấu trúc AffectNet dataset

Chạy: python scripts/check_affectnet_structure.py
"""

import os
from pathlib import Path
import pandas as pd


def check_affectnet_structure(affectnet_path: str = r"C:\Users\admin\Downloads\affectnet"):
    """Kiểm tra cấu trúc AffectNet dataset"""
    
    print("="*70)
    print("AffectNet Structure Check")
    print("="*70)
    
    affectnet_path = Path(affectnet_path)
    
    # Check main directory
    print(f"\n1. Checking main directory: {affectnet_path}")
    if not affectnet_path.exists():
        print(f"   ✗ Directory not found!")
        print(f"\n   Please ensure AffectNet is at: {affectnet_path}")
        return False
    print(f"   ✓ Directory exists")
    
    # Check splits
    splits = ['train_set', 'val_set']
    all_ok = True
    
    for split in splits:
        print(f"\n2. Checking {split}:")
        split_dir = affectnet_path / split
        
        if not split_dir.exists():
            print(f"   ✗ {split} directory not found!")
            all_ok = False
            continue
        print(f"   ✓ {split} directory exists")
        
        # Check images directory
        images_dir = split_dir / "images"
        if not images_dir.exists():
            print(f"   ✗ images/ directory not found in {split}!")
            all_ok = False
        else:
            # Count images
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            print(f"   ✓ images/ directory exists ({len(image_files)} images found)")
        
        # Check annotations
        annotation_file = split_dir / "annotations.csv"
        if not annotation_file.exists():
            print(f"   ✗ annotations.csv not found in {split}!")
            all_ok = False
        else:
            print(f"   ✓ annotations.csv exists")
            
            # Try to read annotations
            try:
                df = pd.read_csv(annotation_file)
                print(f"   ✓ annotations.csv readable ({len(df)} rows)")
                print(f"   Columns: {df.columns.tolist()}")
                
                # Check for required columns
                has_image_col = any('image' in col.lower() or 'file' in col.lower() 
                                   for col in df.columns)
                has_emotion_col = any('emotion' in col.lower() or 'expression' in col.lower() 
                                     for col in df.columns)
                
                if has_image_col and has_emotion_col:
                    print(f"   ✓ Required columns found")
                else:
                    print(f"   ⚠ Warning: Could not identify image/emotion columns")
                    print(f"   Available columns: {df.columns.tolist()}")
                
                # Show sample
                print(f"\n   Sample data:")
                print(df.head(3).to_string(index=False))
                
            except Exception as e:
                print(f"   ✗ Error reading annotations.csv: {e}")
                all_ok = False
    
    # Summary
    print("\n" + "="*70)
    if all_ok:
        print("✓ AffectNet structure is CORRECT!")
        print("\nYou can now run:")
        print("  python scripts/quick_add_affectnet.py")
    else:
        print("✗ AffectNet structure has ISSUES!")
        print("\nExpected structure:")
        print(f"{affectnet_path}/")
        print("├── train_set/")
        print("│   ├── images/")
        print("│   │   ├── 0.jpg")
        print("│   │   └── ...")
        print("│   └── annotations.csv")
        print("└── val_set/")
        print("    ├── images/")
        print("    │   ├── 0.jpg")
        print("    │   └── ...")
        print("    └── annotations.csv")
    print("="*70)
    
    return all_ok


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Check AffectNet structure")
    parser.add_argument(
        '--path',
        type=str,
        default=r'C:\Users\admin\Downloads\affectnet',
        help='Path to AffectNet directory'
    )
    
    args = parser.parse_args()
    
    check_affectnet_structure(args.path)


if __name__ == "__main__":
    main()
