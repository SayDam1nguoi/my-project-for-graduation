"""
Script để chuẩn bị AffectNet dataset đã được organize theo thư mục emotions

Cấu trúc:
affectnet/
├── Train/
│   ├── anger/
│   ├── contempt/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── Test/
    ├── Anger/
    ├── Contempt/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── neutral/
    ├── sad/
    └── surprise/
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class AffectNetOrganizedPreparer:
    """Chuẩn bị AffectNet dataset đã organize theo emotions"""
    
    # Emotion mapping to standard 7 emotions
    EMOTION_MAPPING = {
        'anger': 'Angry',
        'Anger': 'Angry',
        'contempt': 'Neutral',  # Map contempt to Neutral
        'Contempt': 'Neutral',
        'disgust': 'Disgust',
        'fear': 'Fear',
        'happy': 'Happy',
        'neutral': 'Neutral',
        'sad': 'Sad',
        'surprise': 'Surprise'
    }
    
    def __init__(self, source_path: str = r"C:\Users\admin\Downloads\affectnet"):
        """
        Initialize preparer
        
        Args:
            source_path: Đường dẫn đến thư mục AffectNet
        """
        self.source_path = Path(source_path)
        
        if not self.source_path.exists():
            raise FileNotFoundError(f"AffectNet not found: {self.source_path}")
        
        print(f"Source: {self.source_path}")
    
    def prepare_dataset(self, max_samples_per_emotion: int = None):
        """
        Chuẩn bị AffectNet dataset
        
        Args:
            max_samples_per_emotion: Giới hạn số samples mỗi emotion (None = không giới hạn)
        """
        print("\n" + "="*70)
        print("AffectNet Dataset Preparation (Organized Structure)")
        print("="*70)
        
        all_data = []
        
        # Process Train và Test
        for split_name in ['Train', 'Test']:
            print(f"\n{'='*70}")
            print(f"Processing {split_name}")
            print(f"{'='*70}")
            
            split_dir = self.source_path / split_name
            
            if not split_dir.exists():
                print(f"⚠ {split_name} directory not found!")
                continue
            
            # Lấy tất cả emotion folders
            emotion_folders = [d for d in split_dir.iterdir() if d.is_dir()]
            print(f"Found {len(emotion_folders)} emotion folders")
            
            for emotion_folder in emotion_folders:
                emotion_name = emotion_folder.name
                
                # Map emotion name
                if emotion_name not in self.EMOTION_MAPPING:
                    print(f"⚠ Unknown emotion folder: {emotion_name}, skipping...")
                    continue
                
                standard_emotion = self.EMOTION_MAPPING[emotion_name]
                
                # Lấy tất cả ảnh
                image_files = list(emotion_folder.glob("*.jpg")) + \
                             list(emotion_folder.glob("*.png")) + \
                             list(emotion_folder.glob("*.jpeg"))
                
                print(f"\n{emotion_name} → {standard_emotion}: {len(image_files)} images")
                
                # Giới hạn nếu cần
                if max_samples_per_emotion and len(image_files) > max_samples_per_emotion:
                    image_files = image_files[:max_samples_per_emotion]
                    print(f"  Limited to {len(image_files)} samples")
                
                # Process từng ảnh
                processed = 0
                skipped = 0
                
                for img_file in tqdm(image_files, desc=f"  Processing {emotion_name}"):
                    try:
                        # Lấy thông tin ảnh
                        with Image.open(img_file) as img:
                            width, height = img.size
                        
                        file_size = img_file.stat().st_size
                        
                        # Thêm vào data
                        all_data.append({
                            'image': img_file.name,
                            'expression': emotion_name,
                            'emotion_name': standard_emotion,
                            'image_path': str(img_file),
                            'source_dataset': 'affectnet',
                            'split': split_name.lower(),
                            'original_label': emotion_name,
                            'emotion': standard_emotion,
                            'width': width,
                            'height': height,
                            'file_size': file_size
                        })
                        
                        processed += 1
                        
                    except Exception as e:
                        skipped += 1
                        continue
                
                print(f"  ✓ Processed: {processed}, ✗ Skipped: {skipped}")
        
        # Tạo DataFrame
        if not all_data:
            print("\n✗ No data processed!")
            return None
        
        df = pd.DataFrame(all_data)
        
        # Lưu CSV
        output_dir = Path("data/raw/affectnet")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv = output_dir / "affectnet_prepared.csv"
        df.to_csv(output_csv, index=False)
        
        print("\n" + "="*70)
        print("AFFECTNET PREPARATION COMPLETE!")
        print("="*70)
        print(f"\nTotal samples: {len(df)}")
        print(f"Saved to: {output_csv}")
        
        print("\nOverall emotion distribution:")
        emotion_counts = df['emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            pct = count / len(df) * 100
            print(f"  {emotion}: {count} ({pct:.1f}%)")
        
        print("\nSplit distribution:")
        split_counts = df['split'].value_counts()
        for split, count in split_counts.items():
            pct = count / len(df) * 100
            print(f"  {split}: {count} ({pct:.1f}%)")
        
        return df
    
    def merge_with_existing_datasets(self):
        """Merge AffectNet với các datasets hiện có"""
        print("\n" + "="*70)
        print("Merging AffectNet with Existing Datasets")
        print("="*70)
        
        # Load AffectNet
        affectnet_csv = Path("data/raw/affectnet/affectnet_prepared.csv")
        
        if not affectnet_csv.exists():
            print(f"✗ AffectNet CSV not found: {affectnet_csv}")
            print("Please run prepare_dataset() first!")
            return None
        
        print(f"\nLoading AffectNet from {affectnet_csv}...")
        affectnet_df = pd.read_csv(affectnet_csv)
        print(f"✓ Loaded {len(affectnet_df)} samples from AffectNet")
        
        # Load existing datasets
        processed_dir = Path("data/processed")
        existing_datasets = []
        
        for csv_file in processed_dir.glob("dataset_*.csv"):
            # Skip already merged files
            if 'with_affectnet' in csv_file.name:
                continue
            
            print(f"\nLoading {csv_file.name}...")
            df = pd.read_csv(csv_file)
            print(f"✓ Loaded {len(df)} samples")
            existing_datasets.append((csv_file.stem, df))
        
        if not existing_datasets:
            print("\n⚠ No existing datasets found in data/processed/")
            print("Creating new dataset with AffectNet only...")
            
            output_path = processed_dir / "dataset_with_affectnet.csv"
            affectnet_df.to_csv(output_path, index=False)
            print(f"✓ Saved to {output_path}")
            return affectnet_df
        
        # Merge với từng dataset
        merged_files = []
        
        for dataset_name, existing_df in existing_datasets:
            print(f"\n{'='*70}")
            print(f"Merging with {dataset_name}")
            print(f"{'='*70}")
            
            # Đảm bảo có cùng columns
            common_cols = list(set(affectnet_df.columns) & set(existing_df.columns))
            print(f"Common columns: {len(common_cols)}")
            
            # Nếu thiếu columns, thêm vào
            for col in affectnet_df.columns:
                if col not in existing_df.columns:
                    existing_df[col] = None
            
            for col in existing_df.columns:
                if col not in affectnet_df.columns:
                    affectnet_df[col] = None
            
            # Merge
            merged_df = pd.concat([existing_df, affectnet_df], ignore_index=True)
            print(f"✓ Merged dataset: {len(merged_df)} samples")
            
            # Thống kê
            print("\nDataset sources:")
            source_counts = merged_df['source_dataset'].value_counts()
            for source, count in source_counts.items():
                pct = count / len(merged_df) * 100
                print(f"  {source}: {count} ({pct:.1f}%)")
            
            print("\nEmotion distribution:")
            emotion_counts = merged_df['emotion'].value_counts()
            for emotion, count in emotion_counts.items():
                pct = count / len(merged_df) * 100
                print(f"  {emotion}: {count} ({pct:.1f}%)")
            
            # Lưu merged dataset
            output_name = f"{dataset_name}_with_affectnet.csv"
            output_path = processed_dir / output_name
            merged_df.to_csv(output_path, index=False)
            print(f"\n✓ Saved merged dataset to {output_path}")
            merged_files.append(output_path)
        
        print("\n" + "="*70)
        print("MERGE COMPLETE!")
        print("="*70)
        print(f"\nCreated {len(merged_files)} merged datasets:")
        for f in merged_files:
            print(f"  - {f.name}")
        
        return merged_files


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare AffectNet (organized structure) and merge with existing datasets"
    )
    parser.add_argument(
        '--source',
        type=str,
        default=r'C:\Users\admin\Downloads\affectnet',
        help='Path to AffectNet directory'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per emotion (for testing)'
    )
    parser.add_argument(
        '--merge-only',
        action='store_true',
        help='Only merge existing affectnet_prepared.csv'
    )
    
    args = parser.parse_args()
    
    try:
        preparer = AffectNetOrganizedPreparer(args.source)
        
        if not args.merge_only:
            # Prepare AffectNet
            print("\n" + "="*70)
            print("STEP 1: Prepare AffectNet Dataset")
            print("="*70)
            
            affectnet_df = preparer.prepare_dataset(
                max_samples_per_emotion=args.max_samples
            )
            
            if affectnet_df is None:
                print("\n✗ Failed to prepare AffectNet")
                return
        
        # Merge with existing datasets
        print("\n" + "="*70)
        print("STEP 2: Merge with Existing Datasets")
        print("="*70)
        
        merged_files = preparer.merge_with_existing_datasets()
        
        print("\n" + "="*70)
        print("ALL DONE!")
        print("="*70)
        print("\nNext steps:")
        print("1. Review the merged datasets in data/processed/")
        print("2. Check emotion distributions")
        print("3. Proceed with model training")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
