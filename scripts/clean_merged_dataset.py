"""
Clean Merged Dataset Script

This script demonstrates how to clean a merged emotion dataset using the
DataCleaner module. It performs quality-based filtering, duplicate removal,
and class balancing.

Usage:
    python scripts/clean_merged_dataset.py --input data/processed/merged_dataset.csv --output data/cleaned/cleaned_dataset.csv
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.data_cleaner import clean_dataset_pipeline
import json


def main():
    parser = argparse.ArgumentParser(
        description='Clean emotion recognition dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic cleaning (remove poor quality and duplicates)
  python scripts/clean_merged_dataset.py \\
      --input data/processed/merged_dataset.csv \\
      --output data/cleaned/cleaned_dataset.csv \\
      --remove-poor \\
      --remove-duplicates

  # Full cleaning with balancing
  python scripts/clean_merged_dataset.py \\
      --input data/processed/merged_dataset.csv \\
      --output data/cleaned/cleaned_balanced.csv \\
      --remove-poor \\
      --remove-duplicates \\
      --balance \\
      --strategy undersample

  # Clean and enhance
  python scripts/clean_merged_dataset.py \\
      --input data/processed/merged_dataset.csv \\
      --output data/cleaned/cleaned_enhanced.csv \\
      --remove-poor \\
      --enhance \\
      --enhanced-dir data/enhanced
        """
    )
    
    # Input/Output
    parser.add_argument(
        '--input',
        required=True,
        help='Input CSV file path (e.g., data/processed/merged_dataset.csv)'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output CSV file path (e.g., data/cleaned/cleaned_dataset.csv)'
    )
    
    # Cleaning options
    parser.add_argument(
        '--remove-poor',
        action='store_true',
        help='Remove images with poor quality rating'
    )
    parser.add_argument(
        '--remove-duplicates',
        action='store_true',
        help='Remove duplicate images using perceptual hashing'
    )
    parser.add_argument(
        '--enhance',
        action='store_true',
        help='Enhance acceptable quality images (requires --enhanced-dir)'
    )
    parser.add_argument(
        '--enhanced-dir',
        default=None,
        help='Directory to save enhanced images (required if --enhance is used)'
    )
    
    # Balancing options
    parser.add_argument(
        '--balance',
        action='store_true',
        help='Balance emotion class distributions'
    )
    parser.add_argument(
        '--strategy',
        default='undersample',
        choices=['undersample', 'oversample', 'smote'],
        help='Balancing strategy (default: undersample)'
    )
    
    # Column names
    parser.add_argument(
        '--image-column',
        default='image_path',
        help='Name of column containing image paths (default: image_path)'
    )
    parser.add_argument(
        '--label-column',
        default='emotion',
        help='Name of column containing emotion labels (default: emotion)'
    )
    
    # Report options
    parser.add_argument(
        '--report',
        default=None,
        help='Path to save cleaning report JSON (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.enhance and args.enhanced_dir is None:
        parser.error("--enhanced-dir is required when --enhance is used")
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Print configuration
    print("\n" + "=" * 60)
    print("DATASET CLEANING CONFIGURATION")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"\nCleaning options:")
    print(f"  Remove poor quality: {args.remove_poor}")
    print(f"  Remove duplicates: {args.remove_duplicates}")
    print(f"  Enhance images: {args.enhance}")
    if args.enhance:
        print(f"  Enhanced output dir: {args.enhanced_dir}")
    print(f"  Balance classes: {args.balance}")
    if args.balance:
        print(f"  Balance strategy: {args.strategy}")
    print("=" * 60 + "\n")
    
    # Run cleaning pipeline
    try:
        cleaned_df, report = clean_dataset_pipeline(
            input_csv=args.input,
            output_csv=args.output,
            remove_poor=args.remove_poor,
            remove_duplicates=args.remove_duplicates,
            enhance_acceptable=args.enhance,
            balance_classes=args.balance,
            balance_strategy=args.strategy,
            enhanced_output_dir=args.enhanced_dir,
            image_path_column=args.image_column,
            label_column=args.label_column
        )
        
        # Save report if requested
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"\nCleaning report saved to: {args.report}")
        
        print("\n" + "=" * 60)
        print("✓ CLEANING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nCleaned dataset saved to: {args.output}")
        print(f"Final dataset size: {len(cleaned_df)} images")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during cleaning: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
