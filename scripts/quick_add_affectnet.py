"""
Script nhanh để thêm AffectNet vào datasets hiện có

Chạy: python scripts/quick_add_affectnet.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_affectnet import AffectNetPreparer


def main():
    print("="*70)
    print("Quick AffectNet Integration")
    print("="*70)
    
    # Đường dẫn AffectNet của bạn
    affectnet_source = r"C:\Users\admin\Downloads\affectnet"
    
    print(f"\nAffectNet source: {affectnet_source}")
    print("\nOptions:")
    print("1. Prepare AffectNet (không copy ảnh, chỉ tạo CSV với đường dẫn gốc)")
    print("2. Prepare AffectNet (copy ảnh vào project)")
    print("3. Merge AffectNet với datasets hiện có")
    print("4. Làm tất cả (prepare + merge, không copy ảnh)")
    print("5. Test với 1000 samples mỗi split")
    
    choice = input("\nChọn option (1-5): ").strip()
    
    try:
        preparer = AffectNetPreparer(
            source_path=affectnet_source,
            target_path="data/raw/affectnet"
        )
        
        if choice == "1":
            print("\n→ Preparing AffectNet (keeping original paths)...")
            preparer.prepare_dataset(copy_images=False)
            
        elif choice == "2":
            print("\n→ Preparing AffectNet (copying images)...")
            print("⚠ Warning: This will take time and disk space!")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm == 'y':
                preparer.prepare_dataset(copy_images=True)
            else:
                print("Cancelled.")
                
        elif choice == "3":
            print("\n→ Merging with existing datasets...")
            preparer.merge_with_existing_datasets()
            
        elif choice == "4":
            print("\n→ Prepare + Merge (keeping original paths)...")
            preparer.prepare_dataset(copy_images=False)
            preparer.merge_with_existing_datasets()
            
        elif choice == "5":
            print("\n→ Test mode: 1000 samples per split...")
            preparer.prepare_dataset(copy_images=False, max_samples_per_split=1000)
            preparer.merge_with_existing_datasets()
            
        else:
            print("Invalid choice!")
            return
        
        print("\n" + "="*70)
        print("✓ DONE!")
        print("="*70)
        print("\nCheck results in:")
        print("- data/raw/affectnet/affectnet_prepared.csv")
        print("- data/processed/*_with_affectnet.csv")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nKiểm tra lại:")
        print(f"1. Đường dẫn: {affectnet_source}")
        print("2. Cấu trúc thư mục:")
        print("   affectnet/")
        print("   ├── train_set/")
        print("   │   ├── images/")
        print("   │   └── annotations.csv")
        print("   └── val_set/")
        print("       ├── images/")
        print("       └── annotations.csv")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
