#!/usr/bin/env python3
"""
Force Clear All Cache

Xóa hoàn toàn cache dịch để dùng model mới.
"""

import sys
from pathlib import Path
import shutil

def force_clear_cache():
    """Xóa hoàn toàn cache."""
    print("=" * 80)
    print("FORCE CLEAR ALL CACHE")
    print("=" * 80)
    print()
    
    try:
        # 1. Clear diskcache
        try:
            import diskcache
            cache_dir = Path(".cache/transcriptions")
            if cache_dir.exists():
                cache = diskcache.Cache(str(cache_dir))
                entries = len(cache)
                print(f"📦 Found {entries} cache entries")
                cache.clear()
                cache.close()
                print("✅ Cleared diskcache")
        except Exception as e:
            print(f"⚠️  Could not clear diskcache: {e}")
        
        # 2. Delete cache directory
        cache_dir = Path(".cache/transcriptions")
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                print("✅ Deleted cache directory")
            except Exception as e:
                print(f"⚠️  Could not delete directory: {e}")
        
        # 3. Recreate empty directory
        cache_dir.mkdir(parents=True, exist_ok=True)
        print("✅ Recreated empty cache directory")
        
        # 4. Clear transcripts
        transcripts_dir = Path("transcripts")
        if transcripts_dir.exists():
            count = 0
            for file in transcripts_dir.glob("*.txt"):
                if file.name != ".gitkeep":
                    file.unlink()
                    count += 1
            if count > 0:
                print(f"✅ Deleted {count} old transcripts")
        
        print()
        print("=" * 80)
        print("✅ CACHE CLEARED SUCCESSFULLY!")
        print("=" * 80)
        print()
        print("Bây giờ bạn có thể:")
        print("  1. Mở GUI")
        print("  2. Chọn video")
        print("  3. Dịch lại")
        print("  4. ✅ Không còn hallucination!")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ ERROR")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(force_clear_cache())
