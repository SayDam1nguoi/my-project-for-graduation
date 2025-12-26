#!/usr/bin/env python3
"""
Launcher - Mở demo_gui.py

File nhỏ để khởi chạy giao diện chính.
Chỉ cần chạy: python launcher.py
"""

if __name__ == '__main__':
    import os
    import sys
    import warnings
    from pathlib import Path
    
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=info, 2=warning, 3=error
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom ops
    
    # Suppress Python warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    # Suppress TensorFlow deprecation warnings
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    # Add to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    # demo_gui.py ở thư mục apps/
    try:
        sys.path.insert(0, str(Path(__file__).parent / "apps"))
        import demo_gui 
        demo_gui.main()
    except ImportError as e:
        print(f"Lỗi: Không tìm thấy demo_gui.py trong apps/")
        print(f"Chi tiết: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Lỗi: {e}")
        import traceback
        traceback.print_exc()
        traceback.print_exc()
        sys.exit(1)