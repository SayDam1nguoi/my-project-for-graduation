"""
Script Tự Động Migration Sang Whisper

Sử dụng:
    python scripts/migrate_to_whisper.py
    python scripts/migrate_to_whisper.py --model small --backup
"""

import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def backup_config(config_path: Path) -> Path:
    """
    Backup config file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Path to backup file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.parent / f"{config_path.stem}_backup_{timestamp}{config_path.suffix}"
    
    shutil.copy2(config_path, backup_path)
    print(f"✅ Đã backup config: {backup_path}")
    
    return backup_path


def update_config_for_whisper(
    config_path: Path,
    model_size: str = "small",
    beam_size: int = 8,
    enable_enhancements: bool = True,
    create_backup: bool = True
) -> bool:
    """
    Update config file to use Whisper.
    
    Args:
        config_path: Path to config file
        model_size: Whisper model size
        beam_size: Beam size for decoding
        enable_enhancements: Enable audio enhancements
        create_backup: Create backup before updating
        
    Returns:
        True if successful
    """
    try:
        # Backup if requested
        if create_backup:
            backup_config(config_path)
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            config = {}
        
        # Ensure speech_to_text section exists
        if 'speech_to_text' not in config:
            config['speech_to_text'] = {}
        
        stt_config = config['speech_to_text']
        
        # Add Whisper settings
        stt_config['model_type'] = 'whisper'
        stt_config['model_size'] = model_size
        stt_config['compute_type'] = 'int8'
        stt_config['device'] = 'cpu'
        stt_config['num_threads'] = 4
        stt_config['beam_size'] = beam_size
        stt_config['best_of'] = beam_size
        stt_config['temperature'] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        stt_config['word_timestamps'] = True
        stt_config['condition_on_previous_text'] = True
        
        # Enable enhancements
        if enable_enhancements:
            stt_config['enable_audio_cleaning'] = True
            stt_config['enable_vad'] = True
            stt_config['vad_method'] = 'silero'
            stt_config['overlap_duration'] = 0.8
            stt_config['max_buffer_size'] = 15
        
        # Enable custom vocabulary if file exists
        vocab_file = Path("config/vietnamese_custom_vocabulary.json")
        if vocab_file.exists():
            stt_config['enable_vocabulary'] = True
            stt_config['vocabulary_file'] = str(vocab_file)
        
        # Performance settings
        stt_config['max_memory_mb'] = 800
        stt_config['cpu_limit_percent'] = 70.0
        stt_config['min_real_time_factor'] = 0.8
        
        # Fallback settings (keep VOSK as fallback)
        stt_config['fallback_to_vosk'] = True
        if 'vosk_model_path' not in stt_config:
            stt_config['vosk_model_path'] = 'models/vosk-model-vn-0.4'
        
        # Keep existing settings
        if 'language' not in stt_config:
            stt_config['language'] = 'vi'
        if 'sample_rate' not in stt_config:
            stt_config['sample_rate'] = 16000
        if 'chunk_duration' not in stt_config:
            stt_config['chunk_duration'] = 5.0
        if 'max_latency' not in stt_config:
            stt_config['max_latency'] = 8.0
        
        # Save updated config
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"✅ Đã cập nhật config: {config_path}")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi khi cập nhật config: {e}")
        return False


def check_dependencies() -> dict:
    """
    Check if required dependencies are installed.
    
    Returns:
        Dictionary with dependency status
    """
    status = {}
    
    # Check faster-whisper
    try:
        import faster_whisper
        status['faster_whisper'] = True
        print("✅ faster-whisper đã cài đặt")
    except ImportError:
        status['faster_whisper'] = False
        print("❌ faster-whisper chưa cài đặt")
    
    # Check torch
    try:
        import torch
        status['torch'] = True
        print("✅ torch đã cài đặt")
    except ImportError:
        status['torch'] = False
        print("⚠️  torch chưa cài đặt (khuyến nghị)")
    
    # Check noisereduce
    try:
        import noisereduce
        status['noisereduce'] = True
        print("✅ noisereduce đã cài đặt")
    except ImportError:
        status['noisereduce'] = False
        print("⚠️  noisereduce chưa cài đặt (khuyến nghị)")
    
    # Check scipy
    try:
        import scipy
        status['scipy'] = True
        print("✅ scipy đã cài đặt")
    except ImportError:
        status['scipy'] = False
        print("⚠️  scipy chưa cài đặt (khuyến nghị)")
    
    return status


def install_dependencies() -> bool:
    """
    Install required dependencies.
    
    Returns:
        True if successful
    """
    import subprocess
    
    print("\n📦 Đang cài đặt dependencies...")
    
    packages = [
        'faster-whisper',
        'torch',
        'noisereduce',
        'scipy'
    ]
    
    try:
        for package in packages:
            print(f"\n📥 Cài đặt {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ Đã cài đặt {package}")
        
        print("\n✅ Đã cài đặt tất cả dependencies")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Lỗi khi cài đặt dependencies: {e}")
        return False


def print_summary(model_size: str, config_path: Path):
    """Print migration summary."""
    print("\n" + "=" * 80)
    print("MIGRATION HOÀN TẤT")
    print("=" * 80)
    print()
    print(f"✅ Config đã được cập nhật: {config_path}")
    print(f"✅ Model size: {model_size}")
    print()
    print("📊 Kết quả kỳ vọng:")
    
    if model_size == "tiny":
        print("   - Độ chính xác: 70-75%")
        print("   - Tốc độ: Rất nhanh (4-5x real-time)")
        print("   - RAM: 1-2 GB")
    elif model_size == "base":
        print("   - Độ chính xác: 85-90%")
        print("   - Tốc độ: Nhanh (2-3x real-time)")
        print("   - RAM: 2-3 GB")
    elif model_size == "small":
        print("   - Độ chính xác: 90-93% ⭐ Khuyến nghị")
        print("   - Tốc độ: Trung bình (1.5-2x real-time)")
        print("   - RAM: 3-4 GB")
    elif model_size == "medium":
        print("   - Độ chính xác: 93-95%")
        print("   - Tốc độ: Chậm (2-3x real-time)")
        print("   - RAM: 5-6 GB")
    elif model_size == "large":
        print("   - Độ chính xác: 95-97%")
        print("   - Tốc độ: Rất chậm (3-4x real-time)")
        print("   - RAM: 9-10 GB")
    
    print()
    print("🚀 Bước tiếp theo:")
    print("   1. Chạy test: python scripts/test_stt_accuracy.py")
    print(f"   2. Chạy app: python launcher.py --config {config_path}")
    print("   3. Kiểm tra log: logs/speech_analysis.log")
    print()
    print("📚 Tài liệu:")
    print("   - docs/MIGRATION_TO_WHISPER.md")
    print("   - docs/VIETNAMESE_STT_ACCURACY_GUIDE.md")
    print()
    print("=" * 80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Migration Script: Chuyển từ VOSK sang Whisper"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/speech_config.yaml',
        help='Path to config file (default: config/speech_config.yaml)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='small',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: small)'
    )
    parser.add_argument(
        '--beam-size',
        type=int,
        default=8,
        help='Beam size for decoding (default: 8)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup of config file'
    )
    parser.add_argument(
        '--no-enhancements',
        action='store_true',
        help='Do not enable audio enhancements'
    )
    parser.add_argument(
        '--install-deps',
        action='store_true',
        help='Install required dependencies'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check dependencies, do not update config'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("MIGRATION SCRIPT: VOSK → WHISPER")
    print("=" * 80)
    print()
    
    # Check dependencies
    print("🔍 Kiểm tra dependencies...")
    print()
    deps_status = check_dependencies()
    print()
    
    # If check-only mode, exit here
    if args.check_only:
        if not deps_status['faster_whisper']:
            print("⚠️  Cần cài đặt faster-whisper:")
            print("   pip install faster-whisper")
        return
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            print("\n❌ Không thể cài đặt dependencies. Vui lòng cài thủ công:")
            print("   pip install faster-whisper torch noisereduce scipy")
            return
        print()
    
    # Check if faster-whisper is installed
    if not deps_status['faster_whisper']:
        print("❌ faster-whisper chưa được cài đặt!")
        print()
        print("Vui lòng cài đặt:")
        print("   pip install faster-whisper")
        print()
        print("Hoặc chạy với --install-deps:")
        print(f"   python {sys.argv[0]} --install-deps")
        return
    
    # Update config
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"❌ Config file không tồn tại: {config_path}")
        return
    
    print(f"📝 Đang cập nhật config: {config_path}")
    print(f"   Model size: {args.model}")
    print(f"   Beam size: {args.beam_size}")
    print(f"   Audio enhancements: {not args.no_enhancements}")
    print(f"   Create backup: {not args.no_backup}")
    print()
    
    success = update_config_for_whisper(
        config_path=config_path,
        model_size=args.model,
        beam_size=args.beam_size,
        enable_enhancements=not args.no_enhancements,
        create_backup=not args.no_backup
    )
    
    if success:
        print_summary(args.model, config_path)
    else:
        print("\n❌ Migration thất bại!")
        print("Vui lòng kiểm tra lỗi và thử lại.")


if __name__ == "__main__":
    main()

