#!/usr/bin/env python3
"""
Script to download speech recognition models (Whisper and Vosk).
Automatically detects system capabilities and downloads appropriate models.
"""

import os
import sys
import urllib.request
import zipfile
import argparse
from pathlib import Path
from typing import Optional, Tuple
import platform

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class DownloadProgressBar:
    """Display download progress bar."""
    
    def __init__(self, total_size: int, description: str = "Downloading"):
        self.total_size = total_size
        self.description = description
        self.downloaded = 0
        self.bar_length = 50
        
    def update(self, chunk_size: int):
        """Update progress bar."""
        self.downloaded += chunk_size
        if self.total_size > 0:
            progress = self.downloaded / self.total_size
            filled = int(self.bar_length * progress)
            bar = '█' * filled + '░' * (self.bar_length - filled)
            percent = progress * 100
            mb_downloaded = self.downloaded / (1024 * 1024)
            mb_total = self.total_size / (1024 * 1024)
            
            print(f'\r{self.description}: |{bar}| {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', 
                  end='', flush=True)
        else:
            # Unknown size
            mb_downloaded = self.downloaded / (1024 * 1024)
            print(f'\r{self.description}: {mb_downloaded:.1f} MB downloaded', 
                  end='', flush=True)
    
    def finish(self):
        """Finish progress bar."""
        print()  # New line


def check_gpu_available() -> bool:
    """Check if GPU is available for PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_system_info() -> dict:
    """Get system information to recommend appropriate models."""
    info = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'gpu_available': check_gpu_available(),
    }
    
    # Estimate available RAM (rough estimate)
    try:
        if platform.system() == 'Windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulong = ctypes.c_ulong
            class MEMORYSTATUS(ctypes.Structure):
                _fields_ = [
                    ('dwLength', c_ulong),
                    ('dwMemoryLoad', c_ulong),
                    ('dwTotalPhys', c_ulong),
                    ('dwAvailPhys', c_ulong),
                    ('dwTotalPageFile', c_ulong),
                    ('dwAvailPageFile', c_ulong),
                    ('dwTotalVirtual', c_ulong),
                    ('dwAvailVirtual', c_ulong),
                ]
            memoryStatus = MEMORYSTATUS()
            memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
            kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))
            info['ram_gb'] = memoryStatus.dwTotalPhys / (1024 ** 3)
        else:
            # Linux/Mac
            import psutil
            info['ram_gb'] = psutil.virtual_memory().total / (1024 ** 3)
    except:
        info['ram_gb'] = 8  # Default assumption
    
    return info


def recommend_whisper_model(system_info: dict) -> str:
    """Recommend appropriate Whisper model based on system capabilities."""
    gpu = system_info['gpu_available']
    ram = system_info['ram_gb']
    
    if gpu and ram >= 8:
        return 'small'  # Best balance for GPU systems
    elif ram >= 8:
        return 'base'  # Good for CPU with enough RAM
    else:
        return 'tiny'  # Lightweight for limited systems



def download_whisper_model(model_name: str = 'base', force: bool = False) -> bool:
    """
    Download Whisper model.
    
    Args:
        model_name: Model size (tiny, base, small, medium, large)
        force: Force re-download even if model exists
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Downloading Whisper '{model_name}' model...")
    print(f"{'='*60}")
    
    try:
        import whisper
        
        # Check if model already exists
        model_path = whisper._MODELS[model_name]
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        model_file = os.path.join(cache_dir, os.path.basename(model_path))
        
        if os.path.exists(model_file) and not force:
            print(f"✓ Model '{model_name}' already exists at: {model_file}")
            print("  Use --force to re-download")
            return True
        
        # Download model
        print(f"Downloading from OpenAI...")
        print(f"This may take a few minutes depending on your connection.")
        
        model = whisper.load_model(model_name)
        
        print(f"\n✓ Successfully downloaded Whisper '{model_name}' model!")
        print(f"  Location: {model_file}")
        
        # Show model info
        model_sizes = {
            'tiny': '39 MB',
            'base': '74 MB',
            'small': '244 MB',
            'medium': '769 MB',
            'large': '1.5 GB'
        }
        print(f"  Size: {model_sizes.get(model_name, 'Unknown')}")
        
        return True
        
    except ImportError:
        print("✗ Error: openai-whisper not installed!")
        print("  Install with: pip install openai-whisper")
        return False
    except Exception as e:
        print(f"✗ Error downloading Whisper model: {e}")
        return False


def download_vosk_model(force: bool = False) -> bool:
    """
    Download Vosk Vietnamese model.
    
    Args:
        force: Force re-download even if model exists
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Downloading Vosk Vietnamese model...")
    print(f"{'='*60}")
    
    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-vi-0.4.zip"
    model_name = "vosk-model-small-vi-0.4"
    models_dir = Path("models/vosk")
    model_path = models_dir / model_name
    zip_path = models_dir / f"{model_name}.zip"
    
    # Check if model already exists
    if model_path.exists() and not force:
        print(f"✓ Vosk model already exists at: {model_path}")
        print("  Use --force to re-download")
        return True
    
    # Create models directory
    models_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download model
        print(f"Downloading from: {model_url}")
        print(f"Size: ~32 MB")
        
        def reporthook(block_num, block_size, total_size):
            if not hasattr(reporthook, 'progress_bar'):
                reporthook.progress_bar = DownloadProgressBar(total_size, "Downloading")
            reporthook.progress_bar.update(block_size)
        
        urllib.request.urlretrieve(model_url, zip_path, reporthook)
        if hasattr(reporthook, 'progress_bar'):
            reporthook.progress_bar.finish()
        
        # Extract model
        print(f"Extracting model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        
        # Clean up zip file
        zip_path.unlink()
        
        print(f"\n✓ Successfully downloaded Vosk Vietnamese model!")
        print(f"  Location: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error downloading Vosk model: {e}")
        if zip_path.exists():
            zip_path.unlink()
        return False


def verify_models() -> Tuple[bool, bool]:
    """
    Verify that models are properly installed.
    
    Returns:
        Tuple of (whisper_ok, vosk_ok)
    """
    print(f"\n{'='*60}")
    print("Verifying installed models...")
    print(f"{'='*60}")
    
    whisper_ok = False
    vosk_ok = False
    
    # Check Whisper
    try:
        import whisper
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "whisper")
        if os.path.exists(cache_dir):
            models = [f for f in os.listdir(cache_dir) if f.endswith('.pt')]
            if models:
                print(f"\n✓ Whisper models found:")
                for model in models:
                    size_mb = os.path.getsize(os.path.join(cache_dir, model)) / (1024 * 1024)
                    print(f"  - {model} ({size_mb:.1f} MB)")
                whisper_ok = True
            else:
                print(f"\n✗ No Whisper models found in cache")
        else:
            print(f"\n✗ Whisper cache directory not found")
    except ImportError:
        print(f"\n✗ Whisper not installed")
    
    # Check Vosk
    vosk_dir = Path("models/vosk")
    if vosk_dir.exists():
        models = [d for d in vosk_dir.iterdir() if d.is_dir() and d.name.startswith('vosk-model')]
        if models:
            print(f"\n✓ Vosk models found:")
            for model in models:
                print(f"  - {model.name}")
            vosk_ok = True
        else:
            print(f"\n✗ No Vosk models found in {vosk_dir}")
    else:
        print(f"\n✗ Vosk models directory not found")
    
    return whisper_ok, vosk_ok


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Download speech recognition models for the emotion detection system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download recommended models automatically
  python download_speech_models.py --auto
  
  # Download specific Whisper model
  python download_speech_models.py --whisper base
  
  # Download both Whisper and Vosk
  python download_speech_models.py --whisper small --vosk
  
  # Force re-download
  python download_speech_models.py --whisper base --force
  
  # Just verify existing models
  python download_speech_models.py --verify
        """
    )
    
    parser.add_argument('--whisper', type=str, choices=['tiny', 'base', 'small', 'medium', 'large'],
                        help='Download specific Whisper model')
    parser.add_argument('--vosk', action='store_true',
                        help='Download Vosk Vietnamese model')
    parser.add_argument('--auto', action='store_true',
                        help='Automatically detect system and download recommended models')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if models exist')
    parser.add_argument('--verify', action='store_true',
                        help='Verify installed models')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if not any([args.whisper, args.vosk, args.auto, args.verify]):
        parser.print_help()
        return
    
    print("="*60)
    print("Speech Recognition Model Downloader")
    print("="*60)
    
    # Verify mode
    if args.verify:
        verify_models()
        return
    
    # Auto mode - detect system and recommend
    if args.auto:
        print("\nDetecting system capabilities...")
        system_info = get_system_info()
        
        print(f"\nSystem Information:")
        print(f"  Platform: {system_info['platform']}")
        print(f"  Architecture: {system_info['machine']}")
        print(f"  GPU Available: {'Yes' if system_info['gpu_available'] else 'No'}")
        print(f"  RAM: {system_info['ram_gb']:.1f} GB")
        
        recommended = recommend_whisper_model(system_info)
        print(f"\nRecommended Whisper model: {recommended}")
        
        # Ask user confirmation
        response = input(f"\nDownload Whisper '{recommended}' and Vosk models? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
        
        args.whisper = recommended
        args.vosk = True
    
    success = True
    
    # Download Whisper
    if args.whisper:
        if not download_whisper_model(args.whisper, args.force):
            success = False
    
    # Download Vosk
    if args.vosk:
        if not download_vosk_model(args.force):
            success = False
    
    # Verify installation
    print("\n" + "="*60)
    whisper_ok, vosk_ok = verify_models()
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    if success and (whisper_ok or vosk_ok):
        print("\n✓ Model download completed successfully!")
        print("\nYou can now use the speech analysis features.")
        print("\nNext steps:")
        print("  1. Test audio capture: python scripts/list_audio_devices.py")
        print("  2. Test speech analysis: python scripts/test_speech_analysis.py")
    else:
        print("\n✗ Some models failed to download.")
        print("Please check the errors above and try again.")
        sys.exit(1)


if __name__ == '__main__':
    main()
