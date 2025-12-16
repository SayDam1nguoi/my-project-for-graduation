"""
Script to download Whisper models for Vietnamese STT

This script downloads and verifies Whisper models using faster-whisper library.
Models are automatically downloaded from Hugging Face and cached locally.
"""

import argparse
import sys
from pathlib import Path

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper library not installed.")
    print("Please install it with: pip install faster-whisper")
    sys.exit(1)


def download_whisper_model(
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    download_root: str = None
):
    """
    Download and verify Whisper model.
    
    Args:
        model_size: Model size (tiny, base, small, medium, large)
        device: Device to use (cpu or cuda)
        compute_type: Quantization type (int8, int16, float16, float32)
        download_root: Directory to download models to (default: cache directory)
    """
    print(f"Downloading Whisper model: {model_size}")
    print(f"Device: {device}")
    print(f"Compute type: {compute_type}")
    
    if download_root:
        print(f"Download directory: {download_root}")
        Path(download_root).mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize model (this will download if not cached)
        print("\nInitializing model...")
        model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=download_root
        )
        
        print(f"✓ Model '{model_size}' downloaded and verified successfully!")
        
        # Test transcription with a short audio
        print("\nTesting model with sample audio...")
        import numpy as np
        
        # Generate 1 second of silence as test audio
        sample_rate = 16000
        test_audio = np.zeros(sample_rate, dtype=np.float32)
        
        segments, info = model.transcribe(
            test_audio,
            language="vi",
            beam_size=5
        )
        
        print(f"✓ Model test successful!")
        print(f"  Detected language: {info.language}")
        print(f"  Language probability: {info.language_probability:.2f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading or testing model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download Whisper models for Vietnamese STT"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Model size to download (default: base)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        choices=["int8", "int16", "float16", "float32"],
        help="Quantization type (default: int8)"
    )
    parser.add_argument(
        "--download-root",
        type=str,
        default=None,
        help="Directory to download models to (default: cache directory)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all recommended models (tiny, base, small)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Download multiple models
        models = ["tiny", "base", "small"]
        print(f"Downloading {len(models)} models: {', '.join(models)}\n")
        
        success_count = 0
        for model_size in models:
            print(f"\n{'='*60}")
            print(f"Downloading model: {model_size}")
            print(f"{'='*60}")
            
            if download_whisper_model(
                model_size=model_size,
                device=args.device,
                compute_type=args.compute_type,
                download_root=args.download_root
            ):
                success_count += 1
            
            print()
        
        print(f"\n{'='*60}")
        print(f"Downloaded {success_count}/{len(models)} models successfully")
        print(f"{'='*60}")
        
        return success_count == len(models)
    else:
        # Download single model
        return download_whisper_model(
            model_size=args.model,
            device=args.device,
            compute_type=args.compute_type,
            download_root=args.download_root
        )


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
