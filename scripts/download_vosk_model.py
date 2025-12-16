"""
Script to download Vosk Vietnamese model.

Downloads the Vosk Vietnamese model from the official repository
and extracts it to the models directory.
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path


# Vosk Vietnamese models
# Note: Check https://alphacephei.com/vosk/models for latest versions
VOSK_MODELS = {
    "small": {
        "name": "vosk-model-small-vi-0.3",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-vi-0.3.zip",
        "size": "~32 MB",
        "description": "Small Vietnamese model, good for real-time applications",
        "alternative_url": "https://github.com/alphacep/vosk-api/releases/download/v0.3.32/vosk-model-small-vi-0.3.zip"
    },
    "large": {
        "name": "vosk-model-vi-0.3",
        "url": "https://alphacephei.com/vosk/models/vosk-model-vi-0.3.zip",
        "size": "~78 MB",
        "description": "Large Vietnamese model, better accuracy",
        "alternative_url": "https://github.com/alphacep/vosk-api/releases/download/v0.3.32/vosk-model-vi-0.3.zip"
    }
}


def download_file(url: str, destination: str) -> None:
    """
    Download file with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
    """
    print(f"Downloading from {url}...")
    
    def reporthook(count, block_size, total_size):
        """Progress callback."""
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rProgress: {percent}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, destination, reporthook)
    print("\nDownload complete!")


def extract_zip(zip_path: str, extract_to: str) -> None:
    """
    Extract ZIP file.
    
    Args:
        zip_path: Path to ZIP file
        extract_to: Directory to extract to
    """
    print(f"Extracting {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print("Extraction complete!")


def download_vosk_model(model_type: str = "small") -> None:
    """
    Download and setup Vosk Vietnamese model.
    
    Args:
        model_type: Type of model to download ("small" or "large")
    """
    if model_type not in VOSK_MODELS:
        print(f"Error: Invalid model type '{model_type}'")
        print(f"Available models: {', '.join(VOSK_MODELS.keys())}")
        return
    
    model_info = VOSK_MODELS[model_type]
    model_name = model_info["name"]
    model_url = model_info["url"]
    alternative_url = model_info.get("alternative_url")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Check if model already exists
    model_path = models_dir / model_name
    if model_path.exists():
        print(f"Model '{model_name}' already exists at {model_path}")
        response = input("Do you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
        
        # Remove existing model
        shutil.rmtree(model_path)
    
    # Download model
    zip_filename = f"{model_name}.zip"
    zip_path = models_dir / zip_filename
    
    try:
        print(f"\nDownloading Vosk {model_type} Vietnamese model...")
        print(f"Size: {model_info['size']}")
        print(f"Description: {model_info['description']}")
        print()
        
        # Try primary URL first
        try:
            download_file(model_url, str(zip_path))
        except Exception as e:
            if alternative_url:
                print(f"\nPrimary URL failed: {e}")
                print(f"Trying alternative URL...")
                download_file(alternative_url, str(zip_path))
            else:
                raise
        
        # Extract model
        extract_zip(str(zip_path), str(models_dir))
        
        # Remove ZIP file
        zip_path.unlink()
        
        print(f"\nModel successfully installed at: {model_path}")
        print(f"\nTo use this model, update your config file:")
        print(f"  vosk_model_path: \"{model_name}\"")
        
    except Exception as e:
        print(f"\nError downloading model: {e}")
        
        # Cleanup on error
        if zip_path.exists():
            zip_path.unlink()
        if model_path.exists():
            shutil.rmtree(model_path)
        
        sys.exit(1)


def list_models() -> None:
    """List available Vosk models."""
    print("\nAvailable Vosk Vietnamese models:\n")
    
    for model_type, info in VOSK_MODELS.items():
        print(f"{model_type.upper()}:")
        print(f"  Name: {info['name']}")
        print(f"  Size: {info['size']}")
        print(f"  Description: {info['description']}")
        print()


def check_installed_models() -> None:
    """Check which models are already installed."""
    models_dir = Path("models")
    
    if not models_dir.exists():
        print("No models directory found.")
        return
    
    print("\nInstalled Vosk models:\n")
    
    found_models = False
    for model_type, info in VOSK_MODELS.items():
        model_name = info['name']
        model_path = models_dir / model_name
        
        if model_path.exists():
            print(f"✓ {model_type.upper()}: {model_name}")
            found_models = True
    
    if not found_models:
        print("No Vosk models installed.")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Vosk Vietnamese speech recognition models"
    )
    parser.add_argument(
        "--model",
        choices=["small", "large"],
        default="small",
        help="Model size to download (default: small)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check installed models"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    if args.check:
        check_installed_models()
        return
    
    # Download model
    download_vosk_model(args.model)


if __name__ == "__main__":
    main()
