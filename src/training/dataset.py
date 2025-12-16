"""
PyTorch Dataset and DataLoader for emotion recognition training.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import pandas as pd
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EmotionDataset(Dataset):
    """
    PyTorch Dataset for emotion recognition.
    
    Loads images from CSV file with emotion labels and applies data augmentation.
    Supports both absolute and relative image paths.
    """
    
    # Standard 7 emotions mapping
    EMOTION_LABELS = {
        'Angry': 0,
        'Disgust': 1,
        'Fear': 2,
        'Happy': 3,
        'Sad': 4,
        'Surprise': 5,
        'Neutral': 6
    }
    
    def __init__(
        self,
        csv_path: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize EmotionDataset.
        
        Args:
            csv_path: Path to CSV file containing image paths and labels
            split: Dataset split ('train', 'val', 'test', or 'train_set'/'test_set' for RAF-DB)
            transform: Albumentations transform pipeline (if None, uses default)
            image_size: Target image size (height, width)
        """
        self.csv_path = csv_path
        self.split = split
        self.image_size = image_size
        
        # Load CSV
        self.df = pd.read_csv(csv_path)
        
        # Filter by split
        # Handle both 'train'/'test' and 'train_set'/'test_set' formats
        if split in ['train', 'val', 'test']:
            self.df = self.df[self.df['split'] == split]
        elif split in ['train_set', 'test_set']:
            self.df = self.df[self.df['split'] == split]
        else:
            # If split not found, use all data
            print(f"Warning: Split '{split}' not found in CSV. Using all data.")
        
        # Reset index after filtering
        self.df = self.df.reset_index(drop=True)
        
        # Set transform
        if transform is None:
            self.transform = self._get_default_transform(split)
        else:
            self.transform = transform
        
        print(f"Loaded {len(self.df)} samples for split '{split}'")
        
    def _get_default_transform(self, split: str) -> A.Compose:
        """
        Get default augmentation pipeline based on split.
        
        Args:
            split: Dataset split
            
        Returns:
            Albumentations transform pipeline
        """
        if split in ['train', 'train_set']:
            # Training augmentation: aggressive
            return A.Compose([
                A.Resize(256, 256),
                A.RandomCrop(self.image_size[0], self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1,
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(8, 16),
                    hole_width_range=(8, 16),
                    p=0.3
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            # Validation/Test augmentation: minimal
            return A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load image from path, handling both absolute and relative paths.
        
        Args:
            image_path: Path to image (absolute or relative)
            
        Returns:
            Image as numpy array (RGB)
        """
        # Try as absolute path first
        import re
        
        # AUTO-FIX: Remove Windows absolute paths (TRIỆT ĐỂ)
        fixed_path = image_path
        
        # Remove all Windows absolute path patterns
        patterns_to_remove = [
            r'C:/Users/[^/]+/Downloads/',
            r'C:\\Users\\[^\\]+\\Downloads\\',
            r'^[A-Z]:/Users/[^/]+/Downloads/',
            r'^[A-Z]:\\Users\\[^\\]+\\Downloads\\',
        ]
        
        for pattern in patterns_to_remove:
            fixed_path = re.sub(pattern, '', fixed_path, flags=re.IGNORECASE)
        
        # Replace all backslashes with forward slashes
        fixed_path = fixed_path.replace('\\', '/')
        
        # Ensure path starts with data/raw/ if it contains dataset names
        if not fixed_path.startswith('data/raw/'):
            dataset_names = ['affectnet', 'fer2013', 'ckplus', 'rafdb']
            for dataset in dataset_names:
                if dataset.lower() in fixed_path.lower():
                    idx = fixed_path.lower().find(dataset.lower())
                    fixed_path = 'data/raw/' + fixed_path[idx:]
                    break
        
        # Try to load with fixed path
        if os.path.isabs(fixed_path) and os.path.exists(fixed_path):
            img = cv2.imread(fixed_path)
        else:
            # Try relative to CSV directory
            csv_dir = Path(self.csv_path).parent
            full_path = csv_dir / fixed_path
            
            if full_path.exists():
                img = cv2.imread(str(full_path))
            else:
                # Try relative to current working directory
                full_path = Path(fixed_path)
                if full_path.exists():
                    img = cv2.imread(str(full_path))
                else:
                    raise FileNotFoundError(f"Image not found: {image_path} -> fixed to: {fixed_path}")
        
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        # Get row
        row = self.df.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        image = self._load_image(image_path)
        
        # Get label
        emotion_name = row['emotion']
        label = self.EMOTION_LABELS[emotion_name]
        
        # Apply transform
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Returns:
            Tensor of class weights (inverse frequency)
        """
        # Count samples per class
        emotion_counts = self.df['emotion'].value_counts()
        
        # Calculate weights (inverse frequency)
        total_samples = len(self.df)
        weights = []
        
        for emotion in ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']:
            count = emotion_counts.get(emotion, 1)  # Avoid division by zero
            weight = total_samples / (len(self.EMOTION_LABELS) * count)
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_emotion_distribution(self) -> Dict[str, int]:
        """
        Get emotion distribution in dataset.
        
        Returns:
            Dictionary mapping emotion names to counts
        """
        return self.df['emotion'].value_counts().to_dict()


def create_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    train_transform: Optional[A.Compose] = None,
    val_transform: Optional[A.Compose] = None
) -> Dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from CSV file.
    
    Args:
        csv_path: Path to CSV file
        batch_size: Batch size for DataLoader
        num_workers: Number of worker processes for data loading
        image_size: Target image size (height, width)
        train_transform: Custom transform for training (optional)
        val_transform: Custom transform for validation/test (optional)
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    # Read CSV to check available splits
    df = pd.read_csv(csv_path)
    available_splits = df['split'].unique()
    
    print(f"Available splits in CSV: {available_splits}")
    
    dataloaders = {}
    
    # Determine split names (handle both formats)
    if 'train' in available_splits:
        train_split = 'train'
        test_split = 'test'
        val_split = 'val' if 'val' in available_splits else None
    elif 'train_set' in available_splits:
        train_split = 'train_set'
        test_split = 'test_set'
        val_split = None
    else:
        raise ValueError(f"No recognized train split found in CSV. Available: {available_splits}")
    
    # Create train dataset
    train_dataset = EmotionDataset(
        csv_path=csv_path,
        split=train_split,
        transform=train_transform,
        image_size=image_size
    )
    
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create validation dataset if available
    if val_split and val_split in available_splits:
        val_dataset = EmotionDataset(
            csv_path=csv_path,
            split=val_split,
            transform=val_transform,
            image_size=image_size
        )
        
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        # If no validation split, create one from train (80/20 split)
        print("No validation split found. Creating from train split (80/20)...")
        
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        dataloaders['train'] = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        dataloaders['val'] = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Create test dataset
    test_dataset = EmotionDataset(
        csv_path=csv_path,
        split=test_split,
        transform=val_transform,
        image_size=image_size
    )
    
    dataloaders['test'] = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    print(f"Train samples: {len(dataloaders['train'].dataset)}")
    print(f"Val samples: {len(dataloaders['val'].dataset)}")
    print(f"Test samples: {len(dataloaders['test'].dataset)}")
    
    # Print emotion distribution for train set
    if hasattr(dataloaders['train'].dataset, 'get_emotion_distribution'):
        print("\nTrain emotion distribution:")
        dist = dataloaders['train'].dataset.get_emotion_distribution()
        for emotion, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            print(f"  {emotion}: {count}")
    
    return dataloaders


def get_train_transform(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    """
    Get training augmentation pipeline.
    
    Args:
        image_size: Target image size (height, width)
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(8, 16),
            hole_width_range=(8, 16),
            p=0.3
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transform(image_size: Tuple[int, int] = (224, 224)) -> A.Compose:
    """
    Get validation/test augmentation pipeline.
    
    Args:
        image_size: Target image size (height, width)
        
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
