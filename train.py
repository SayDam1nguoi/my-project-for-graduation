"""
Training pipeline for emotion recognition models.

This script trains emotion classification models with:
- Multiple backbone architectures (EfficientNet, ResNet, ViT)
- Mixed precision training for speed
- Learning rate scheduling with warm restarts
- Early stopping based on validation accuracy
- TensorBoard logging
- Model checkpointing
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.training.dataset import create_dataloaders, EmotionDataset
from src.training.models import create_model


class EarlyStopping:
    """Early stopping to stop training when validation accuracy stops improving."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for accuracy (higher is better), 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score (accuracy or loss)
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        # Check if score improved
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        
        return False


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: float = 1.0
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        scaler: Gradient scaler for mixed precision (optional)
        max_grad_norm: Maximum gradient norm for clipping
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training', leave=False)
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, np.ndarray]:
    """
    Validate model.
    
    Args:
        model: Model to validate
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Tuple of (average_loss, accuracy, per_class_accuracy)
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Per-class accuracy
    class_correct = np.zeros(7)
    class_total = np.zeros(7)
    
    pbar = tqdm(dataloader, desc='Validation', leave=False)
    
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    
    # Calculate per-class accuracy
    per_class_acc = np.zeros(7)
    for i in range(7):
        if class_total[i] > 0:
            per_class_acc[i] = 100.0 * class_correct[i] / class_total[i]
    
    return epoch_loss, epoch_acc, per_class_acc


def main():
    parser = argparse.ArgumentParser(description='Train emotion recognition model')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='efficientnet_b2',
                       choices=['efficientnet_b2', 'efficientnet_b3', 'resnet101', 'vit_b16'],
                       help='Model architecture')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to CSV dataset file')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay for optimizer')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                       help='Maximum gradient norm for clipping')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Use mixed precision training (FP16)')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save models')
    parser.add_argument('--log-dir', type=str, default='runs',
                       help='Directory for TensorBoard logs')
    
    # Early stopping arguments
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Class weights
    parser.add_argument('--use-class-weights', action='store_true',
                       help='Use class weights for imbalanced data')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(args.log_dir) / f"{args.model}_{timestamp}"
    writer = SummaryWriter(log_dir)
    
    print(f"TensorBoard logs: {log_dir}")
    print(f"Run: tensorboard --logdir {args.log_dir}")
    
    # Load data
    print(f"\nLoading dataset from: {args.dataset}")
    dataloaders = create_dataloaders(
        csv_path=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Get class weights if requested
    class_weights = None
    if args.use_class_weights:
        print("\nCalculating class weights...")
        train_dataset = dataloaders['train'].dataset
        
        # Handle subset case (when validation split is created from train)
        if hasattr(train_dataset, 'dataset'):
            train_dataset = train_dataset.dataset
        
        class_weights = train_dataset.get_class_weights().to(device)
        print("Class weights:", class_weights.cpu().numpy())
    
    # Create model
    print(f"\nCreating model: {args.model}")
    model = create_model(
        backbone=args.model,
        pretrained=True,
        freeze_backbone=False
    )
    model = model.to(device)
    model.summary()
    
    # Loss function
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler (Cosine Annealing with Warm Restarts)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the restart period after each restart
        eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler() if args.mixed_precision and device.type == 'cuda' else None
    if scaler:
        print("Using mixed precision training (FP16)")
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    best_val_acc = 0.0
    best_model_path = None
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=dataloaders['train'],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            max_grad_norm=args.max_grad_norm
        )
        
        # Validate
        val_loss, val_acc, per_class_acc = validate(
            model=model,
            dataloader=dataloaders['val'],
            criterion=criterion,
            device=device
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print results
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Print per-class accuracy
        emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        print("\nPer-class accuracy:")
        for i, (emotion, acc) in enumerate(zip(emotion_names, per_class_acc)):
            print(f"  {emotion:10s}: {acc:.2f}%")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Log per-class accuracy
        for i, (emotion, acc) in enumerate(zip(emotion_names, per_class_acc)):
            writer.add_scalar(f'Accuracy_per_class/{emotion}', acc, epoch)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            # Remove previous best model
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            # Save new best model
            best_model_path = output_dir / f"{args.model}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'per_class_acc': per_class_acc,
                'args': vars(args)
            }, best_model_path)
            
            print(f"\n✓ Best model saved: {best_model_path} (Val Acc: {val_acc:.2f}%)")
        
        # Check early stopping
        if early_stopping(val_acc):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # Save final model
    final_model_path = output_dir / f"{args.model}_final.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'per_class_acc': per_class_acc,
        'args': vars(args)
    }, final_model_path)
    
    print(f"\n✓ Final model saved: {final_model_path}")
    
    # Test on test set
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("-" * 70)
    
    # Load best model
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_per_class_acc = validate(
        model=model,
        dataloader=dataloaders['test'],
        criterion=criterion,
        device=device
    )
    
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    print("\nPer-class accuracy on test set:")
    for emotion, acc in zip(emotion_names, test_per_class_acc):
        print(f"  {emotion:10s}: {acc:.2f}%")
    
    # Log test results
    writer.add_scalar('Test/loss', test_loss, 0)
    writer.add_scalar('Test/accuracy', test_acc, 0)
    for i, (emotion, acc) in enumerate(zip(emotion_names, test_per_class_acc)):
        writer.add_scalar(f'Test_per_class/{emotion}', acc, 0)
    
    # Close writer
    writer.close()
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Best model: {best_model_path}")
    print(f"TensorBoard logs: {log_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
