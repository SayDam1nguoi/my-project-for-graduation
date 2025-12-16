"""
Base emotion model architectures for facial emotion recognition.

Supports multiple backbone architectures with configurable parameters,
transfer learning capabilities, and 7-class emotion classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Dict, List
import warnings


class BaseEmotionModel(nn.Module):
    """
    Base emotion recognition model with configurable backbone.
    
    Supports multiple pre-trained backbones from torchvision:
    - efficientnet_b2, efficientnet_b3
    - resnet101
    - vit_b16 (Vision Transformer)
    
    Features:
    - 7-class emotion output (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
    - Dropout layers (p=0.3) for regularization
    - Batch normalization
    - Transfer learning support (freeze/unfreeze backbone)
    - Pre-trained ImageNet weights
    """
    
    # Standard 7 emotions
    EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    NUM_CLASSES = 7
    
    # Supported backbones and their feature dimensions
    BACKBONE_CONFIGS = {
        'efficientnet_b2': {
            'model_fn': models.efficientnet_b2,
            'feature_dim': 1408,
            'weights': models.EfficientNet_B2_Weights.IMAGENET1K_V1
        },
        'efficientnet_b3': {
            'model_fn': models.efficientnet_b3,
            'feature_dim': 1536,
            'weights': models.EfficientNet_B3_Weights.IMAGENET1K_V1
        },
        'resnet101': {
            'model_fn': models.resnet101,
            'feature_dim': 2048,
            'weights': models.ResNet101_Weights.IMAGENET1K_V1
        },
        'vit_b16': {
            'model_fn': models.vit_b_16,
            'feature_dim': 768,
            'weights': models.ViT_B_16_Weights.IMAGENET1K_V1
        }
    }
    
    def __init__(
        self,
        backbone: str = 'efficientnet_b2',
        pretrained: bool = True,
        dropout_p: float = 0.3,
        num_classes: int = 7
    ):
        """
        Initialize BaseEmotionModel.
        
        Args:
            backbone: Backbone architecture name
                     ('efficientnet_b2', 'efficientnet_b3', 'resnet101', 'vit_b16')
            pretrained: Whether to load ImageNet pre-trained weights
            dropout_p: Dropout probability (default: 0.3)
            num_classes: Number of emotion classes (default: 7)
        
        Raises:
            ValueError: If backbone is not supported
        """
        super().__init__()
        
        if backbone not in self.BACKBONE_CONFIGS:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Supported: {list(self.BACKBONE_CONFIGS.keys())}"
            )
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        
        # Get backbone configuration
        config = self.BACKBONE_CONFIGS[backbone]
        self.feature_dim = config['feature_dim']
        
        # Load backbone model
        weights = config['weights'] if pretrained else None
        self.backbone = self._create_backbone(config['model_fn'], weights)
        
        # Create classification head
        self.classifier = self._create_classifier()
        
        # Track frozen state
        self._backbone_frozen = False
        
    def _create_backbone(self, model_fn, weights) -> nn.Module:
        """
        Create backbone model and remove original classifier.
        
        Args:
            model_fn: Model constructor function
            weights: Pre-trained weights
            
        Returns:
            Backbone model without classifier
        """
        # Load model with weights
        if weights is not None:
            model = model_fn(weights=weights)
        else:
            model = model_fn()
        
        # Remove classifier based on architecture
        if 'efficientnet' in self.backbone_name:
            # EfficientNet: remove classifier
            model.classifier = nn.Identity()
        elif 'resnet' in self.backbone_name:
            # ResNet: remove fc layer
            model.fc = nn.Identity()
        elif 'vit' in self.backbone_name:
            # Vision Transformer: remove heads
            model.heads = nn.Identity()
        
        return model
    
    def _create_classifier(self) -> nn.Module:
        """
        Create classification head with dropout and batch normalization.
        
        Returns:
            Classification head module
        """
        return nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(512, self.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Logits tensor of shape (batch_size, num_classes)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def freeze_backbone(self):
        """
        Freeze backbone layers for transfer learning.
        
        Only the classification head will be trainable.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self._backbone_frozen = True
        print(f"Backbone '{self.backbone_name}' frozen. Only classifier is trainable.")
    
    def unfreeze_backbone(self):
        """
        Unfreeze backbone layers for fine-tuning.
        
        All layers will be trainable.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        self._backbone_frozen = False
        print(f"Backbone '{self.backbone_name}' unfrozen. All layers are trainable.")
    
    def is_backbone_frozen(self) -> bool:
        """
        Check if backbone is frozen.
        
        Returns:
            True if backbone is frozen, False otherwise
        """
        return self._backbone_frozen
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """
        Get list of trainable parameters.
        
        Returns:
            List of trainable parameters
        """
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count total and trainable parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': total_params - trainable_params
        }
    
    def summary(self):
        """
        Print model architecture summary.
        
        Displays:
        - Backbone architecture
        - Number of parameters (total, trainable, frozen)
        - Classifier architecture
        - Frozen state
        """
        print("=" * 70)
        print(f"BaseEmotionModel Summary")
        print("=" * 70)
        print(f"Backbone: {self.backbone_name}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Dropout probability: {self.dropout_p}")
        print(f"Backbone frozen: {self._backbone_frozen}")
        print()
        
        # Parameter counts
        param_counts = self.count_parameters()
        print(f"Parameters:")
        print(f"  Total:     {param_counts['total']:,}")
        print(f"  Trainable: {param_counts['trainable']:,}")
        print(f"  Frozen:    {param_counts['frozen']:,}")
        print()
        
        # Classifier architecture
        print("Classifier architecture:")
        print(self.classifier)
        print("=" * 70)


def create_model(
    backbone: str = 'efficientnet_b2',
    pretrained: bool = True,
    dropout_p: float = 0.3,
    num_classes: int = 7,
    freeze_backbone: bool = False
) -> BaseEmotionModel:
    """
    Factory function to create emotion recognition model.
    
    Args:
        backbone: Backbone architecture name
        pretrained: Whether to load ImageNet pre-trained weights
        dropout_p: Dropout probability
        num_classes: Number of emotion classes
        freeze_backbone: Whether to freeze backbone initially
        
    Returns:
        Initialized BaseEmotionModel
        
    Example:
        >>> # Create EfficientNet-B2 model with frozen backbone
        >>> model = create_model('efficientnet_b2', pretrained=True, freeze_backbone=True)
        >>> model.summary()
        
        >>> # Create ResNet-101 model for fine-tuning
        >>> model = create_model('resnet101', pretrained=True, freeze_backbone=False)
    """
    model = BaseEmotionModel(
        backbone=backbone,
        pretrained=pretrained,
        dropout_p=dropout_p,
        num_classes=num_classes
    )
    
    if freeze_backbone:
        model.freeze_backbone()
    
    return model


# Convenience functions for specific architectures
def create_efficientnet_b2(pretrained: bool = True, freeze_backbone: bool = False) -> BaseEmotionModel:
    """Create EfficientNet-B2 based emotion model."""
    return create_model('efficientnet_b2', pretrained, freeze_backbone=freeze_backbone)


def create_efficientnet_b3(pretrained: bool = True, freeze_backbone: bool = False) -> BaseEmotionModel:
    """Create EfficientNet-B3 based emotion model."""
    return create_model('efficientnet_b3', pretrained, freeze_backbone=freeze_backbone)


def create_resnet101(pretrained: bool = True, freeze_backbone: bool = False) -> BaseEmotionModel:
    """Create ResNet-101 based emotion model."""
    return create_model('resnet101', pretrained, freeze_backbone=freeze_backbone)


def create_vit_b16(pretrained: bool = True, freeze_backbone: bool = False) -> BaseEmotionModel:
    """Create Vision Transformer (ViT-B/16) based emotion model."""
    return create_model('vit_b16', pretrained, freeze_backbone=freeze_backbone)


if __name__ == '__main__':
    # Demo: Create and test models
    print("Testing BaseEmotionModel with different backbones...\n")
    
    backbones = ['efficientnet_b2', 'efficientnet_b3', 'resnet101', 'vit_b16']
    
    for backbone in backbones:
        print(f"\nTesting {backbone}:")
        print("-" * 70)
        
        # Create model
        model = create_model(backbone, pretrained=True, freeze_backbone=False)
        model.summary()
        
        # Test forward pass
        batch_size = 4
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"\nForward pass test:")
        print(f"  Input shape:  {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected:     ({batch_size}, {model.num_classes})")
        
        # Test freeze/unfreeze
        print(f"\nTesting freeze/unfreeze:")
        model.freeze_backbone()
        params_frozen = model.count_parameters()
        print(f"  Trainable params (frozen):   {params_frozen['trainable']:,}")
        
        model.unfreeze_backbone()
        params_unfrozen = model.count_parameters()
        print(f"  Trainable params (unfrozen): {params_unfrozen['trainable']:,}")
        
        print("\n" + "=" * 70)
