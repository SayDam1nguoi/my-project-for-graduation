"""
Video Normalizer Module

This module provides video normalization functionality including FPS normalization,
resolution scaling, sharpening, and denoising.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VideoNormalizer:
    """
    Normalizes video frames to standard format.
    
    Performs:
    - FPS normalization (interpolation/decimation)
    - Resolution normalization with aspect ratio preservation
    - Sharpening to enhance facial features
    - Noise reduction to remove artifacts
    """
    
    def __init__(
        self,
        target_fps: int = 30,
        target_resolution: Tuple[int, int] = (640, 480),
        enable_sharpening: bool = True,
        sharpen_strength: float = 0.5,
        enable_denoising: bool = True,
        denoise_strength: int = 10,
        denoise_method: str = 'fast_nlmeans'
    ):
        """
        Initialize VideoNormalizer.
        
        Args:
            target_fps: Target frames per second
            target_resolution: Target resolution (width, height)
            enable_sharpening: Enable sharpening filter
            sharpen_strength: Sharpening strength (0.0-1.0)
            enable_denoising: Enable noise reduction
            denoise_strength: Denoising strength (1-20)
            denoise_method: Denoising method ('fast_nlmeans' or 'bilateral')
        """
        self.target_fps = target_fps
        self.target_resolution = target_resolution
        self.enable_sharpening = enable_sharpening
        self.sharpen_strength = sharpen_strength
        self.enable_denoising = enable_denoising
        self.denoise_strength = denoise_strength
        self.denoise_method = denoise_method
        
        # Frame interpolation state
        self.last_frame = None
        self.frame_accumulator = 0.0
        
        logger.info(f"VideoNormalizer initialized: target_fps={target_fps}, "
                   f"target_resolution={target_resolution}")
    
    def normalize_frame(
        self,
        frame: np.ndarray,
        source_fps: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply all normalization steps to a frame.
        
        Args:
            frame: Input frame
            source_fps: Source video FPS (for FPS normalization)
            
        Returns:
            Normalized frame
        """
        # Resolution normalization
        frame = self.normalize_resolution(frame)
        
        # Denoising (before sharpening for better results)
        if self.enable_denoising:
            frame = self.denoise(frame)
        
        # Sharpening
        if self.enable_sharpening:
            frame = self.sharpen(frame)
        
        return frame
    
    def normalize_resolution(
        self,
        frame: np.ndarray,
        preserve_aspect_ratio: bool = True
    ) -> np.ndarray:
        """
        Normalize frame resolution to target size.
        
        Args:
            frame: Input frame
            preserve_aspect_ratio: Whether to preserve aspect ratio
            
        Returns:
            Resized frame
        """
        if frame.shape[:2][::-1] == self.target_resolution:
            return frame
        
        h, w = frame.shape[:2]
        target_w, target_h = self.target_resolution
        
        if preserve_aspect_ratio:
            # Calculate scaling factor to fit within target resolution
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Resize frame
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Create canvas with target size
            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            
            # Center the resized frame
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return canvas
        else:
            # Direct resize without preserving aspect ratio
            return cv2.resize(frame, self.target_resolution, interpolation=cv2.INTER_AREA)
    
    def sharpen(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter to enhance facial features.
        
        Uses unsharp mask technique:
        sharpened = original + strength * (original - blurred)
        
        Args:
            frame: Input frame
            
        Returns:
            Sharpened frame
        """
        if self.sharpen_strength <= 0:
            return frame
        
        # Create Gaussian blur
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(
            frame, 1.0 + self.sharpen_strength,
            blurred, -self.sharpen_strength,
            0
        )
        
        return sharpened
    
    def denoise(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to remove visual artifacts.
        
        Args:
            frame: Input frame
            
        Returns:
            Denoised frame
        """
        if self.denoise_strength <= 0:
            return frame
        
        if self.denoise_method == 'fast_nlmeans':
            # Non-local means denoising (better quality, slower)
            denoised = cv2.fastNlMeansDenoisingColored(
                frame,
                None,
                h=self.denoise_strength,
                hColor=self.denoise_strength,
                templateWindowSize=7,
                searchWindowSize=21
            )
        elif self.denoise_method == 'bilateral':
            # Bilateral filter (faster, good edge preservation)
            denoised = cv2.bilateralFilter(
                frame,
                d=9,
                sigmaColor=self.denoise_strength * 7.5,
                sigmaSpace=self.denoise_strength * 7.5
            )
        else:
            logger.warning(f"Unknown denoise method: {self.denoise_method}")
            return frame
        
        return denoised
    
    def should_keep_frame(
        self,
        frame_number: int,
        source_fps: float
    ) -> bool:
        """
        Determine if a frame should be kept based on FPS normalization.
        
        For downsampling (e.g., 60fps -> 30fps): skip frames
        For upsampling (e.g., 24fps -> 30fps): duplicate frames
        
        Args:
            frame_number: Current frame number
            source_fps: Source video FPS
            
        Returns:
            True if frame should be kept/processed
        """
        if source_fps == self.target_fps:
            return True
        
        # Calculate frame ratio
        ratio = self.target_fps / source_fps
        
        # Update accumulator
        self.frame_accumulator += ratio
        
        # Check if we should output a frame
        if self.frame_accumulator >= 1.0:
            self.frame_accumulator -= 1.0
            return True
        
        return False
    
    def interpolate_frame(
        self,
        frame1: np.ndarray,
        frame2: np.ndarray,
        alpha: float
    ) -> np.ndarray:
        """
        Interpolate between two frames for smooth FPS upsampling.
        
        Args:
            frame1: First frame
            frame2: Second frame
            alpha: Interpolation factor (0.0-1.0)
            
        Returns:
            Interpolated frame
        """
        return cv2.addWeighted(frame1, 1.0 - alpha, frame2, alpha, 0)
    
    def reset(self):
        """Reset internal state for new video."""
        self.last_frame = None
        self.frame_accumulator = 0.0
