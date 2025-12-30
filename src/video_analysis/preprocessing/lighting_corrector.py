"""
Lighting Corrector Module

This module provides lighting and color correction functionality including
CLAHE enhancement, white balance, and contrast adjustment.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LightingCorrector:
    """
    Corrects lighting and color in face regions.
    
    Applies:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization)
    - White balance correction
    - Contrast enhancement
    - Skin tone preservation
    """
    
    def __init__(
        self,
        enable_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
        enable_white_balance: bool = True,
        white_balance_method: str = "gray_world",
        enable_contrast: bool = True,
        contrast_factor: float = 1.2
    ):
        """
        Initialize LightingCorrector.
        
        Args:
            enable_clahe: Enable CLAHE enhancement
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_size: CLAHE tile grid size
            enable_white_balance: Enable white balance correction
            white_balance_method: White balance method ('gray_world' or 'retinex')
            enable_contrast: Enable contrast enhancement
            contrast_factor: Contrast multiplication factor
        """
        self.enable_clahe = enable_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.enable_white_balance = enable_white_balance
        self.white_balance_method = white_balance_method
        self.enable_contrast = enable_contrast
        self.contrast_factor = contrast_factor
        
        # Create CLAHE object
        if self.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=clahe_clip_limit,
                tileGridSize=clahe_tile_size
            )
        
        logger.info(f"LightingCorrector initialized: clahe={enable_clahe}, "
                   f"white_balance={enable_white_balance}, contrast={enable_contrast}")
    
    def correct_lighting(
        self,
        frame: np.ndarray,
        face_bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Apply all lighting corrections to frame.
        
        Args:
            frame: Input frame
            face_bbox: Optional face bounding box to focus corrections
            
        Returns:
            Corrected frame
        """
        corrected = frame.copy()
        
        # Extract face region if provided
        if face_bbox is not None:
            x, y, w, h = face_bbox
            # Ensure coordinates are within bounds
            h_frame, w_frame = corrected.shape[:2]
            x = max(0, min(x, w_frame - 1))
            y = max(0, min(y, h_frame - 1))
            w = min(w, w_frame - x)
            h = min(h, h_frame - y)
            
            face_region = corrected[y:y+h, x:x+w]
            
            # Apply corrections to face region
            face_region = self._apply_corrections(face_region)
            
            # Blend back into frame with smooth transition
            corrected[y:y+h, x:x+w] = face_region
        else:
            # Apply to entire frame
            corrected = self._apply_corrections(corrected)
        
        return corrected
    
    def _apply_corrections(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all enabled corrections to image.
        
        Args:
            image: Input image
            
        Returns:
            Corrected image
        """
        corrected = image.copy()
        
        # White balance (first, to normalize colors)
        if self.enable_white_balance:
            corrected = self.white_balance(corrected)
        
        # CLAHE (to enhance local contrast)
        if self.enable_clahe:
            corrected = self.apply_clahe(corrected)
        
        # Contrast enhancement (last, for overall adjustment)
        if self.enable_contrast:
            corrected = self.enhance_contrast(corrected)
        
        return corrected
    
    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to enhance local contrast.
        
        Converts to LAB color space, applies CLAHE to L channel,
        then converts back to BGR.
        
        Args:
            frame: Input frame in BGR format
            
        Returns:
            Enhanced frame
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split channels
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_clahe = self.clahe.apply(l)
        
        # Merge channels
        lab_clahe = cv2.merge([l_clahe, a, b])
        
        # Convert back to BGR
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def white_balance(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply white balance correction.
        
        Args:
            frame: Input frame
            
        Returns:
            White balanced frame
        """
        if self.white_balance_method == "gray_world":
            return self._gray_world_white_balance(frame)
        elif self.white_balance_method == "retinex":
            return self._retinex_white_balance(frame)
        else:
            logger.warning(f"Unknown white balance method: {self.white_balance_method}")
            return frame
    
    def _gray_world_white_balance(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply Gray World white balance algorithm.
        
        Assumes the average color in the image should be gray.
        
        Args:
            frame: Input frame
            
        Returns:
            White balanced frame
        """
        # Calculate mean of each channel
        b_mean = np.mean(frame[:, :, 0])
        g_mean = np.mean(frame[:, :, 1])
        r_mean = np.mean(frame[:, :, 2])
        
        # Calculate gray value (average of all channels)
        gray_mean = (b_mean + g_mean + r_mean) / 3
        
        # Calculate scaling factors
        b_scale = gray_mean / b_mean if b_mean > 0 else 1.0
        g_scale = gray_mean / g_mean if g_mean > 0 else 1.0
        r_scale = gray_mean / r_mean if r_mean > 0 else 1.0
        
        # Apply scaling
        balanced = frame.astype(np.float32)
        balanced[:, :, 0] *= b_scale
        balanced[:, :, 1] *= g_scale
        balanced[:, :, 2] *= r_scale
        
        # Clip and convert back
        balanced = np.clip(balanced, 0, 255).astype(np.uint8)
        
        return balanced
    
    def _retinex_white_balance(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply simplified Retinex-based white balance.
        
        Args:
            frame: Input frame
            
        Returns:
            White balanced frame
        """
        # Convert to float
        img_float = frame.astype(np.float32) / 255.0
        
        # Apply log transform
        img_log = np.log1p(img_float)
        
        # Calculate illumination (Gaussian blur)
        illumination = cv2.GaussianBlur(img_log, (0, 0), 15)
        
        # Calculate reflectance
        reflectance = img_log - illumination
        
        # Normalize
        reflectance = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min())
        
        # Convert back
        balanced = (reflectance * 255).astype(np.uint8)
        
        return balanced
    
    def enhance_contrast(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance contrast moderately.
        
        Args:
            frame: Input frame
            
        Returns:
            Contrast enhanced frame
        """
        if self.contrast_factor == 1.0:
            return frame
        
        # Convert to float
        img_float = frame.astype(np.float32)
        
        # Calculate mean
        mean = np.mean(img_float)
        
        # Apply contrast adjustment
        enhanced = mean + self.contrast_factor * (img_float - mean)
        
        # Clip and convert back
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def preserve_skin_tones(
        self,
        original: np.ndarray,
        corrected: np.ndarray,
        blend_factor: float = 0.7
    ) -> np.ndarray:
        """
        Preserve natural skin tones by blending with original.
        
        Detects skin regions and reduces correction strength there.
        
        Args:
            original: Original frame
            corrected: Corrected frame
            blend_factor: Blending factor (0.0-1.0)
            
        Returns:
            Frame with preserved skin tones
        """
        # Convert to YCrCb for skin detection
        ycrcb = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
        
        # Smooth mask
        skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)
        skin_mask = skin_mask.astype(np.float32) / 255.0
        
        # Expand mask to 3 channels
        skin_mask_3ch = np.stack([skin_mask for _ in range(3)], axis=2)
        
        # Blend: more original in skin regions, more corrected elsewhere
        blended = (
            corrected * (1 - skin_mask_3ch * (1 - blend_factor)) +
            original * skin_mask_3ch * (1 - blend_factor)
        )
        
        return blended.astype(np.uint8)
