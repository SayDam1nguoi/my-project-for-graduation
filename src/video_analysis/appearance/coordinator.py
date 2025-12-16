# -*- coding: utf-8 -*-
"""
Appearance Assessment Coordinator

Coordinates lighting and clothing assessments, manages configuration,
and integrates with the existing video analysis pipeline.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import AppearanceConfig
from .lighting_assessor import LightingAssessor
from .clothing_assessor import ClothingAssessor
from .score_calculator import AppearanceScoreCalculator
from .models import AppearanceAssessment, LightingAssessment, ClothingAssessment


logger = logging.getLogger(__name__)


class AppearanceAssessmentCoordinator:
    """
    Coordinates appearance assessment (lighting and clothing).
    
    Responsibilities:
    - Initialize assessors based on configuration
    - Extract face and upper body regions
    - Run assessments in parallel
    - Calculate overall scores
    - Handle errors gracefully
    """
    
    def __init__(self, config: AppearanceConfig):
        """
        Initialize coordinator with configuration.
        
        Args:
            config: AppearanceConfig instance
        """
        self.config = config
        self.frame_count = 0
        
        # Initialize assessors based on config
        self.lighting_assessor = None
        self.clothing_assessor = None
        self.score_calculator = AppearanceScoreCalculator()
        
        self._initialize_assessors()
    
    def _initialize_assessors(self):
        """Initialize assessors based on current configuration."""
        # Initialize lighting assessor if enabled
        if self.config.lighting_enabled:
            self.lighting_assessor = LightingAssessor(
                min_brightness=self.config.min_brightness,
                max_brightness=self.config.max_brightness,
                min_contrast=self.config.min_contrast,
                min_uniformity=self.config.min_uniformity
            )
            logger.info("Lighting assessor initialized")
        else:
            self.lighting_assessor = None
            logger.info("Lighting assessor disabled")
        
        # Initialize clothing assessor if enabled
        if self.config.clothing_enabled:
            self.clothing_assessor = ClothingAssessor()
            logger.info("Clothing assessor initialized")
        else:
            self.clothing_assessor = None
            logger.info("Clothing assessor disabled")
    
    def assess_appearance(
        self,
        frame: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> AppearanceAssessment:
        """
        Assess appearance (lighting and clothing) for a frame.
        
        Args:
            frame: BGR frame from video
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            AppearanceAssessment with lighting, clothing, and overall scores
        """
        self.frame_count += 1
        
        try:
            # Extract regions
            face_region = self._extract_face_region(frame, face_bbox)
            upper_body_region = self._extract_upper_body_region(frame, face_bbox)
            
            # Run assessments
            lighting_assessment = None
            clothing_assessment = None
            
            # If both assessors are enabled, run in parallel
            if self.lighting_assessor and self.clothing_assessor:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit tasks
                    lighting_future = executor.submit(
                        self.lighting_assessor.assess,
                        frame,
                        face_region
                    )
                    clothing_future = executor.submit(
                        self.clothing_assessor.assess,
                        frame,
                        upper_body_region,
                        face_bbox
                    )
                    
                    # Get results
                    lighting_assessment = lighting_future.result()
                    clothing_assessment = clothing_future.result()
            else:
                # Run sequentially if only one is enabled
                if self.lighting_assessor:
                    lighting_assessment = self.lighting_assessor.assess(frame, face_region)
                
                if self.clothing_assessor:
                    clothing_assessment = self.clothing_assessor.assess(frame, upper_body_region, face_bbox)
            
            # Calculate overall score
            lighting_score = lighting_assessment.score if lighting_assessment else None
            clothing_score = clothing_assessment.score if clothing_assessment else None
            
            overall_score = self.score_calculator.calculate_overall_score(
                lighting_score,
                clothing_score
            )
            
            is_interview_ready = self.score_calculator.is_interview_ready(overall_score)
            
            return AppearanceAssessment(
                lighting=lighting_assessment,
                clothing=clothing_assessment,
                overall_score=overall_score,
                is_interview_ready=is_interview_ready,
                timestamp=time.time(),
                frame_number=self.frame_count
            )
        
        except Exception as e:
            logger.error(f"Error during appearance assessment: {e}", exc_info=True)
            return self._get_neutral_assessment()
    
    def update_config(self, config: AppearanceConfig):
        """
        Update configuration and reinitialize assessors.
        
        Args:
            config: New AppearanceConfig instance
        """
        self.config = config
        self._initialize_assessors()
        logger.info("Configuration updated and assessors reinitialized")
    
    def _extract_face_region(
        self,
        frame: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract face region from frame.
        
        Args:
            frame: BGR frame
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Face region as numpy array
        """
        x, y, w, h = face_bbox
        
        # Ensure coordinates are within frame bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        # Extract region
        face_region = frame[y:y+h, x:x+w]
        
        # Ensure region is not empty
        if face_region.size == 0:
            # Return a small default region if extraction fails
            logger.warning("Face region extraction resulted in empty region, using default")
            return np.zeros((50, 50, 3), dtype=np.uint8)
        
        return face_region
    
    def _extract_upper_body_region(
        self,
        frame: np.ndarray,
        face_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Extract upper body region from frame based on face location.
        
        Estimates upper body as region below face, approximately 2x face width
        and 1.5x face height.
        
        Args:
            frame: BGR frame
            face_bbox: Face bounding box (x, y, w, h)
            
        Returns:
            Upper body region as numpy array
        """
        x, y, w, h = face_bbox
        
        # Estimate upper body region
        # Start below the face
        body_y = y + h
        # Extend downward by 1.5x face height
        body_h = int(h * 1.5)
        # Center horizontally with 2x face width
        body_w = int(w * 2)
        body_x = max(0, x - w // 2)
        
        # Ensure coordinates are within frame bounds
        body_x = max(0, body_x)
        body_y = max(0, body_y)
        body_w = min(body_w, frame.shape[1] - body_x)
        body_h = min(body_h, frame.shape[0] - body_y)
        
        # Extract region
        upper_body_region = frame[body_y:body_y+body_h, body_x:body_x+body_w]
        
        # Ensure region is not empty
        if upper_body_region.size == 0:
            # Return a small default region if extraction fails
            logger.warning("Upper body region extraction resulted in empty region, using default")
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        return upper_body_region
    
    def _get_neutral_assessment(self) -> AppearanceAssessment:
        """
        Return neutral assessment when errors occur.
        
        Returns:
            AppearanceAssessment with neutral scores (50.0)
        """
        return AppearanceAssessment(
            lighting=None,
            clothing=None,
            overall_score=50.0,
            is_interview_ready=False,
            timestamp=time.time(),
            frame_number=self.frame_count
        )
