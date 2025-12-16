"""
Score Fusion Module for Dual Attention Detection System

This module combines attention scores from camera and screen monitoring
to produce a unified attention score.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import time


@dataclass
class ScreenMonitorResult:
    """Result from screen monitoring (imported for type hints)"""
    screen_presence: bool
    face_detected: bool
    face_bbox: Optional[tuple]
    confidence: float
    timestamp: float
    screen_score: float


@dataclass
class FusedScore:
    """
    Combined attention score from camera and screen monitoring.
    
    Attributes:
        combined_score: Final fused attention score (0-100)
        camera_score: Raw camera attention score (0-100)
        screen_score: Raw screen attention score (0-100)
        face_presence: Whether face is detected in camera
        screen_presence: Whether face is detected on screen
        fusion_mode: Mode used for fusion ("full", "camera_only", "screen_only", "none")
        timestamp: Time when fusion was calculated
    """
    combined_score: float
    camera_score: float
    screen_score: float
    face_presence: bool
    screen_presence: bool
    fusion_mode: str
    timestamp: float


class ScoreFusion:
    """
    Fuses attention scores from camera and screen monitoring.
    
    Fusion Modes:
    - "full": Both camera and screen have data (60% camera + 40% screen)
    - "camera_only": Only camera data available (30% penalty applied)
    - "screen_only": Only screen data available (fixed score of 20)
    - "none": No data from either source (score = 0)
    """
    
    def __init__(
        self,
        camera_weight: float = 0.6,
        screen_weight: float = 0.4
    ):
        """
        Initialize ScoreFusion.
        
        Args:
            camera_weight: Weight for camera score in full mode (default: 0.6)
            screen_weight: Weight for screen score in full mode (default: 0.4)
        """
        self.camera_weight = camera_weight
        self.screen_weight = screen_weight
        self.camera_only_penalty = 0.3  # Penalty multiplier when screen not present
        self.screen_only_score = 20.0  # Fixed score when only screen present
        
        # Statistics for debugging
        self._last_fusion_details: Dict[str, Any] = {}
    
    def fuse_scores(
        self,
        camera_score: float,
        face_presence: bool,
        screen_result: ScreenMonitorResult
    ) -> FusedScore:
        """
        Fuse camera and screen scores into a combined attention score.
        
        Args:
            camera_score: Attention score from camera (0-100)
            face_presence: Whether face is detected in camera
            screen_result: Result from screen monitoring
        
        Returns:
            FusedScore object with combined score and metadata
        """
        timestamp = time.time()
        screen_score = screen_result.screen_score
        screen_presence = screen_result.screen_presence
        
        # Determine fusion mode and calculate combined score
        if face_presence and screen_presence:
            # Full mode: Both camera and screen have data
            fusion_mode = "full"
            combined_score = (
                camera_score * self.camera_weight +
                screen_score * self.screen_weight
            )
        elif face_presence and not screen_presence:
            # Camera only mode: KHÔNG áp dụng penalty nữa
            # Logic mới: Có khuôn mặt = Tập trung, không cần penalty
            fusion_mode = "camera_only"
            combined_score = camera_score  # KHÔNG nhân với penalty
        elif not face_presence and screen_presence:
            # Screen only mode: Fixed low score
            fusion_mode = "screen_only"
            combined_score = self.screen_only_score
        else:
            # None mode: No data from either source
            fusion_mode = "none"
            combined_score = 0.0
        
        # Ensure score is within valid range
        combined_score = max(0.0, min(100.0, combined_score))
        
        # Store fusion details for debugging
        self._last_fusion_details = {
            'fusion_mode': fusion_mode,
            'camera_score': camera_score,
            'screen_score': screen_score,
            'face_presence': face_presence,
            'screen_presence': screen_presence,
            'combined_score': combined_score,
            'camera_weight': self.camera_weight,
            'screen_weight': self.screen_weight,
            'timestamp': timestamp
        }
        
        return FusedScore(
            combined_score=combined_score,
            camera_score=camera_score,
            screen_score=screen_score,
            face_presence=face_presence,
            screen_presence=screen_presence,
            fusion_mode=fusion_mode,
            timestamp=timestamp
        )
    
    def get_fusion_details(self) -> Dict[str, Any]:
        """
        Get detailed information about the last fusion operation.
        
        Useful for debugging and understanding how scores were combined.
        
        Returns:
            Dictionary containing fusion details including:
            - fusion_mode: The mode used for fusion
            - camera_score: Raw camera score
            - screen_score: Raw screen score
            - face_presence: Camera face detection status
            - screen_presence: Screen face detection status
            - combined_score: Final fused score
            - camera_weight: Weight applied to camera score
            - screen_weight: Weight applied to screen score
            - timestamp: When fusion was calculated
        """
        return self._last_fusion_details.copy()
