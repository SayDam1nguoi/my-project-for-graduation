"""
Alert Manager for Dual Attention Detection System

This module manages and displays distraction alerts with fade animations.
"""

import time
import cv2
import numpy as np
from typing import Tuple, Optional


class AlertManager:
    """
    Manages distraction alerts with visual overlays and animations.
    
    Responsibilities:
    - Trigger and dismiss alerts
    - Draw alert banner on video frames
    - Handle fade in/out animations
    - Auto-dismiss after user refocuses
    """
    
    def __init__(
        self,
        alert_duration: float = 0.0,
        banner_size: Tuple[int, int] = (400, 80),
        fade_in_duration: float = 0.3,
        fade_out_duration: float = 0.3
    ):
        """
        Initialize AlertManager.
        
        Args:
            alert_duration: Time to wait before auto-dismiss after refocus (seconds)
            banner_size: (width, height) of alert banner in pixels
            fade_in_duration: Duration of fade-in animation (seconds)
            fade_out_duration: Duration of fade-out animation (seconds)
        """
        self.alert_duration = alert_duration
        self.banner_width, self.banner_height = banner_size
        self.fade_in_duration = fade_in_duration
        self.fade_out_duration = fade_out_duration
        
        # Alert state
        self._alert_active = False
        self._alert_triggered_time: Optional[float] = None
        self._alert_dismissed_time: Optional[float] = None
        self._refocus_time: Optional[float] = None
        
        # Visual properties
        self.background_color = (0, 0, 255)  # BGR red
        self.text_color = (255, 255, 255)  # BGR white
        self.border_color = (255, 255, 255)  # BGR white
        self.border_thickness = 2
        
    def trigger_alert(self, distraction_duration: float) -> None:
        """
        Trigger a distraction alert.
        
        Args:
            distraction_duration: Current distraction duration in seconds
        """
        if not self._alert_active:
            self._alert_active = True
            self._alert_triggered_time = time.time()
            self._alert_dismissed_time = None
            self._refocus_time = None
    
    def dismiss_alert(self) -> None:
        """
        Dismiss the current alert (called when user refocuses).
        
        The alert will fade out and be removed after alert_duration seconds.
        """
        if self._alert_active and self._refocus_time is None:
            self._refocus_time = time.time()
    
    def is_alert_active(self) -> bool:
        """
        Check if alert is currently active (visible or fading).
        
        Returns:
            True if alert should be displayed, False otherwise
        """
        if not self._alert_active:
            return False
        
        # Check if auto-dismiss timer has expired
        if self._refocus_time is not None:
            elapsed = time.time() - self._refocus_time
            if elapsed >= self.alert_duration + self.fade_out_duration:
                self._alert_active = False
                return False
        
        return True
    
    def _get_alpha(self) -> float:
        """
        Calculate current alpha (opacity) based on animation state.
        
        Returns:
            Alpha value between 0.0 (transparent) and 0.9 (opaque)
        """
        if not self._alert_active or self._alert_triggered_time is None:
            return 0.0
        
        current_time = time.time()
        
        # Fade in animation
        if self._refocus_time is None:
            elapsed = current_time - self._alert_triggered_time
            if elapsed < self.fade_in_duration:
                # Linear fade in from 0 to 0.9
                return 0.9 * (elapsed / self.fade_in_duration)
            else:
                # Fully visible
                return 0.9
        
        # Fade out animation (after refocus)
        else:
            elapsed_since_refocus = current_time - self._refocus_time
            
            # Wait for alert_duration before starting fade out
            if elapsed_since_refocus < self.alert_duration:
                return 0.9
            
            # Fade out
            fade_elapsed = elapsed_since_refocus - self.alert_duration
            if fade_elapsed < self.fade_out_duration:
                # Linear fade out from 0.9 to 0
                return 0.9 * (1.0 - fade_elapsed / self.fade_out_duration)
            else:
                return 0.0
    
    def draw_alert(
        self,
        frame: np.ndarray,
        distraction_duration: float
    ) -> np.ndarray:
        """
        Draw alert banner on the frame with current animation state.
        
        Args:
            frame: Input video frame (BGR format)
            distraction_duration: Current distraction duration in seconds
            
        Returns:
            Frame with alert banner overlay
        """
        if not self.is_alert_active():
            return frame
        
        # Get current alpha for animation
        alpha = self._get_alpha()
        if alpha <= 0.0:
            return frame
        
        # Create a copy to avoid modifying original
        output_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate banner position (centered)
        banner_x = (frame_width - self.banner_width) // 2
        banner_y = (frame_height - self.banner_height) // 2
        
        # Ensure banner fits within frame
        if banner_x < 0 or banner_y < 0:
            # Frame too small, adjust banner size
            banner_x = max(10, banner_x)
            banner_y = max(10, banner_y)
            actual_width = min(self.banner_width, frame_width - 20)
            actual_height = min(self.banner_height, frame_height - 20)
        else:
            actual_width = self.banner_width
            actual_height = self.banner_height
        
        # Create banner overlay
        overlay = output_frame.copy()
        
        # Draw filled rectangle (background)
        cv2.rectangle(
            overlay,
            (banner_x, banner_y),
            (banner_x + actual_width, banner_y + actual_height),
            self.background_color,
            -1  # Filled
        )
        
        # Draw border
        cv2.rectangle(
            overlay,
            (banner_x, banner_y),
            (banner_x + actual_width, banner_y + actual_height),
            self.border_color,
            self.border_thickness
        )
        
        # Blend overlay with original frame using alpha
        cv2.addWeighted(overlay, alpha, output_frame, 1 - alpha, 0, output_frame)
        
        # Add text with full opacity (text is always visible when banner is visible)
        # Main alert message (Vietnamese without diacritics to avoid rendering issues)
        alert_text = "CANH BAO: MAT TAP TRUNG!"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        # Get text size for centering
        (text_width, text_height), baseline = cv2.getTextSize(
            alert_text, font, font_scale, font_thickness
        )
        
        # Ensure text fits within banner (with padding)
        max_text_width = actual_width - 20  # 10px padding on each side
        if text_width > max_text_width:
            # Scale down font to fit
            font_scale = font_scale * (max_text_width / text_width)
            (text_width, text_height), baseline = cv2.getTextSize(
                alert_text, font, font_scale, font_thickness
            )
        
        # Position main text (upper part of banner)
        text_x = banner_x + (actual_width - text_width) // 2
        text_y = banner_y + actual_height // 3 + text_height // 2
        
        # Draw main text with shadow for better visibility
        cv2.putText(
            output_frame,
            alert_text,
            (text_x + 2, text_y + 2),
            font,
            font_scale,
            (0, 0, 0),  # Black shadow
            font_thickness,
            cv2.LINE_AA
        )
        cv2.putText(
            output_frame,
            alert_text,
            (text_x, text_y),
            font,
            font_scale,
            self.text_color,
            font_thickness,
            cv2.LINE_AA
        )
        
        # Duration text (Vietnamese without diacritics)
        duration_text = f"Thoi gian: {distraction_duration:.1f}s"
        duration_font_scale = 0.5
        duration_thickness = 1
        
        (duration_width, duration_height), _ = cv2.getTextSize(
            duration_text, font, duration_font_scale, duration_thickness
        )
        
        # Ensure duration text fits within banner
        if duration_width > max_text_width:
            duration_font_scale = duration_font_scale * (max_text_width / duration_width)
            (duration_width, duration_height), _ = cv2.getTextSize(
                duration_text, font, duration_font_scale, duration_thickness
            )
        
        # Position duration text (lower part of banner)
        duration_x = banner_x + (actual_width - duration_width) // 2
        duration_y = banner_y + 2 * actual_height // 3 + duration_height
        
        # Draw duration text with shadow
        cv2.putText(
            output_frame,
            duration_text,
            (duration_x + 1, duration_y + 1),
            font,
            duration_font_scale,
            (0, 0, 0),  # Black shadow
            duration_thickness,
            cv2.LINE_AA
        )
        cv2.putText(
            output_frame,
            duration_text,
            (duration_x, duration_y),
            font,
            duration_font_scale,
            self.text_color,
            duration_thickness,
            cv2.LINE_AA
        )
        
        return output_frame
    
    def reset(self) -> None:
        """Reset alert manager state."""
        self._alert_active = False
        self._alert_triggered_time = None
        self._alert_dismissed_time = None
        self._refocus_time = None
