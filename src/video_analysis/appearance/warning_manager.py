# -*- coding: utf-8 -*-
"""
Warning Manager for Appearance Assessment

Manages real-time warnings for lighting and clothing issues.
Tracks warning state, duration, and determines when to display warnings.
"""

import time
from typing import Dict, List, Optional
from .models import AppearanceWarning, AppearanceAssessment


class WarningManager:
    """
    Manages appearance warnings with threshold and duration logic.
    
    Requirements:
    - 7.1: Analyze appearance every 5 seconds during video capture
    - 7.2: Display warning when lighting score < 60
    - 7.3: Display warning when clothing score < 60
    - 7.4: Remove warnings within 10 seconds when resolved
    """
    
    # Warning thresholds
    DEFAULT_WARNING_THRESHOLD = 60.0
    WARNING_REMOVAL_DELAY = 10.0  # seconds
    
    # Severity thresholds
    HIGH_SEVERITY_THRESHOLD = 40.0  # score < 40 = high severity
    MEDIUM_SEVERITY_THRESHOLD = 50.0  # score < 50 = medium severity
    # score >= 50 = low severity
    
    def __init__(self, warning_threshold: float = DEFAULT_WARNING_THRESHOLD):
        """
        Initialize WarningManager.
        
        Args:
            warning_threshold: Score threshold below which warnings are triggered
        """
        self.warning_threshold = warning_threshold
        self.active_warnings: Dict[str, AppearanceWarning] = {}
        self.warning_start_times: Dict[str, float] = {}
        self.warning_resolved_times: Dict[str, float] = {}
    
    def update_warnings(self, assessment: AppearanceAssessment) -> List[AppearanceWarning]:
        """
        Update warnings based on current assessment.
        
        Args:
            assessment: Current appearance assessment
            
        Returns:
            List of warnings that should be displayed
        """
        current_time = time.time()
        warnings_to_display = []
        
        # Check lighting warnings
        if assessment.lighting is not None:
            lighting_warning = self._check_lighting_warning(
                assessment.lighting.score,
                assessment.lighting.issues,
                assessment.lighting.recommendations,
                current_time
            )
            if lighting_warning:
                warnings_to_display.append(lighting_warning)
        
        # Check clothing warnings
        if assessment.clothing is not None:
            clothing_warning = self._check_clothing_warning(
                assessment.clothing.score,
                assessment.clothing.issues,
                assessment.clothing.recommendations,
                current_time
            )
            if clothing_warning:
                warnings_to_display.append(clothing_warning)
        
        # Clear resolved warnings
        self.clear_resolved_warnings(assessment)
        
        return warnings_to_display
    
    def _check_lighting_warning(
        self,
        score: float,
        issues: List[str],
        recommendations: List[str],
        current_time: float
    ) -> Optional[AppearanceWarning]:
        """Check if lighting warning should be shown."""
        warning_type = "lighting"
        
        if self.should_show_warning(warning_type, score):
            # Track warning start time
            if warning_type not in self.warning_start_times:
                self.warning_start_times[warning_type] = current_time
            
            # Clear resolved time if warning is active again
            if warning_type in self.warning_resolved_times:
                del self.warning_resolved_times[warning_type]
            
            # Determine severity
            severity = self._calculate_severity(score)
            
            # Create warning message
            message = f"Lighting quality is below acceptable level (score: {score:.1f})"
            
            # Create warning
            warning = AppearanceWarning(
                warning_type=warning_type,
                severity=severity,
                message=message,
                recommendations=recommendations,
                timestamp=current_time,
                should_display=True
            )
            
            self.active_warnings[warning_type] = warning
            return warning
        else:
            # Score is above threshold - mark as resolved
            if warning_type in self.active_warnings:
                if warning_type not in self.warning_resolved_times:
                    self.warning_resolved_times[warning_type] = current_time
        
        return None
    
    def _check_clothing_warning(
        self,
        score: float,
        issues: List[str],
        recommendations: List[str],
        current_time: float
    ) -> Optional[AppearanceWarning]:
        """Check if clothing warning should be shown."""
        warning_type = "clothing"
        
        if self.should_show_warning(warning_type, score):
            # Track warning start time
            if warning_type not in self.warning_start_times:
                self.warning_start_times[warning_type] = current_time
            
            # Clear resolved time if warning is active again
            if warning_type in self.warning_resolved_times:
                del self.warning_resolved_times[warning_type]
            
            # Determine severity
            severity = self._calculate_severity(score)
            
            # Create warning message
            message = f"Clothing professionalism is below acceptable level (score: {score:.1f})"
            
            # Create warning
            warning = AppearanceWarning(
                warning_type=warning_type,
                severity=severity,
                message=message,
                recommendations=recommendations,
                timestamp=current_time,
                should_display=True
            )
            
            self.active_warnings[warning_type] = warning
            return warning
        else:
            # Score is above threshold - mark as resolved
            if warning_type in self.active_warnings:
                if warning_type not in self.warning_resolved_times:
                    self.warning_resolved_times[warning_type] = current_time
        
        return None
    
    def should_show_warning(self, warning_type: str, score: float) -> bool:
        """
        Determine if warning should be displayed.
        
        Args:
            warning_type: Type of warning ("lighting" or "clothing")
            score: Current score for the assessment
            
        Returns:
            True if warning should be shown, False otherwise
        """
        return score < self.warning_threshold
    
    def _calculate_severity(self, score: float) -> str:
        """
        Calculate warning severity based on score.
        
        Args:
            score: Assessment score (0-100)
            
        Returns:
            Severity level: "high", "medium", or "low"
        """
        if score < self.HIGH_SEVERITY_THRESHOLD:
            return "high"
        elif score < self.MEDIUM_SEVERITY_THRESHOLD:
            return "medium"
        else:
            return "low"
    
    def clear_resolved_warnings(self, assessment: AppearanceAssessment):
        """
        Remove warnings that have been resolved for sufficient time.
        
        A warning is considered resolved when:
        1. The score is above the threshold
        2. It has been above threshold for at least WARNING_REMOVAL_DELAY seconds
        
        Args:
            assessment: Current appearance assessment
        """
        current_time = time.time()
        warnings_to_remove = []
        
        for warning_type in list(self.warning_resolved_times.keys()):
            resolved_time = self.warning_resolved_times[warning_type]
            time_since_resolved = current_time - resolved_time
            
            # Remove warning if it's been resolved for long enough
            if time_since_resolved >= self.WARNING_REMOVAL_DELAY:
                warnings_to_remove.append(warning_type)
        
        # Clean up resolved warnings
        for warning_type in warnings_to_remove:
            if warning_type in self.active_warnings:
                del self.active_warnings[warning_type]
            if warning_type in self.warning_start_times:
                del self.warning_start_times[warning_type]
            if warning_type in self.warning_resolved_times:
                del self.warning_resolved_times[warning_type]
    
    def get_active_warnings(self) -> List[AppearanceWarning]:
        """
        Get list of currently active warnings.
        
        Returns:
            List of active warnings
        """
        return list(self.active_warnings.values())
    
    def reset(self):
        """Reset all warning state."""
        self.active_warnings.clear()
        self.warning_start_times.clear()
        self.warning_resolved_times.clear()
