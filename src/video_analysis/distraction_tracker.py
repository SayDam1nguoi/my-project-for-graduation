"""
DistractionTracker - Tracks distraction duration and manages alert states.

This module implements a state machine to track when users lose focus and
triggers alerts after sustained distraction periods.
"""

import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum


class DistractionStateEnum(Enum):
    """States for the distraction tracking state machine."""
    FOCUSED = "focused"
    DISTRACTED = "distracted"
    ALERT = "alert"
    COOLDOWN = "cooldown"


@dataclass
class DistractionState:
    """
    State information for distraction tracking.
    
    Attributes:
        is_distracted: Whether user is currently distracted
        distraction_duration: Current continuous distraction time in seconds
        should_alert: Whether an alert should be triggered
        total_distraction_time: Cumulative distraction time in seconds
        last_alert_time: Timestamp of last alert (None if no alert yet)
        current_state: Current state in the state machine
    """
    is_distracted: bool
    distraction_duration: float
    should_alert: bool
    total_distraction_time: float
    last_alert_time: Optional[float]
    current_state: DistractionStateEnum


class DistractionTracker:
    """
    Tracks user distraction over time and manages alert triggering.
    
    Implements a state machine:
    FOCUSED → DISTRACTED → ALERT → COOLDOWN → FOCUSED
    
    - FOCUSED: User is paying attention (score >= threshold)
    - DISTRACTED: User is distracted, duration counting up
    - ALERT: Distraction duration reached alert threshold (5s)
    - COOLDOWN: Alert triggered, waiting 10s before next alert
    """
    
    def __init__(
        self,
        distraction_threshold: float = 70.0,  # Thang 0-100 (70/100 = tập trung)
        alert_threshold: float = 5.0,
        alert_cooldown: float = 10.0
    ):
        """
        Initialize the DistractionTracker.
        
        Args:
            distraction_threshold: Attention score below this is considered distracted (thang 0-100)
            alert_threshold: Seconds of distraction before triggering alert
            alert_cooldown: Seconds to wait between alerts
        """
        self.distraction_threshold = distraction_threshold
        self.alert_threshold = alert_threshold
        self.alert_cooldown = alert_cooldown
        
        # State tracking
        self._current_state = DistractionStateEnum.FOCUSED
        self._distraction_start_time: Optional[float] = None
        self._distraction_duration: float = 0.0
        self._last_alert_time: Optional[float] = None
        self._cooldown_start_time: Optional[float] = None
        
        # Cumulative statistics
        self._total_distraction_time: float = 0.0
        self._alert_count: int = 0  # Track number of alerts triggered
        self._last_update_time: Optional[float] = None
        
        # Thread safety
        self._lock = threading.Lock()
    
    def update(self, attention_score: float) -> DistractionState:
        """
        Update distraction state based on current attention score.
        
        This method is thread-safe and can be called from multiple threads.
        
        Args:
            attention_score: Current attention score (0-100)
            
        Returns:
            DistractionState with current state information
        """
        with self._lock:
            current_time = time.time()
            
            # Initialize last update time on first call
            if self._last_update_time is None:
                self._last_update_time = current_time
                return self._get_current_state()
            
            # Calculate time delta for accurate timing (±0.05s accuracy)
            time_delta = current_time - self._last_update_time
            self._last_update_time = current_time
            
            # Determine if currently distracted based on score
            is_currently_distracted = attention_score < self.distraction_threshold
            
            # State machine logic
            if self._current_state == DistractionStateEnum.FOCUSED:
                self._handle_focused_state(is_currently_distracted, current_time)
                
            elif self._current_state == DistractionStateEnum.DISTRACTED:
                self._handle_distracted_state(
                    is_currently_distracted, 
                    current_time, 
                    time_delta
                )
                
            elif self._current_state == DistractionStateEnum.ALERT:
                self._handle_alert_state(is_currently_distracted, current_time)
                
            elif self._current_state == DistractionStateEnum.COOLDOWN:
                self._handle_cooldown_state(
                    is_currently_distracted, 
                    current_time, 
                    time_delta
                )
            
            return self._get_current_state()
    
    def _handle_focused_state(
        self, 
        is_currently_distracted: bool, 
        current_time: float
    ) -> None:
        """Handle state transitions from FOCUSED state."""
        if is_currently_distracted:
            # Transition to DISTRACTED
            self._current_state = DistractionStateEnum.DISTRACTED
            self._distraction_start_time = current_time
            self._distraction_duration = 0.0
    
    def _handle_distracted_state(
        self, 
        is_currently_distracted: bool, 
        current_time: float,
        time_delta: float
    ) -> None:
        """Handle state transitions from DISTRACTED state."""
        if not is_currently_distracted:
            # User refocused - transition back to FOCUSED
            # Add accumulated distraction time to total
            self._total_distraction_time += self._distraction_duration
            self._current_state = DistractionStateEnum.FOCUSED
            self._distraction_duration = 0.0
            self._distraction_start_time = None
        else:
            # Still distracted - update duration
            self._distraction_duration += time_delta
            
            # Check if we should trigger alert
            if self._distraction_duration >= self.alert_threshold:
                # Check cooldown before triggering alert
                if self._last_alert_time is None or \
                   (current_time - self._last_alert_time) >= self.alert_cooldown:
                    # Transition to ALERT
                    self._current_state = DistractionStateEnum.ALERT
                    self._last_alert_time = current_time
                    self._alert_count += 1  # Increment alert counter
    
    def _handle_alert_state(
        self, 
        is_currently_distracted: bool, 
        current_time: float
    ) -> None:
        """Handle state transitions from ALERT state."""
        if not is_currently_distracted:
            # User refocused - add time to total and go to COOLDOWN
            self._total_distraction_time += self._distraction_duration
            self._current_state = DistractionStateEnum.COOLDOWN
            self._cooldown_start_time = current_time
            self._distraction_duration = 0.0
            self._distraction_start_time = None
        else:
            # Still distracted after alert - go to COOLDOWN but keep tracking duration
            # Don't reset duration so we can trigger alert again after cooldown
            self._current_state = DistractionStateEnum.COOLDOWN
            self._cooldown_start_time = current_time
            # Keep _distraction_duration and _distraction_start_time for continuous tracking
    
    def _handle_cooldown_state(
        self, 
        is_currently_distracted: bool, 
        current_time: float,
        time_delta: float
    ) -> None:
        """Handle state transitions from COOLDOWN state."""
        # Check if cooldown period has elapsed
        cooldown_elapsed = (current_time - self._cooldown_start_time) >= self.alert_cooldown
        
        if not is_currently_distracted:
            # User is focused - transition to FOCUSED immediately
            # No need to wait for cooldown to complete
            self._current_state = DistractionStateEnum.FOCUSED
            self._cooldown_start_time = None
            self._distraction_duration = 0.0
        else:
            # User is still/again distracted
            # Continue accumulating distraction time during cooldown
            self._distraction_duration += time_delta
            self._total_distraction_time += time_delta
            
            if cooldown_elapsed:
                # Cooldown complete
                self._cooldown_start_time = None
                
                # Check if duration already exceeds threshold (continuous distraction)
                if self._distraction_duration >= self.alert_threshold:
                    # Trigger alert immediately
                    self._current_state = DistractionStateEnum.ALERT
                    self._last_alert_time = current_time
                    self._alert_count += 1
                    print(f"[TRACKER] Alert triggered after cooldown! Duration: {self._distraction_duration:.1f}s")
                else:
                    # Transition to DISTRACTED to continue counting
                    self._current_state = DistractionStateEnum.DISTRACTED
                    if self._distraction_start_time is None:
                        self._distraction_start_time = current_time
    
    def _get_current_state(self) -> DistractionState:
        """
        Get current distraction state.
        
        Returns:
            DistractionState with current information
        """
        # Determine if should show alert (only in ALERT state)
        should_alert = self._current_state == DistractionStateEnum.ALERT
        
        # Determine if currently distracted (DISTRACTED, ALERT, or COOLDOWN with distraction)
        is_distracted = self._current_state in [
            DistractionStateEnum.DISTRACTED,
            DistractionStateEnum.ALERT,
            DistractionStateEnum.COOLDOWN
        ]
        
        return DistractionState(
            is_distracted=is_distracted,
            distraction_duration=self._distraction_duration,
            should_alert=should_alert,
            total_distraction_time=self._total_distraction_time,
            last_alert_time=self._last_alert_time,
            current_state=self._current_state
        )
    
    def reset(self) -> None:
        """
        Reset all distraction tracking state and statistics.
        
        This method is thread-safe.
        """
        with self._lock:
            self._current_state = DistractionStateEnum.FOCUSED
            self._distraction_start_time = None
            self._distraction_duration = 0.0
            self._last_alert_time = None
            self._cooldown_start_time = None
            self._total_distraction_time = 0.0
            self._alert_count = 0  # Reset alert counter
            self._last_update_time = None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current statistics for distraction tracking.
        
        Returns:
            Dictionary with statistics including:
                - current_state: Current state name
                - is_distracted: Whether currently distracted
                - distraction_duration: Current distraction duration
                - total_distraction_time: Cumulative distraction time
                - alert_count: Number of alerts triggered
        """
        with self._lock:
            state = self._get_current_state()
            
            return {
                'current_state': state.current_state.value,
                'is_distracted': state.is_distracted,
                'distraction_duration': round(state.distraction_duration, 2),
                'total_distraction_time': round(state.total_distraction_time, 2),
                'alert_count': self._alert_count,  # Use the actual counter
                'last_alert_time': state.last_alert_time
            }
