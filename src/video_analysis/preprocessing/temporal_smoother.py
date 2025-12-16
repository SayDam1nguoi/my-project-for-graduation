"""
Temporal Smoother Module

This module provides temporal smoothing functionality for emotion predictions
to reduce jitter and create stable, natural-looking results.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class TemporalSmoother:
    """
    Smooths emotion predictions across time.
    
    Supports multiple smoothing methods:
    - EMA (Exponential Moving Average)
    - Sliding Window Average
    - Hybrid approach
    """
    
    def __init__(
        self,
        method: str = "ema",
        window_size: int = 7,
        ema_alpha: float = 0.3,
        min_confidence_for_smoothing: float = 0.6
    ):
        """
        Initialize TemporalSmoother.
        
        Args:
            method: Smoothing method ('ema', 'sliding_window', or 'hybrid')
            window_size: Window size for sliding window method
            ema_alpha: Alpha parameter for EMA (0.0-1.0, higher = less smoothing)
            min_confidence_for_smoothing: Minimum confidence to apply smoothing
        """
        self.method = method
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.min_confidence_for_smoothing = min_confidence_for_smoothing
        
        # History buffers
        self.probability_history: deque = deque(maxlen=window_size)
        self.label_history: deque = deque(maxlen=window_size)
        self.confidence_history: deque = deque(maxlen=window_size)
        
        # EMA state
        self.ema_probabilities: Optional[np.ndarray] = None
        self.ema_confidence: Optional[float] = None
        
        logger.info(f"TemporalSmoother initialized: method={method}, "
                   f"window_size={window_size}, ema_alpha={ema_alpha}")
    
    def smooth_prediction(
        self,
        probabilities: np.ndarray,
        label: str,
        confidence: float
    ) -> Tuple[np.ndarray, str, float]:
        """
        Apply temporal smoothing to emotion prediction.
        
        Args:
            probabilities: Emotion class probabilities
            label: Predicted emotion label
            confidence: Prediction confidence
            
        Returns:
            Tuple of (smoothed_probabilities, smoothed_label, smoothed_confidence)
        """
        # Add to history
        self.probability_history.append(probabilities.copy())
        self.label_history.append(label)
        self.confidence_history.append(confidence)
        
        # Check if we should apply smoothing
        if confidence < self.min_confidence_for_smoothing:
            # Low confidence, don't smooth (might be a transition)
            return probabilities, label, confidence
        
        # Apply smoothing based on method
        if self.method == "ema":
            smoothed_probs, smoothed_conf = self._apply_ema(probabilities, confidence)
        elif self.method == "sliding_window":
            smoothed_probs, smoothed_conf = self._apply_sliding_window()
        elif self.method == "hybrid":
            smoothed_probs, smoothed_conf = self._apply_hybrid(probabilities, confidence)
        else:
            logger.warning(f"Unknown smoothing method: {self.method}")
            return probabilities, label, confidence
        
        # Get smoothed label
        smoothed_label = self._get_label_from_probabilities(smoothed_probs)
        
        return smoothed_probs, smoothed_label, smoothed_conf
    
    def _apply_ema(
        self,
        current_probs: np.ndarray,
        current_conf: float
    ) -> Tuple[np.ndarray, float]:
        """
        Apply Exponential Moving Average smoothing.
        
        Formula: smoothed[t] = alpha * current[t] + (1-alpha) * smoothed[t-1]
        
        Args:
            current_probs: Current probabilities
            current_conf: Current confidence
            
        Returns:
            Tuple of (smoothed_probabilities, smoothed_confidence)
        """
        if self.ema_probabilities is None:
            # First frame, initialize
            self.ema_probabilities = current_probs.copy()
            self.ema_confidence = current_conf
            return current_probs, current_conf
        
        # Apply EMA
        smoothed_probs = (
            self.ema_alpha * current_probs +
            (1 - self.ema_alpha) * self.ema_probabilities
        )
        
        smoothed_conf = (
            self.ema_alpha * current_conf +
            (1 - self.ema_alpha) * self.ema_confidence
        )
        
        # Update state
        self.ema_probabilities = smoothed_probs
        self.ema_confidence = smoothed_conf
        
        return smoothed_probs, smoothed_conf
    
    def _apply_sliding_window(self) -> Tuple[np.ndarray, float]:
        """
        Apply sliding window average smoothing.
        
        Returns:
            Tuple of (smoothed_probabilities, smoothed_confidence)
        """
        if len(self.probability_history) == 0:
            return np.zeros(7), 0.0
        
        # Calculate average over window
        probs_array = np.array(list(self.probability_history))
        smoothed_probs = np.mean(probs_array, axis=0)
        
        # Average confidence
        smoothed_conf = np.mean(list(self.confidence_history))
        
        return smoothed_probs, smoothed_conf
    
    def _apply_hybrid(
        self,
        current_probs: np.ndarray,
        current_conf: float
    ) -> Tuple[np.ndarray, float]:
        """
        Apply hybrid smoothing (EMA + sliding window).
        
        Uses EMA for recent frames and sliding window for overall trend.
        
        Args:
            current_probs: Current probabilities
            current_conf: Current confidence
            
        Returns:
            Tuple of (smoothed_probabilities, smoothed_confidence)
        """
        # Get EMA smoothing
        ema_probs, ema_conf = self._apply_ema(current_probs, current_conf)
        
        # Get sliding window smoothing
        if len(self.probability_history) >= 3:
            window_probs, window_conf = self._apply_sliding_window()
            
            # Blend: 70% EMA (responsive) + 30% window (stable)
            smoothed_probs = 0.7 * ema_probs + 0.3 * window_probs
            smoothed_conf = 0.7 * ema_conf + 0.3 * window_conf
        else:
            # Not enough history, use EMA only
            smoothed_probs = ema_probs
            smoothed_conf = ema_conf
        
        return smoothed_probs, smoothed_conf
    
    def _get_label_from_probabilities(
        self,
        probabilities: np.ndarray,
        emotion_labels: Optional[List[str]] = None
    ) -> str:
        """
        Get emotion label from probabilities.
        
        Args:
            probabilities: Emotion probabilities
            emotion_labels: Optional list of emotion labels
            
        Returns:
            Emotion label
        """
        if emotion_labels is None:
            emotion_labels = [
                "Happy", "Sad", "Angry", "Fear",
                "Surprise", "Disgust", "Neutral"
            ]
        
        max_idx = np.argmax(probabilities)
        return emotion_labels[max_idx]
    
    def detect_significant_change(
        self,
        current_probs: np.ndarray,
        threshold: float = 0.3
    ) -> bool:
        """
        Detect if there's a significant emotion change.
        
        Significant changes should bypass smoothing for responsiveness.
        
        Args:
            current_probs: Current probabilities
            threshold: Change threshold
            
        Returns:
            True if significant change detected
        """
        if self.ema_probabilities is None or len(self.probability_history) == 0:
            return False
        
        # Calculate difference from smoothed state
        diff = np.abs(current_probs - self.ema_probabilities)
        max_diff = np.max(diff)
        
        return max_diff > threshold
    
    def smooth_with_adaptive_alpha(
        self,
        current_probs: np.ndarray,
        current_conf: float
    ) -> Tuple[np.ndarray, float]:
        """
        Apply EMA with adaptive alpha based on confidence.
        
        Higher confidence = higher alpha (less smoothing, more responsive)
        Lower confidence = lower alpha (more smoothing, more stable)
        
        Args:
            current_probs: Current probabilities
            current_conf: Current confidence
            
        Returns:
            Tuple of (smoothed_probabilities, smoothed_confidence)
        """
        # Adapt alpha based on confidence
        # High confidence (0.9-1.0) -> alpha = 0.5 (responsive)
        # Low confidence (0.6-0.7) -> alpha = 0.2 (stable)
        adaptive_alpha = self.ema_alpha + (current_conf - 0.7) * 0.5
        adaptive_alpha = np.clip(adaptive_alpha, 0.1, 0.7)
        
        if self.ema_probabilities is None:
            self.ema_probabilities = current_probs.copy()
            self.ema_confidence = current_conf
            return current_probs, current_conf
        
        # Apply adaptive EMA
        smoothed_probs = (
            adaptive_alpha * current_probs +
            (1 - adaptive_alpha) * self.ema_probabilities
        )
        
        smoothed_conf = (
            adaptive_alpha * current_conf +
            (1 - adaptive_alpha) * self.ema_confidence
        )
        
        # Update state
        self.ema_probabilities = smoothed_probs
        self.ema_confidence = smoothed_conf
        
        return smoothed_probs, smoothed_conf
    
    def get_dominant_emotion(
        self,
        min_duration: int = 3
    ) -> Optional[str]:
        """
        Get dominant emotion over recent history.
        
        Args:
            min_duration: Minimum number of frames for dominance
            
        Returns:
            Dominant emotion label or None
        """
        if len(self.label_history) < min_duration:
            return None
        
        # Count occurrences
        label_counts: Dict[str, int] = {}
        for label in self.label_history:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Find most common
        dominant_label = max(label_counts, key=label_counts.get)
        dominant_count = label_counts[dominant_label]
        
        # Check if it's truly dominant
        if dominant_count >= min_duration:
            return dominant_label
        
        return None
    
    def reset(self):
        """Reset smoothing state."""
        self.probability_history.clear()
        self.label_history.clear()
        self.confidence_history.clear()
        self.ema_probabilities = None
        self.ema_confidence = None
        logger.debug("TemporalSmoother reset")
