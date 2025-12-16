"""
Score Calculator for Dual Person Comparison System.

This module calculates emotion and attention scores from historical data
using a rolling window approach. Scores are normalized to 0-100 range.
"""

import time
from typing import List, Dict
from src.video_analysis.dual_person_models import EmotionData


class ScoreCalculator:
    """Calculate emotion and attention scores from historical data.
    
    Uses a rolling window approach to calculate scores based on recent data.
    Emotion scores are weighted based on emotion type (positive vs negative).
    Attention scores are simple averages of attention values.
    
    Attributes:
        window_seconds: Size of rolling window in seconds (default 60)
        update_interval: Minimum time between score updates in seconds (default 5)
        emotion_weights: Mapping of emotion labels to weight values (0-1)
    """
    
    # Emotion weights (positive emotions = higher score)
    EMOTION_WEIGHTS = {
        'happy': 1.0,
        'surprise': 0.8,
        'neutral': 0.5,
        'sad': 0.2,
        'fear': 0.1,
        'disgust': 0.1,
        'angry': 0.0
    }
    
    def __init__(
        self,
        window_seconds: int = 60,
        update_interval: float = 5.0,
        emotion_weights: Dict[str, float] = None
    ):
        """Initialize ScoreCalculator.
        
        Args:
            window_seconds: Size of rolling window in seconds
            update_interval: Minimum time between score updates in seconds
            emotion_weights: Custom emotion weights (optional)
        """
        self.window_seconds = window_seconds
        self.update_interval = update_interval
        
        # Use custom weights if provided, otherwise use defaults
        if emotion_weights is not None:
            self.emotion_weights = emotion_weights
        else:
            self.emotion_weights = self.EMOTION_WEIGHTS.copy()
        
        # Track last update time for interval logic
        self.last_emotion_update = 0.0
        self.last_attention_update = 0.0
        
        # Cache for scores
        self._cached_emotion_score = 0.0
        self._cached_attention_score = 0.0
    
    def _filter_by_window(
        self,
        data_points: List,
        current_time: float = None
    ) -> List:
        """Filter data points to only include those within the rolling window.
        
        Args:
            data_points: List of data points with timestamp attribute
            current_time: Current timestamp (uses time.time() if None)
        
        Returns:
            List of data points within the window
        """
        if current_time is None:
            current_time = time.time()
        
        window_start = current_time - self.window_seconds
        
        return [
            point for point in data_points
            if point.timestamp >= window_start
        ]
    
    def should_update_emotion_score(self) -> bool:
        """Check if enough time has passed to update emotion score.
        
        Returns:
            True if score should be updated based on update_interval
        """
        current_time = time.time()
        return (current_time - self.last_emotion_update) >= self.update_interval
    
    def should_update_attention_score(self) -> bool:
        """Check if enough time has passed to update attention score.
        
        Returns:
            True if score should be updated based on update_interval
        """
        current_time = time.time()
        return (current_time - self.last_attention_update) >= self.update_interval
    
    def calculate_emotion_score(
        self,
        emotion_history: List[EmotionData],
        force_update: bool = False
    ) -> float:
        """Calculate emotion score (0-100) from emotion history.
        
        Algorithm:
        1. Filter emotion history to rolling window (last 60 seconds)
        2. For each emotion detection in window:
           - Get emotion label and confidence
           - Apply emotion weight based on emotion type
           - Weighted score = confidence * emotion_weight
        3. Average all weighted scores
        4. Scale to 0-100 range
        
        Args:
            emotion_history: List of EmotionData objects
            force_update: If True, bypass update interval check
        
        Returns:
            Emotion score in range 0-100
        """
        # Check if we should update based on interval
        if not force_update and not self.should_update_emotion_score():
            return self._cached_emotion_score
        
        # Filter to rolling window
        current_time = time.time()
        windowed_data = self._filter_by_window(emotion_history, current_time)
        
        # Handle empty history
        if not windowed_data:
            self._cached_emotion_score = 0.0
            self.last_emotion_update = current_time
            return 0.0
        
        # Calculate weighted scores
        weighted_scores = []
        for emotion_data in windowed_data:
            emotion = emotion_data.emotion.lower()
            confidence = emotion_data.confidence
            
            # Get weight for this emotion (default to 0.5 if unknown)
            weight = self.emotion_weights.get(emotion, 0.5)
            
            # Calculate weighted score
            weighted_score = confidence * weight
            weighted_scores.append(weighted_score)
        
        # Average weighted scores
        avg_weighted_score = sum(weighted_scores) / len(weighted_scores)
        
        # Scale to 0-100 range
        emotion_score = avg_weighted_score * 100.0
        
        # Update cache and timestamp
        self._cached_emotion_score = emotion_score
        self.last_emotion_update = current_time
        
        return emotion_score
    
    def calculate_attention_score(
        self,
        attention_history: List[float],
        timestamps: List[float] = None,
        force_update: bool = False
    ) -> float:
        """Calculate attention score (0-100) from attention history.
        
        Algorithm:
        1. Filter attention history to rolling window (last 60 seconds)
        2. Average attention scores in window
        3. Scores are already in 0-100 range from AttentionDetector
        
        Args:
            attention_history: List of attention scores (0-100)
            timestamps: List of timestamps corresponding to attention scores
            force_update: If True, bypass update interval check
        
        Returns:
            Attention score in range 0-100
        """
        # Check if we should update based on interval
        if not force_update and not self.should_update_attention_score():
            return self._cached_attention_score
        
        current_time = time.time()
        
        # Handle empty history
        if not attention_history:
            self._cached_attention_score = 0.0
            self.last_attention_update = current_time
            return 0.0
        
        # Filter to rolling window if timestamps provided
        if timestamps is not None and len(timestamps) == len(attention_history):
            window_start = current_time - self.window_seconds
            windowed_scores = [
                score for score, ts in zip(attention_history, timestamps)
                if ts >= window_start
            ]
        else:
            # If no timestamps, use most recent data up to window size
            # Assume data is ordered chronologically
            max_points = int(self.window_seconds)  # Rough estimate
            windowed_scores = attention_history[-max_points:]
        
        # Handle empty windowed data
        if not windowed_scores:
            self._cached_attention_score = 0.0
            self.last_attention_update = current_time
            return 0.0
        
        # Calculate average attention score
        attention_score = sum(windowed_scores) / len(windowed_scores)
        
        # Ensure score is in valid range
        attention_score = max(0.0, min(100.0, attention_score))
        
        # Update cache and timestamp
        self._cached_attention_score = attention_score
        self.last_attention_update = current_time
        
        return attention_score
    
    def reset(self):
        """Reset calculator state.
        
        Clears cached scores and update timestamps. Useful when starting
        a new session or switching modes.
        """
        self.last_emotion_update = 0.0
        self.last_attention_update = 0.0
        self._cached_emotion_score = 0.0
        self._cached_attention_score = 0.0
    
    def get_cached_scores(self) -> Dict[str, float]:
        """Get currently cached scores without recalculation.
        
        Returns:
            Dictionary with 'emotion_score' and 'attention_score' keys
        """
        return {
            'emotion_score': self._cached_emotion_score,
            'attention_score': self._cached_attention_score
        }
    
    def set_emotion_weights(self, weights: Dict[str, float]):
        """Update emotion weights.
        
        Args:
            weights: Dictionary mapping emotion labels to weight values (0-1)
        """
        self.emotion_weights.update(weights)
    
    def get_emotion_weights(self) -> Dict[str, float]:
        """Get current emotion weights.
        
        Returns:
            Dictionary of emotion weights
        """
        return self.emotion_weights.copy()
