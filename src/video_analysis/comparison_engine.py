"""
Comparison Engine for Dual Person Comparison System.

This module implements the ComparisonEngine class that compares emotion and
attention scores between two users, determines winners, calculates percentage
differences, and detects significant gaps.
"""

import time
from typing import Optional
from .dual_person_models import ComparisonResult


class ComparisonEngine:
    """Engine for comparing scores between two users.
    
    Compares emotion and attention scores between primary and secondary users,
    determines winners, calculates percentage differences, and identifies
    significant gaps (>20 points).
    
    Attributes:
        SIGNIFICANT_DIFFERENCE_THRESHOLD: Threshold for significant gap (20 points)
        TIE_THRESHOLD: Threshold for considering scores as tie (1 point)
        update_interval: Interval in seconds between comparison updates
        last_comparison_time: Timestamp of last comparison
        last_comparison: Last ComparisonResult generated
    """
    
    SIGNIFICANT_DIFFERENCE_THRESHOLD = 20.0  # points
    TIE_THRESHOLD = 1.0  # points - scores within 1 point are considered tie
    
    def __init__(self, update_interval: float = 10.0):
        """Initialize comparison engine.
        
        Args:
            update_interval: Interval in seconds between comparison updates (default: 10s)
        """
        self.update_interval = update_interval
        self.last_comparison_time = 0.0
        self.last_comparison: Optional[ComparisonResult] = None
    
    def should_update(self) -> bool:
        """Check if enough time has passed for a comparison update.
        
        Returns:
            True if update interval has elapsed, False otherwise
        """
        current_time = time.time()
        return (current_time - self.last_comparison_time) >= self.update_interval
    
    def compare(
        self,
        primary_emotion_score: float,
        primary_attention_score: float,
        secondary_emotion_score: float,
        secondary_attention_score: float,
        force_update: bool = False
    ) -> Optional[ComparisonResult]:
        """Compare scores between two users.
        
        Compares emotion and attention scores, determines winners, calculates
        percentage differences, and detects significant gaps.
        
        Args:
            primary_emotion_score: Primary user's emotion score (0-100)
            primary_attention_score: Primary user's attention score (0-100)
            secondary_emotion_score: Secondary user's emotion score (0-100)
            secondary_attention_score: Secondary user's attention score (0-100)
            force_update: Force comparison even if update interval hasn't elapsed
        
        Returns:
            ComparisonResult with comparison metrics, or None if update interval
            hasn't elapsed and force_update is False
        """
        # Check if we should update
        if not force_update and not self.should_update():
            return self.last_comparison
        
        current_time = time.time()
        
        # Compare emotion scores
        emotion_winner = self._determine_winner(
            primary_emotion_score,
            secondary_emotion_score
        )
        emotion_difference = self._calculate_percentage_difference(
            primary_emotion_score,
            secondary_emotion_score
        )
        significant_emotion_gap = self._is_significant_gap(
            primary_emotion_score,
            secondary_emotion_score
        )
        
        # Compare attention scores
        attention_winner = self._determine_winner(
            primary_attention_score,
            secondary_attention_score
        )
        attention_difference = self._calculate_percentage_difference(
            primary_attention_score,
            secondary_attention_score
        )
        significant_attention_gap = self._is_significant_gap(
            primary_attention_score,
            secondary_attention_score
        )
        
        # Create comparison result
        comparison = ComparisonResult(
            timestamp=current_time,
            primary_emotion_score=primary_emotion_score,
            secondary_emotion_score=secondary_emotion_score,
            emotion_winner=emotion_winner,
            emotion_difference=emotion_difference,
            significant_emotion_gap=significant_emotion_gap,
            primary_attention_score=primary_attention_score,
            secondary_attention_score=secondary_attention_score,
            attention_winner=attention_winner,
            attention_difference=attention_difference,
            significant_attention_gap=significant_attention_gap
        )
        
        # Update state
        self.last_comparison_time = current_time
        self.last_comparison = comparison
        
        return comparison
    
    def _determine_winner(self, primary_score: float, secondary_score: float) -> str:
        """Determine winner based on scores.
        
        Args:
            primary_score: Primary user's score (0-100)
            secondary_score: Secondary user's score (0-100)
        
        Returns:
            "primary" if primary has higher score,
            "secondary" if secondary has higher score,
            "tie" if scores are within TIE_THRESHOLD
        """
        difference = abs(primary_score - secondary_score)
        
        if difference <= self.TIE_THRESHOLD:
            return "tie"
        elif primary_score > secondary_score:
            return "primary"
        else:
            return "secondary"
    
    def _calculate_percentage_difference(
        self,
        primary_score: float,
        secondary_score: float
    ) -> float:
        """Calculate percentage difference between scores.
        
        Calculates the percentage difference relative to the higher score.
        For example, if primary=80 and secondary=60, the difference is 25%
        (20 points difference / 80 higher score * 100).
        
        Args:
            primary_score: Primary user's score (0-100)
            secondary_score: Secondary user's score (0-100)
        
        Returns:
            Percentage difference (0-100)
        """
        # Handle edge case where both scores are 0
        if primary_score == 0 and secondary_score == 0:
            return 0.0
        
        # Calculate absolute difference
        difference = abs(primary_score - secondary_score)
        
        # Calculate percentage relative to higher score
        max_score = max(primary_score, secondary_score)
        
        # Handle edge case where max_score is 0 (shouldn't happen but be safe)
        if max_score == 0:
            return 0.0
        
        percentage = (difference / max_score) * 100.0
        
        # Round to 2 decimal places
        return round(percentage, 2)
    
    def _is_significant_gap(
        self,
        primary_score: float,
        secondary_score: float
    ) -> bool:
        """Check if there's a significant gap between scores.
        
        A gap is considered significant if the absolute difference is greater
        than SIGNIFICANT_DIFFERENCE_THRESHOLD (20 points).
        
        Args:
            primary_score: Primary user's score (0-100)
            secondary_score: Secondary user's score (0-100)
        
        Returns:
            True if gap is significant (>20 points), False otherwise
        """
        difference = abs(primary_score - secondary_score)
        return difference > self.SIGNIFICANT_DIFFERENCE_THRESHOLD
    
    def get_last_comparison(self) -> Optional[ComparisonResult]:
        """Get the last comparison result.
        
        Returns:
            Last ComparisonResult, or None if no comparison has been made yet
        """
        return self.last_comparison
    
    def reset(self):
        """Reset comparison engine state.
        
        Clears last comparison time and result. Useful when starting a new session.
        """
        self.last_comparison_time = 0.0
        self.last_comparison = None
    
    def set_update_interval(self, interval: float):
        """Set the comparison update interval.
        
        Args:
            interval: New update interval in seconds
        
        Raises:
            ValueError: If interval is not positive
        """
        if interval <= 0:
            raise ValueError("Update interval must be positive")
        self.update_interval = interval
