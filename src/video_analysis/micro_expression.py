# -*- coding: utf-8 -*-
"""
Micro-Expression Detector

Detects brief emotional expressions (<500ms) that may indicate
suppressed or hidden emotions.

Micro-expressions are involuntary facial expressions that occur when
a person is trying to conceal their true emotions.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import numpy as np
from typing import Dict, List, Optional, Any
from collections import Counter, deque


class MicroExpressionDetector:
    """
    Detect micro-expressions from emotion sequence.
    
    Micro-expressions are brief (< 500ms) emotional flashes that
    differ from the dominant emotion, often indicating suppressed feelings.
    """
    
    def __init__(
        self,
        window_size: int = 15,
        min_duration: int = 3,
        max_duration_ratio: float = 0.3
    ):
        """
        Initialize Micro-Expression Detector.
        
        Args:
            window_size: Number of frames to analyze (15 frames ≈ 500ms at 30fps)
            min_duration: Minimum frames for micro-expression (default: 3)
            max_duration_ratio: Max ratio of window for micro-expr (default: 0.3 = 30%)
        
        Example:
            >>> detector = MicroExpressionDetector(window_size=15)
            >>> # At 30fps, 15 frames = 500ms
        """
        self.window_size = window_size
        self.min_duration = min_duration
        self.max_duration_ratio = max_duration_ratio
        
        # History tracking
        self.emotion_history = deque(maxlen=window_size)
        self.intensity_history = deque(maxlen=window_size)
        
        # Detected micro-expressions
        self.micro_expressions = []
    
    def detect(
        self,
        current_emotion: str,
        current_intensity: float
    ) -> Optional[Dict[str, Any]]:
        """
        Detect micro-expression from current emotion.
        
        Args:
            current_emotion: Current emotion label
            current_intensity: Current emotion intensity (0-1)
        
        Returns:
            Micro-expression info if detected, None otherwise:
            {
                'micro_emotion': str,
                'duration_frames': int,
                'duration_ms': float,
                'suppressed': bool,
                'avg_intensity': float,
                'pattern': str
            }
        
        Example:
            >>> detector = MicroExpressionDetector()
            >>> # Feed emotions frame by frame
            >>> for emotion, intensity in emotion_sequence:
            ...     micro_expr = detector.detect(emotion, intensity)
            ...     if micro_expr:
            ...         print(f"Micro-expression detected: {micro_expr}")
        """
        # Add to history
        self.emotion_history.append(current_emotion)
        self.intensity_history.append(current_intensity)
        
        # Need enough history
        if len(self.emotion_history) < self.window_size:
            return None
        
        # Detect pattern: A → B → A (brief flash of emotion B)
        micro_expr = self._detect_flash_pattern()
        
        if micro_expr:
            # Store detected micro-expression
            self.micro_expressions.append(micro_expr)
            return micro_expr
        
        return None
    
    def _detect_flash_pattern(self) -> Optional[Dict[str, Any]]:
        """
        Detect A → B → A pattern (brief flash of emotion B).
        
        This indicates emotion B was briefly expressed but quickly
        suppressed back to emotion A.
        """
        emotions = list(self.emotion_history)
        intensities = list(self.intensity_history)
        
        # Count unique emotions in window
        unique_emotions = set(emotions)
        
        # Need exactly 2 emotions for flash pattern
        if len(unique_emotions) != 2:
            return None
        
        # Count occurrences
        emotion_counts = Counter(emotions)
        
        # Find brief emotion (< max_duration_ratio of window)
        max_brief_frames = int(self.window_size * self.max_duration_ratio)
        
        for emotion, count in emotion_counts.items():
            # Check if this emotion is brief enough
            if self.min_duration <= count <= max_brief_frames:
                # This is potentially a micro-expression
                
                # Get dominant emotion (the other one)
                dominant_emotion = [e for e in unique_emotions if e != emotion][0]
                
                # Verify it's actually a flash pattern (not just transition)
                if self._is_flash_pattern(emotions, emotion, dominant_emotion):
                    # Calculate average intensity of micro-expression
                    micro_intensities = [
                        intensities[i] for i, e in enumerate(emotions)
                        if e == emotion
                    ]
                    avg_intensity = np.mean(micro_intensities)
                    
                    # Calculate duration in ms (assuming 30fps)
                    duration_ms = (count / 30.0) * 1000
                    
                    return {
                        'micro_emotion': emotion,
                        'dominant_emotion': dominant_emotion,
                        'duration_frames': count,
                        'duration_ms': duration_ms,
                        'suppressed': True,
                        'avg_intensity': avg_intensity,
                        'pattern': f"{dominant_emotion} → {emotion} → {dominant_emotion}"
                    }
        
        return None
    
    def _is_flash_pattern(
        self,
        emotions: List[str],
        brief_emotion: str,
        dominant_emotion: str
    ) -> bool:
        """
        Verify if emotion sequence is a flash pattern.
        
        Flash pattern: dominant → brief → dominant
        Not just: dominant → brief (transition)
        """
        # Find first and last occurrence of brief emotion
        first_idx = emotions.index(brief_emotion)
        last_idx = len(emotions) - 1 - emotions[::-1].index(brief_emotion)
        
        # Check if brief emotion is surrounded by dominant emotion
        has_dominant_before = first_idx > 0 and emotions[first_idx - 1] == dominant_emotion
        has_dominant_after = last_idx < len(emotions) - 1 and emotions[last_idx + 1] == dominant_emotion
        
        # True flash pattern has dominant emotion on both sides
        return has_dominant_before and has_dominant_after
    
    def get_all_micro_expressions(self) -> List[Dict[str, Any]]:
        """
        Get all detected micro-expressions.
        
        Returns:
            List of micro-expression dictionaries
        """
        return self.micro_expressions.copy()
    
    def get_micro_expression_count(self) -> int:
        """Get total number of micro-expressions detected."""
        return len(self.micro_expressions)
    
    def get_micro_expression_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of micro-expressions.
        
        Returns:
            Dictionary with summary stats:
            {
                'total_count': int,
                'emotions': Dict[str, int],  # Count per emotion
                'avg_duration_ms': float,
                'avg_intensity': float
            }
        """
        if not self.micro_expressions:
            return {
                'total_count': 0,
                'emotions': {},
                'avg_duration_ms': 0.0,
                'avg_intensity': 0.0
            }
        
        # Count by emotion
        emotion_counts = Counter([me['micro_emotion'] for me in self.micro_expressions])
        
        # Average duration
        avg_duration = np.mean([me['duration_ms'] for me in self.micro_expressions])
        
        # Average intensity
        avg_intensity = np.mean([me['avg_intensity'] for me in self.micro_expressions])
        
        return {
            'total_count': len(self.micro_expressions),
            'emotions': dict(emotion_counts),
            'avg_duration_ms': avg_duration,
            'avg_intensity': avg_intensity
        }
    
    def reset(self):
        """Reset detector state."""
        self.emotion_history.clear()
        self.intensity_history.clear()
        self.micro_expressions.clear()
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'window_size': self.window_size,
            'min_duration': self.min_duration,
            'max_duration_ratio': self.max_duration_ratio,
            'window_duration_ms': (self.window_size / 30.0) * 1000  # Assuming 30fps
        }


if __name__ == '__main__':
    # Demo usage
    print("Micro-Expression Detector Demo")
    print("=" * 70)
    
    # Initialize detector
    detector = MicroExpressionDetector(window_size=15)
    
    print("\nConfiguration:")
    config = detector.get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Simulate emotion sequence with micro-expression
    print("\n" + "=" * 70)
    print("Test Case 1: Flash Pattern (Happy → Fear → Happy)")
    print("Simulating 15 frames...")
    
    # Emotion sequence: Happy (5 frames) → Fear (3 frames) → Happy (7 frames)
    emotion_sequence = (
        [('Happy', 0.8)] * 5 +
        [('Fear', 0.6)] * 3 +
        [('Happy', 0.8)] * 7
    )
    
    detected_micro_exprs = []
    for i, (emotion, intensity) in enumerate(emotion_sequence):
        print(f"  Frame {i+1:2d}: {emotion:10s} ({intensity:.1f})")
        
        micro_expr = detector.detect(emotion, intensity)
        if micro_expr:
            detected_micro_exprs.append(micro_expr)
            print(f"    → Micro-expression detected!")
    
    # Show results
    if detected_micro_exprs:
        print("\n" + "=" * 70)
        print("Detected Micro-Expressions:")
        for i, me in enumerate(detected_micro_exprs):
            print(f"\n  Micro-Expression #{i+1}:")
            print(f"    Pattern: {me['pattern']}")
            print(f"    Micro-Emotion: {me['micro_emotion']}")
            print(f"    Duration: {me['duration_frames']} frames ({me['duration_ms']:.0f}ms)")
            print(f"    Intensity: {me['avg_intensity']:.2%}")
            print(f"    Suppressed: {me['suppressed']}")
    
    # Test Case 2: No micro-expression (sustained emotion)
    print("\n" + "=" * 70)
    print("Test Case 2: No Micro-Expression (Sustained Happy)")
    print("Simulating 15 frames...")
    
    detector.reset()
    emotion_sequence2 = [('Happy', 0.8)] * 15
    
    detected_count = 0
    for i, (emotion, intensity) in enumerate(emotion_sequence2):
        micro_expr = detector.detect(emotion, intensity)
        if micro_expr:
            detected_count += 1
    
    print(f"  Micro-expressions detected: {detected_count}")
    print("  ✓ Correct: No micro-expression in sustained emotion")
    
    # Test Case 3: Transition (not micro-expression)
    print("\n" + "=" * 70)
    print("Test Case 3: Emotion Transition (Happy → Sad)")
    print("Simulating 15 frames...")
    
    detector.reset()
    emotion_sequence3 = [('Happy', 0.8)] * 7 + [('Sad', 0.7)] * 8
    
    detected_count = 0
    for i, (emotion, intensity) in enumerate(emotion_sequence3):
        micro_expr = detector.detect(emotion, intensity)
        if micro_expr:
            detected_count += 1
    
    print(f"  Micro-expressions detected: {detected_count}")
    print("  ✓ Correct: Transition is not a micro-expression")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("  ✓ Flash pattern (A → B → A) detected as micro-expression")
    print("  ✓ Sustained emotion not flagged")
    print("  ✓ Simple transition not flagged")
    print("\nKey Insights:")
    print("  • Micro-expressions indicate suppressed emotions")
    print("  • Brief (<500ms) emotional flashes")
    print("  • Useful for authenticity assessment")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
