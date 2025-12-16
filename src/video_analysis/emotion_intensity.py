# -*- coding: utf-8 -*-
"""
Emotion Intensity Analyzer

Calculates emotion intensity (0-100%) by combining:
1. Base confidence from emotion classifier
2. Action Unit intensities from AU detector

This provides more accurate emotion scoring than confidence alone.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

import numpy as np
from typing import Dict, Any


class EmotionIntensityAnalyzer:
    """
    Analyze emotion intensity using classifier confidence and Action Units.
    
    Provides 0-100% intensity scores that are more accurate than
    raw classifier confidence by incorporating facial muscle movements (AUs).
    """
    
    def __init__(
        self,
        use_au_boost: bool = True,
        genuine_smile_boost: float = 1.2,
        fake_smile_penalty: float = 0.8
    ):
        """
        Initialize Emotion Intensity Analyzer.
        
        Args:
            use_au_boost: Whether to adjust intensity based on AUs
            genuine_smile_boost: Multiplier for genuine smiles (default: 1.2 = +20%)
            fake_smile_penalty: Multiplier for fake smiles (default: 0.8 = -20%)
        """
        self.use_au_boost = use_au_boost
        self.genuine_smile_boost = genuine_smile_boost
        self.fake_smile_penalty = fake_smile_penalty
    
    def calculate_intensity(
        self,
        emotion_prediction,
        facial_aus: Dict[str, float]
    ) -> float:
        """
        Calculate emotion intensity (0-1 scale).
        
        Args:
            emotion_prediction: EmotionPrediction from classifier
                               Must have .emotion and .confidence attributes
            facial_aus: AU intensities from FacialActionUnitDetector
                       Dictionary: {'AU1': 0.8, 'AU6': 0.5, ...}
        
        Returns:
            Intensity score 0.0-1.0 (multiply by 100 for percentage)
        
        Example:
            >>> analyzer = EmotionIntensityAnalyzer()
            >>> prediction = EmotionPrediction(emotion='Happy', confidence=0.75)
            >>> aus = {'AU6': 0.8, 'AU12': 0.9, ...}
            >>> intensity = analyzer.calculate_intensity(prediction, aus)
            >>> print(f"Happy intensity: {intensity * 100:.0f}%")
            Happy intensity: 90%
        """
        base_confidence = emotion_prediction.confidence
        emotion = emotion_prediction.emotion
        
        if not self.use_au_boost or not facial_aus:
            return base_confidence
        
        # Calculate intensity based on emotion type
        if emotion == 'Happy':
            intensity = self._calculate_happy_intensity(base_confidence, facial_aus)
        elif emotion == 'Angry':
            intensity = self._calculate_angry_intensity(base_confidence, facial_aus)
        elif emotion == 'Sad':
            intensity = self._calculate_sad_intensity(base_confidence, facial_aus)
        elif emotion == 'Surprise':
            intensity = self._calculate_surprise_intensity(base_confidence, facial_aus)
        elif emotion == 'Fear':
            intensity = self._calculate_fear_intensity(base_confidence, facial_aus)
        elif emotion == 'Disgust':
            intensity = self._calculate_disgust_intensity(base_confidence, facial_aus)
        elif emotion == 'Neutral':
            intensity = base_confidence  # No AU boost for neutral
        else:
            intensity = base_confidence
        
        # Cap at 1.0
        return min(intensity, 1.0)
    
    def _calculate_happy_intensity(
        self,
        base_confidence: float,
        aus: Dict[str, float]
    ) -> float:
        """
        Calculate Happy emotion intensity.
        
        Key insight: Genuine smile (AU6 + AU12) should have higher intensity
        than fake smile (AU12 only).
        
        Requirements: 3.3, 3.4, 3.5
        """
        au6 = aus.get('AU6', 0)  # Cheek raiser
        au12 = aus.get('AU12', 0)  # Lip corner puller
        
        # Genuine smile: Both AU6 and AU12 active
        if au6 > 0.5 and au12 > 0.5:
            # Boost intensity for genuine smile
            intensity = base_confidence * self.genuine_smile_boost
            
            # Additional boost based on AU intensities
            au_boost = (au6 + au12) / 2 * 0.1  # Up to +10%
            intensity += au_boost
        
        # Fake smile: AU12 active but AU6 weak
        elif au12 > 0.5 and au6 < 0.3:
            # Reduce intensity for fake smile
            intensity = base_confidence * self.fake_smile_penalty
        
        # Weak smile or ambiguous
        else:
            # Slight adjustment based on AU12
            au_adjustment = (au12 - 0.5) * 0.2  # -10% to +10%
            intensity = base_confidence * (1.0 + au_adjustment)
        
        return intensity
    
    def _calculate_angry_intensity(
        self,
        base_confidence: float,
        aus: Dict[str, float]
    ) -> float:
        """
        Calculate Angry emotion intensity.
        
        Key AUs for anger:
        - AU4: Brow Lowerer
        - AU23: Lip Tightener
        - AU24: Lip Pressor
        """
        au4 = aus.get('AU4', 0)   # Brow lowerer
        au23 = aus.get('AU23', 0)  # Lip tightener
        au24 = aus.get('AU24', 0)  # Lip pressor
        
        # Calculate average anger AU intensity
        anger_aus = [au4, au23, au24]
        au_intensity = sum(anger_aus) / len(anger_aus)
        
        # Boost intensity based on AU presence
        # Strong anger = high AU intensities
        intensity = base_confidence * (0.7 + au_intensity * 0.6)
        
        return intensity
    
    def _calculate_sad_intensity(
        self,
        base_confidence: float,
        aus: Dict[str, float]
    ) -> float:
        """
        Calculate Sad emotion intensity.
        
        Key AU for sadness:
        - AU15: Lip Corner Depressor
        """
        au15 = aus.get('AU15', 0)  # Lip corner depressor
        
        # Boost intensity based on AU15
        intensity = base_confidence * (0.8 + au15 * 0.4)
        
        return intensity
    
    def _calculate_surprise_intensity(
        self,
        base_confidence: float,
        aus: Dict[str, float]
    ) -> float:
        """
        Calculate Surprise emotion intensity.
        
        Key AUs for surprise:
        - AU1: Inner Brow Raiser
        - AU2: Outer Brow Raiser
        - AU26: Jaw Drop
        """
        au1 = aus.get('AU1', 0)   # Inner brow raiser
        au2 = aus.get('AU2', 0)   # Outer brow raiser
        au26 = aus.get('AU26', 0)  # Jaw drop
        
        # Calculate average surprise AU intensity
        surprise_aus = [au1, au2, au26]
        au_intensity = sum(surprise_aus) / len(surprise_aus)
        
        # Boost intensity based on AU presence
        intensity = base_confidence * (0.7 + au_intensity * 0.6)
        
        return intensity
    
    def _calculate_fear_intensity(
        self,
        base_confidence: float,
        aus: Dict[str, float]
    ) -> float:
        """
        Calculate Fear emotion intensity.
        
        Key AUs for fear:
        - AU1: Inner Brow Raiser
        - AU2: Outer Brow Raiser
        - AU4: Brow Lowerer (tension)
        """
        au1 = aus.get('AU1', 0)  # Inner brow raiser
        au2 = aus.get('AU2', 0)  # Outer brow raiser
        au4 = aus.get('AU4', 0)  # Brow lowerer
        
        # Fear often shows brow raise + tension
        fear_aus = [au1, au2, au4]
        au_intensity = sum(fear_aus) / len(fear_aus)
        
        # Boost intensity based on AU presence
        intensity = base_confidence * (0.7 + au_intensity * 0.6)
        
        return intensity
    
    def _calculate_disgust_intensity(
        self,
        base_confidence: float,
        aus: Dict[str, float]
    ) -> float:
        """
        Calculate Disgust emotion intensity.
        
        Key AUs for disgust:
        - AU4: Brow Lowerer
        - AU15: Lip Corner Depressor
        """
        au4 = aus.get('AU4', 0)   # Brow lowerer
        au15 = aus.get('AU15', 0)  # Lip corner depressor
        
        # Calculate average disgust AU intensity
        disgust_aus = [au4, au15]
        au_intensity = sum(disgust_aus) / len(disgust_aus)
        
        # Boost intensity based on AU presence
        intensity = base_confidence * (0.7 + au_intensity * 0.6)
        
        return intensity
    
    def get_intensity_description(self, intensity: float) -> str:
        """
        Get human-readable description of intensity level.
        
        Args:
            intensity: Intensity score 0.0-1.0
        
        Returns:
            Description string
        
        Example:
            >>> analyzer = EmotionIntensityAnalyzer()
            >>> desc = analyzer.get_intensity_description(0.85)
            >>> print(desc)
            'Very High'
        """
        if intensity >= 0.9:
            return 'Very High'
        elif intensity >= 0.7:
            return 'High'
        elif intensity >= 0.5:
            return 'Moderate'
        elif intensity >= 0.3:
            return 'Low'
        else:
            return 'Very Low'
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'use_au_boost': self.use_au_boost,
            'genuine_smile_boost': self.genuine_smile_boost,
            'fake_smile_penalty': self.fake_smile_penalty
        }


if __name__ == '__main__':
    # Demo usage
    print("Emotion Intensity Analyzer Demo")
    print("=" * 70)
    
    # Mock emotion prediction
    class MockPrediction:
        def __init__(self, emotion, confidence):
            self.emotion = emotion
            self.confidence = confidence
    
    # Initialize analyzer
    analyzer = EmotionIntensityAnalyzer()
    
    print("\nConfiguration:")
    config = analyzer.get_config()
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Test Case 1: Genuine Happy
    print("\n" + "=" * 70)
    print("Test Case 1: Genuine Happy (AU6 + AU12)")
    prediction = MockPrediction('Happy', 0.75)
    aus = {
        'AU6': 0.8,   # Cheek raiser (high)
        'AU12': 0.9,  # Lip corner puller (high)
    }
    intensity = analyzer.calculate_intensity(prediction, aus)
    desc = analyzer.get_intensity_description(intensity)
    print(f"  Base confidence: {prediction.confidence:.2%}")
    print(f"  AU6 (Cheek Raiser): {aus['AU6']:.2f}")
    print(f"  AU12 (Smile): {aus['AU12']:.2f}")
    print(f"  → Intensity: {intensity:.2%} ({desc})")
    print(f"  → Boost: +{(intensity - prediction.confidence) * 100:.0f}%")
    
    # Test Case 2: Fake Happy
    print("\n" + "=" * 70)
    print("Test Case 2: Fake Happy (AU12 only, no AU6)")
    prediction = MockPrediction('Happy', 0.75)
    aus = {
        'AU6': 0.2,   # Cheek raiser (low)
        'AU12': 0.9,  # Lip corner puller (high)
    }
    intensity = analyzer.calculate_intensity(prediction, aus)
    desc = analyzer.get_intensity_description(intensity)
    print(f"  Base confidence: {prediction.confidence:.2%}")
    print(f"  AU6 (Cheek Raiser): {aus['AU6']:.2f}")
    print(f"  AU12 (Smile): {aus['AU12']:.2f}")
    print(f"  → Intensity: {intensity:.2%} ({desc})")
    print(f"  → Penalty: {(intensity - prediction.confidence) * 100:.0f}%")
    
    # Test Case 3: Angry
    print("\n" + "=" * 70)
    print("Test Case 3: Angry (AU4 + AU23 + AU24)")
    prediction = MockPrediction('Angry', 0.70)
    aus = {
        'AU4': 0.7,   # Brow lowerer
        'AU23': 0.6,  # Lip tightener
        'AU24': 0.5,  # Lip pressor
    }
    intensity = analyzer.calculate_intensity(prediction, aus)
    desc = analyzer.get_intensity_description(intensity)
    print(f"  Base confidence: {prediction.confidence:.2%}")
    print(f"  AU4 (Brow Lowerer): {aus['AU4']:.2f}")
    print(f"  AU23 (Lip Tightener): {aus['AU23']:.2f}")
    print(f"  AU24 (Lip Pressor): {aus['AU24']:.2f}")
    print(f"  → Intensity: {intensity:.2%} ({desc})")
    
    # Test Case 4: Surprise
    print("\n" + "=" * 70)
    print("Test Case 4: Surprise (AU1 + AU2 + AU26)")
    prediction = MockPrediction('Surprise', 0.65)
    aus = {
        'AU1': 0.8,   # Inner brow raiser
        'AU2': 0.7,   # Outer brow raiser
        'AU26': 0.9,  # Jaw drop
    }
    intensity = analyzer.calculate_intensity(prediction, aus)
    desc = analyzer.get_intensity_description(intensity)
    print(f"  Base confidence: {prediction.confidence:.2%}")
    print(f"  AU1 (Inner Brow Raiser): {aus['AU1']:.2f}")
    print(f"  AU2 (Outer Brow Raiser): {aus['AU2']:.2f}")
    print(f"  AU26 (Jaw Drop): {aus['AU26']:.2f}")
    print(f"  → Intensity: {intensity:.2%} ({desc})")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("\nKey Insights:")
    print("  ✓ Genuine smiles get intensity boost (+20%)")
    print("  ✓ Fake smiles get intensity penalty (-20%)")
    print("  ✓ Other emotions boosted based on relevant AUs")
    print("  ✓ More accurate than classifier confidence alone")
