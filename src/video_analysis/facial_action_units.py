# -*- coding: utf-8 -*-
"""
Facial Action Units Detector

Detects Facial Action Units (AUs) based on FACS (Facial Action Coding System).
Uses facial landmarks to calculate AU intensities.

Key Action Units:
- AU1: Inner Brow Raiser (surprise, worry)
- AU2: Outer Brow Raiser (surprise)
- AU4: Brow Lowerer (anger, concentration)
- AU6: Cheek Raiser (genuine smile indicator)
- AU12: Lip Corner Puller (smile)
- AU15: Lip Corner Depressor (sadness)
- AU23: Lip Tightener (anger)
- AU24: Lip Pressor (anger, concentration)
- AU26: Jaw Drop (surprise)

Requirements: 2.1, 2.2, 2.4, 2.5
"""

import numpy as np
from typing import Dict, Any, Optional


class FacialActionUnitDetector:
    """
    Detect Facial Action Units from facial landmarks.
    
    Uses geometric measurements from 68-point facial landmarks
    to calculate AU intensities (0-1 scale).
    """
    
    def __init__(self):
        """Initialize AU detector."""
        # Baseline measurements (will be calibrated per face)
        self.baseline_measurements = {}
        
        # AU definitions (landmark indices)
        self.au_definitions = {
            'AU1': {'name': 'Inner Brow Raiser', 'landmarks': [19, 20, 37, 38]},
            'AU2': {'name': 'Outer Brow Raiser', 'landmarks': [22, 23, 42, 43]},
            'AU4': {'name': 'Brow Lowerer', 'landmarks': [19, 20, 21, 22, 23, 24]},
            'AU6': {'name': 'Cheek Raiser', 'landmarks': [40, 41, 46, 47]},
            'AU12': {'name': 'Lip Corner Puller', 'landmarks': [48, 54]},
            'AU15': {'name': 'Lip Corner Depressor', 'landmarks': [48, 54, 57]},
            'AU23': {'name': 'Lip Tightener', 'landmarks': [48, 54, 60, 64]},
            'AU24': {'name': 'Lip Pressor', 'landmarks': [51, 57, 62, 66]},
            'AU26': {'name': 'Jaw Drop', 'landmarks': [51, 57, 8]},
        }
    
    def detect_aus(self, face_landmarks: np.ndarray) -> Dict[str, float]:
        """
        Detect Action Units from facial landmarks.
        
        Args:
            face_landmarks: 68 facial landmarks (shape: 68x2)
                           From MTCNN or dlib face detector
        
        Returns:
            Dictionary: {'AU1': 0.8, 'AU2': 0.3, 'AU6': 0.9, ...}
                       Values are intensities from 0.0 to 1.0
        
        Example:
            >>> detector = FacialActionUnitDetector()
            >>> landmarks = np.array([[x1, y1], [x2, y2], ...])  # 68 points
            >>> aus = detector.detect_aus(landmarks)
            >>> print(aus['AU6'])  # Cheek raiser intensity
            0.85
        """
        if face_landmarks is None or len(face_landmarks) < 68:
            return self._get_zero_aus()
        
        aus = {}
        
        # AU1: Inner Brow Raiser
        # Measure vertical distance between inner brow and eye
        aus['AU1'] = self._calculate_au1(face_landmarks)
        
        # AU2: Outer Brow Raiser
        # Measure vertical distance between outer brow and eye
        aus['AU2'] = self._calculate_au2(face_landmarks)
        
        # AU4: Brow Lowerer
        # Measure brow compression
        aus['AU4'] = self._calculate_au4(face_landmarks)
        
        # AU6: Cheek Raiser (CRITICAL for genuine smile)
        # Measure lower eyelid raise
        aus['AU6'] = self._calculate_au6(face_landmarks)
        
        # AU12: Lip Corner Puller (smile)
        # Measure lip corner movement outward
        aus['AU12'] = self._calculate_au12(face_landmarks)
        
        # AU15: Lip Corner Depressor (sadness)
        # Measure lip corner movement downward
        aus['AU15'] = self._calculate_au15(face_landmarks)
        
        # AU23: Lip Tightener (anger)
        # Measure lip tightening
        aus['AU23'] = self._calculate_au23(face_landmarks)
        
        # AU24: Lip Pressor (anger, concentration)
        # Measure lip pressing
        aus['AU24'] = self._calculate_au24(face_landmarks)
        
        # AU26: Jaw Drop (surprise)
        # Measure mouth opening
        aus['AU26'] = self._calculate_au26(face_landmarks)
        
        return aus
    
    def _calculate_au1(self, landmarks: np.ndarray) -> float:
        """Calculate AU1: Inner Brow Raiser intensity."""
        # Inner brow points: 19, 20
        # Inner eye points: 37, 38
        inner_brow = (landmarks[19] + landmarks[20]) / 2
        inner_eye = (landmarks[37] + landmarks[38]) / 2
        
        # Vertical distance
        distance = inner_brow[1] - inner_eye[1]
        
        # Normalize (typical range: 10-30 pixels)
        # Negative distance = brow raised
        intensity = max(0, -distance / 20.0)
        
        return min(intensity, 1.0)
    
    def _calculate_au2(self, landmarks: np.ndarray) -> float:
        """Calculate AU2: Outer Brow Raiser intensity."""
        # Outer brow points: 22, 23
        # Outer eye points: 42, 43
        outer_brow = (landmarks[22] + landmarks[23]) / 2
        outer_eye = (landmarks[42] + landmarks[43]) / 2
        
        # Vertical distance
        distance = outer_brow[1] - outer_eye[1]
        
        # Normalize
        intensity = max(0, -distance / 20.0)
        
        return min(intensity, 1.0)
    
    def _calculate_au4(self, landmarks: np.ndarray) -> float:
        """Calculate AU4: Brow Lowerer intensity."""
        # Measure brow compression (distance between inner and outer brow)
        inner_brow = landmarks[19]
        outer_brow = landmarks[24]
        
        # Horizontal distance
        brow_width = np.linalg.norm(outer_brow - inner_brow)
        
        # Typical width: 60-80 pixels
        # Compressed = smaller width
        baseline_width = 70.0
        compression = (baseline_width - brow_width) / baseline_width
        
        intensity = max(0, compression)
        
        return min(intensity, 1.0)
    
    def _calculate_au6(self, landmarks: np.ndarray) -> float:
        """
        Calculate AU6: Cheek Raiser intensity.
        
        CRITICAL for genuine smile detection!
        Genuine smile = AU6 + AU12
        Fake smile = AU12 only (no AU6)
        """
        # Lower eyelid points: 40, 41 (left), 46, 47 (right)
        # When cheeks raise, lower eyelid moves up
        
        # Left eye
        left_upper = landmarks[37]
        left_lower = landmarks[41]
        left_eye_height = left_lower[1] - left_upper[1]
        
        # Right eye
        right_upper = landmarks[43]
        right_lower = landmarks[47]
        right_eye_height = right_lower[1] - right_upper[1]
        
        # Average eye height
        avg_eye_height = (left_eye_height + right_eye_height) / 2
        
        # Typical eye height: 10-15 pixels
        # When cheeks raise, eye height decreases
        baseline_height = 12.0
        cheek_raise = (baseline_height - avg_eye_height) / baseline_height
        
        intensity = max(0, cheek_raise)
        
        return min(intensity, 1.0)
    
    def _calculate_au12(self, landmarks: np.ndarray) -> float:
        """
        Calculate AU12: Lip Corner Puller intensity (smile).
        
        Present in both genuine and fake smiles.
        Must check AU6 to determine authenticity.
        """
        # Lip corners: 48 (left), 54 (right)
        left_corner = landmarks[48]
        right_corner = landmarks[54]
        
        # Lip width
        lip_width = np.linalg.norm(right_corner - left_corner)
        
        # Typical neutral lip width: 50-60 pixels
        # Smile = wider lips
        baseline_width = 55.0
        stretch = (lip_width - baseline_width) / baseline_width
        
        intensity = max(0, stretch)
        
        return min(intensity, 1.0)
    
    def _calculate_au15(self, landmarks: np.ndarray) -> float:
        """Calculate AU15: Lip Corner Depressor intensity (sadness)."""
        # Lip corners: 48, 54
        # Bottom lip center: 57
        left_corner = landmarks[48]
        right_corner = landmarks[54]
        bottom_center = landmarks[57]
        
        # Average corner height
        avg_corner_y = (left_corner[1] + right_corner[1]) / 2
        
        # Downward pull = corners below center
        depression = (avg_corner_y - bottom_center[1]) / 10.0
        
        intensity = max(0, depression)
        
        return min(intensity, 1.0)
    
    def _calculate_au23(self, landmarks: np.ndarray) -> float:
        """Calculate AU23: Lip Tightener intensity (anger)."""
        # Outer lip points: 48, 54, 60, 64
        # Inner lip points: 60, 64
        
        # Lip thickness (vertical distance)
        upper_outer = (landmarks[48] + landmarks[54]) / 2
        lower_outer = (landmarks[60] + landmarks[64]) / 2
        
        lip_thickness = np.linalg.norm(lower_outer - upper_outer)
        
        # Typical thickness: 15-20 pixels
        # Tightened = thinner lips
        baseline_thickness = 17.0
        tightening = (baseline_thickness - lip_thickness) / baseline_thickness
        
        intensity = max(0, tightening)
        
        return min(intensity, 1.0)
    
    def _calculate_au24(self, landmarks: np.ndarray) -> float:
        """Calculate AU24: Lip Pressor intensity (anger, concentration)."""
        # Upper lip: 51, 62
        # Lower lip: 57, 66
        
        upper_center = (landmarks[51] + landmarks[62]) / 2
        lower_center = (landmarks[57] + landmarks[66]) / 2
        
        # Vertical distance between lips
        lip_gap = np.linalg.norm(lower_center - upper_center)
        
        # Typical gap: 5-10 pixels
        # Pressed = smaller gap
        baseline_gap = 7.0
        pressing = (baseline_gap - lip_gap) / baseline_gap
        
        intensity = max(0, pressing)
        
        return min(intensity, 1.0)
    
    def _calculate_au26(self, landmarks: np.ndarray) -> float:
        """Calculate AU26: Jaw Drop intensity (surprise)."""
        # Upper lip center: 51
        # Lower lip center: 57
        # Chin: 8
        
        upper_lip = landmarks[51]
        lower_lip = landmarks[57]
        
        # Mouth opening (vertical distance)
        mouth_opening = lower_lip[1] - upper_lip[1]
        
        # Typical opening: 0-30 pixels
        # Jaw drop = large opening
        intensity = mouth_opening / 30.0
        
        return min(intensity, 1.0)
    
    def _get_zero_aus(self) -> Dict[str, float]:
        """Return zero intensities for all AUs."""
        return {
            'AU1': 0.0,
            'AU2': 0.0,
            'AU4': 0.0,
            'AU6': 0.0,
            'AU12': 0.0,
            'AU15': 0.0,
            'AU23': 0.0,
            'AU24': 0.0,
            'AU26': 0.0,
        }
    
    def map_aus_to_emotion(self, aus: Dict[str, float]) -> Dict[str, Any]:
        """
        Map Action Units to emotion indicators.
        
        Args:
            aus: AU intensities from detect_aus()
        
        Returns:
            Dictionary with emotion indicators:
            {
                'genuine_smile': bool,
                'fake_smile': bool,
                'anger_indicators': List[str],
                'stress_indicators': List[str],
                'confidence_level': float
            }
        
        Example:
            >>> aus = {'AU6': 0.8, 'AU12': 0.9, ...}
            >>> indicators = detector.map_aus_to_emotion(aus)
            >>> if indicators['genuine_smile']:
            ...     print("Genuine smile detected!")
        """
        indicators = {
            'genuine_smile': False,
            'fake_smile': False,
            'anger_indicators': [],
            'stress_indicators': [],
            'sadness_indicators': [],
            'surprise_indicators': [],
            'confidence_level': 0.5
        }
        
        # Genuine smile: AU6 + AU12 both high
        if aus['AU6'] > 0.5 and aus['AU12'] > 0.5:
            indicators['genuine_smile'] = True
            indicators['confidence_level'] = 0.9
        
        # Fake smile: AU12 high but AU6 low
        elif aus['AU12'] > 0.5 and aus['AU6'] < 0.3:
            indicators['fake_smile'] = True
            indicators['confidence_level'] = 0.6
        
        # Anger indicators: AU4 + AU23 + AU24
        if aus['AU4'] > 0.4:
            indicators['anger_indicators'].append('Brow Lowerer')
        if aus['AU23'] > 0.4:
            indicators['anger_indicators'].append('Lip Tightener')
        if aus['AU24'] > 0.4:
            indicators['anger_indicators'].append('Lip Pressor')
        
        # Stress indicators: AU1 + AU2 + AU4
        if aus['AU1'] > 0.4:
            indicators['stress_indicators'].append('Inner Brow Raiser')
        if aus['AU2'] > 0.4:
            indicators['stress_indicators'].append('Outer Brow Raiser')
        if aus['AU4'] > 0.4:
            indicators['stress_indicators'].append('Brow Lowerer')
        
        # Sadness indicators: AU15
        if aus['AU15'] > 0.4:
            indicators['sadness_indicators'].append('Lip Corner Depressor')
        
        # Surprise indicators: AU1 + AU2 + AU26
        if aus['AU1'] > 0.5 and aus['AU2'] > 0.5:
            indicators['surprise_indicators'].append('Brow Raisers')
        if aus['AU26'] > 0.5:
            indicators['surprise_indicators'].append('Jaw Drop')
        
        return indicators


if __name__ == '__main__':
    # Demo usage
    print("Facial Action Units Detector Demo")
    print("=" * 70)
    
    # Create dummy landmarks (68 points)
    dummy_landmarks = np.random.rand(68, 2) * 100
    
    # Initialize detector
    detector = FacialActionUnitDetector()
    
    # Detect AUs
    aus = detector.detect_aus(dummy_landmarks)
    
    print("\nDetected Action Units:")
    for au, intensity in aus.items():
        au_name = detector.au_definitions.get(au, {}).get('name', 'Unknown')
        print(f"  {au} ({au_name}): {intensity:.2f}")
    
    # Map to emotions
    indicators = detector.map_aus_to_emotion(aus)
    
    print("\nEmotion Indicators:")
    print(f"  Genuine Smile: {indicators['genuine_smile']}")
    print(f"  Fake Smile: {indicators['fake_smile']}")
    print(f"  Anger Indicators: {indicators['anger_indicators']}")
    print(f"  Stress Indicators: {indicators['stress_indicators']}")
    print(f"  Confidence Level: {indicators['confidence_level']:.2f}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
