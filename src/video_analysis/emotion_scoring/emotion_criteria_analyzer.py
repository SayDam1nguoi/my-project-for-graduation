"""
Emotion Criteria Analyzer for Recruitment Emotion Scoring System.

This module analyzes 4 core emotional criteria from facial features and expressions.
Each criterion is scored on a 0-10 scale based on detected indicators.

Requirements: 1.1, 2.1, 3.1, 4.1
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .models import FacialData


class EmotionCriteriaAnalyzer:
    """
    Analyzes 4 core emotional criteria from facial features.
    
    D. Cảm xúc - 5% (0-10 → x0.5 điểm)
    
    This class provides methods to evaluate 4 emotional criteria:
    1. Emotion Stability (40%) - Ổn định cảm xúc
    2. Emotion-Content Alignment (35%) - Khớp cảm xúc & nội dung
    3. Positive Ratio (15%) - Tích cực
    4. Negative Overload (10%) - Mức tiêu cực vượt ngưỡng
    
    Chú ý: Không dùng để đánh giá năng lực, chỉ đánh giá hành vi thể hiện.
    
    Each method returns a raw analysis score (0-1) that will be scaled
    to 0-10 by the EmotionScoringEngine, then multiplied by 0.5 for final score.
    """
    
    def __init__(self):
        """Initialize EmotionCriteriaAnalyzer."""
        # Emotion categories for analysis
        self.positive_emotions = {'happy'}
        self.negative_emotions = {'angry', 'disgust', 'sad', 'fear'}
        self.neutral_emotions = {'neutral'}
        self.extreme_negative_emotions = {'angry', 'disgust'}
        
        print("EmotionCriteriaAnalyzer initialized with new 4-criteria system (D. Cảm xúc - 5%)")

    def analyze_emotion_stability(
        self,
        facial_data_list: List[FacialData]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze emotion stability (Ổn định cảm xúc) - 40%.
        
        Emotion Stability is evaluated based on:
        - Consistency of emotional expressions over time
        - Low variance in emotion changes
        - Absence of erratic emotional shifts
        - Stable emotional baseline
        
        Args:
            facial_data_list: List of FacialData objects from video
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Emotion Stability score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - Evaluate emotional consistency throughout the video
            - Penalize frequent or extreme emotional changes
            - Reward stable, controlled emotional expression
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # 1. Track emotion sequence over time
        emotion_sequence = []
        emotion_intensities = []
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_intensity = dominant_emotion
            
            emotion_sequence.append(emotion_name)
            emotion_intensities.append(emotion_intensity)
        
        # 2. Calculate emotion change rate (lower = more stable)
        emotion_changes = sum(1 for i in range(1, len(emotion_sequence)) 
                             if emotion_sequence[i] != emotion_sequence[i-1])
        change_rate = emotion_changes / len(emotion_sequence) if len(emotion_sequence) > 1 else 0.0
        
        # Stability score based on change rate
        # Excellent: 0-10% changes, Good: 10-20%, Fair: 20-30%, Poor: >30%
        if change_rate <= 0.10:
            change_stability_score = 1.0
        elif change_rate <= 0.20:
            change_stability_score = 0.8
        elif change_rate <= 0.30:
            change_stability_score = 0.6
        else:
            change_stability_score = max(0.3, 1.0 - change_rate)
        
        # 3. Calculate emotion intensity variance (lower = more stable)
        intensity_variance = np.var(emotion_intensities) if emotion_intensities else 0.0
        
        # Optimal variance: 0.02-0.08 (not too flat, not too erratic)
        if intensity_variance < 0.02:
            intensity_stability_score = intensity_variance / 0.02  # Too flat
        elif intensity_variance <= 0.08:
            intensity_stability_score = 1.0  # Optimal
        else:
            intensity_stability_score = max(0.4, 1.0 - (intensity_variance - 0.08) / 0.15)
        
        # 4. Detect extreme emotional shifts (sudden large changes)
        extreme_shifts = 0
        for i in range(1, len(emotion_intensities)):
            intensity_change = abs(emotion_intensities[i] - emotion_intensities[i-1])
            if intensity_change > 0.4:  # Threshold for extreme shift
                extreme_shifts += 1
        
        extreme_shift_rate = extreme_shifts / len(emotion_intensities) if len(emotion_intensities) > 1 else 0.0
        extreme_shift_penalty = min(0.3, extreme_shift_rate * 2.0)  # Max 30% penalty
        
        # 5. Calculate baseline emotion consistency
        # Count most common emotion
        from collections import Counter
        emotion_counts = Counter(emotion_sequence)
        most_common_emotion, most_common_count = emotion_counts.most_common(1)[0]
        baseline_consistency = most_common_count / len(emotion_sequence) if emotion_sequence else 0.0
        
        # 6. Combined emotion stability score
        # Weight: change_rate (40%), intensity_variance (30%), baseline_consistency (30%)
        base_score = (
            change_stability_score * 0.40 +
            intensity_stability_score * 0.30 +
            baseline_consistency * 0.30
        )
        
        # Apply extreme shift penalty
        emotion_stability_score = max(0.0, base_score - extreme_shift_penalty)
        
        # Supporting data
        supporting_data = {
            'emotion_changes': emotion_changes,
            'change_rate': float(change_rate),
            'change_stability_score': float(change_stability_score),
            'intensity_variance': float(intensity_variance),
            'intensity_stability_score': float(intensity_stability_score),
            'extreme_shifts': extreme_shifts,
            'extreme_shift_rate': float(extreme_shift_rate),
            'extreme_shift_penalty': float(extreme_shift_penalty),
            'most_common_emotion': most_common_emotion,
            'baseline_consistency': float(baseline_consistency),
            'avg_intensity': float(np.mean(emotion_intensities)) if emotion_intensities else 0.0,
            'total_frames': len(facial_data_list)
        }
        
        return emotion_stability_score, supporting_data

    def analyze_emotion_content_alignment(
        self,
        facial_data_list: List[FacialData],
        content_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze emotion-content alignment (Khớp cảm xúc & nội dung) - 35%.
        
        Emotion-Content Alignment is evaluated based on:
        - Appropriateness of emotions for interview context
        - Consistency between facial expressions and expected responses
        - Absence of mismatched emotions (e.g., smiling during serious topics)
        - Natural emotional flow matching conversation rhythm
        
        Args:
            facial_data_list: List of FacialData objects from video
            content_context: Optional context about interview content/topics
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Emotion-Content Alignment score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - Evaluate appropriateness of emotional expressions
            - Detect mismatched or inappropriate emotions
            - Reward natural, contextually appropriate expressions
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # Since we don't have actual content/transcript, we evaluate based on
        # appropriateness of emotions for a professional interview context
        
        # 1. Define appropriate emotions for interview context
        appropriate_emotions = {'neutral', 'happy'}  # Professional interview emotions
        inappropriate_emotions = {'angry', 'disgust'}  # Clearly inappropriate
        borderline_emotions = {'sad', 'fear'}  # Context-dependent
        
        appropriate_count = 0
        inappropriate_count = 0
        borderline_count = 0
        
        # 2. Analyze emotion appropriateness
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_intensity = dominant_emotion
            
            if emotion_name in appropriate_emotions:
                appropriate_count += 1
            elif emotion_name in inappropriate_emotions and emotion_intensity > 0.5:
                inappropriate_count += 1
            elif emotion_name in borderline_emotions and emotion_intensity > 0.6:
                borderline_count += 1
        
        total_frames = len(facial_data_list)
        appropriate_ratio = appropriate_count / total_frames if total_frames > 0 else 0.0
        inappropriate_ratio = inappropriate_count / total_frames if total_frames > 0 else 0.0
        borderline_ratio = borderline_count / total_frames if total_frames > 0 else 0.0
        
        # 3. Evaluate emotional transitions (smooth = aligned, abrupt = misaligned)
        emotion_sequence = []
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            emotion_sequence.append(dominant_emotion)
        
        # Count inappropriate transitions (e.g., happy -> angry -> happy)
        inappropriate_transitions = 0
        for i in range(2, len(emotion_sequence)):
            prev_prev = emotion_sequence[i-2]
            prev = emotion_sequence[i-1]
            curr = emotion_sequence[i]
            
            # Detect erratic patterns
            if prev in inappropriate_emotions:
                inappropriate_transitions += 1
            elif prev_prev in appropriate_emotions and prev in borderline_emotions and curr in appropriate_emotions:
                # Brief negative emotion spike (might be misaligned)
                inappropriate_transitions += 0.5
        
        transition_penalty = min(0.3, (inappropriate_transitions / total_frames) * 2.0) if total_frames > 0 else 0.0
        
        # 4. Evaluate intensity appropriateness (extreme emotions = misaligned)
        extreme_intensity_count = 0
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            max_intensity = max(emotions.values())
            
            # Very high intensity (>0.9) might indicate overreaction
            if max_intensity > 0.9:
                extreme_intensity_count += 1
        
        extreme_intensity_ratio = extreme_intensity_count / total_frames if total_frames > 0 else 0.0
        intensity_penalty = min(0.2, extreme_intensity_ratio * 1.5)
        
        # 5. Calculate alignment score
        # Base score from appropriate emotions
        base_score = appropriate_ratio * 0.9 + (1.0 - inappropriate_ratio) * 0.1
        
        # Apply penalties
        alignment_score = max(0.0, base_score - transition_penalty - intensity_penalty - borderline_ratio * 0.3)
        
        # Supporting data
        supporting_data = {
            'appropriate_count': appropriate_count,
            'inappropriate_count': inappropriate_count,
            'borderline_count': borderline_count,
            'appropriate_ratio': float(appropriate_ratio),
            'inappropriate_ratio': float(inappropriate_ratio),
            'borderline_ratio': float(borderline_ratio),
            'inappropriate_transitions': float(inappropriate_transitions),
            'transition_penalty': float(transition_penalty),
            'extreme_intensity_count': extreme_intensity_count,
            'extreme_intensity_ratio': float(extreme_intensity_ratio),
            'intensity_penalty': float(intensity_penalty),
            'total_frames': total_frames
        }
        
        return alignment_score, supporting_data

    def analyze_positive_ratio(
        self,
        facial_data_list: List[FacialData]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze positive ratio (Tích cực) - 15%.
        
        Positive Ratio is evaluated based on:
        - Frequency of positive emotions (happy)
        - Presence of genuine smiles
        - Absence of negative emotions
        - Overall positive demeanor
        
        Args:
            facial_data_list: List of FacialData objects from video
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Positive Ratio score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - Calculate ratio of positive vs negative emotions
            - Reward high positive emotion frequency
            - Penalize negative emotion presence
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # 1. Count positive and negative emotions
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_intensity = dominant_emotion
            
            if emotion_name in self.positive_emotions and emotion_intensity > 0.4:
                positive_count += 1
            elif emotion_name in self.negative_emotions and emotion_intensity > 0.4:
                negative_count += 1
            elif emotion_name in self.neutral_emotions:
                neutral_count += 1
        
        total_frames = len(facial_data_list)
        
        # 2. Calculate ratios
        positive_ratio = positive_count / total_frames if total_frames > 0 else 0.0
        negative_ratio = negative_count / total_frames if total_frames > 0 else 0.0
        neutral_ratio = neutral_count / total_frames if total_frames > 0 else 0.0
        
        # 3. Calculate positive score
        # Ideal: 20-40% positive, <10% negative, rest neutral
        # Score based on positive presence and negative absence
        
        # Positive component (0-1)
        if positive_ratio < 0.10:
            positive_component = positive_ratio / 0.10  # Too low
        elif positive_ratio <= 0.40:
            positive_component = 1.0  # Optimal range
        else:
            positive_component = max(0.7, 1.0 - (positive_ratio - 0.40) / 0.30)  # Too high (might seem fake)
        
        # Negative component (penalty)
        if negative_ratio <= 0.05:
            negative_penalty = 0.0  # Acceptable
        elif negative_ratio <= 0.10:
            negative_penalty = (negative_ratio - 0.05) / 0.05 * 0.2  # Minor penalty
        else:
            negative_penalty = min(0.5, 0.2 + (negative_ratio - 0.10) * 2.0)  # Major penalty
        
        # 4. Analyze smile presence (bonus for genuine smiles)
        from .facial_data_extractor import FacialDataExtractor
        extractor = FacialDataExtractor()
        
        smile_count = 0
        genuine_smile_count = 0
        
        for facial_data in facial_data_list:
            au12 = facial_data.action_units.get('AU12', 0.0)
            
            if au12 > 0.3:
                smile_count += 1
                
                # Check if genuine
                authenticity = extractor.detect_smile_authenticity(facial_data)
                if authenticity > 0.6:
                    genuine_smile_count += 1
        
        smile_ratio = smile_count / total_frames if total_frames > 0 else 0.0
        genuine_smile_ratio = genuine_smile_count / total_frames if total_frames > 0 else 0.0
        
        # Smile bonus (up to +0.2)
        smile_bonus = min(0.2, genuine_smile_ratio * 0.5)
        
        # 5. Calculate final positive ratio score
        base_score = positive_component * 0.7 + neutral_ratio * 0.3
        positive_ratio_score = max(0.0, min(1.0, base_score + smile_bonus - negative_penalty))
        
        # Supporting data
        supporting_data = {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'positive_ratio': float(positive_ratio),
            'negative_ratio': float(negative_ratio),
            'neutral_ratio': float(neutral_ratio),
            'positive_component': float(positive_component),
            'negative_penalty': float(negative_penalty),
            'smile_count': smile_count,
            'genuine_smile_count': genuine_smile_count,
            'smile_ratio': float(smile_ratio),
            'genuine_smile_ratio': float(genuine_smile_ratio),
            'smile_bonus': float(smile_bonus),
            'total_frames': total_frames
        }
        
        return positive_ratio_score, supporting_data

    def analyze_negative_overload(
        self,
        facial_data_list: List[FacialData]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze negative overload (Mức tiêu cực vượt ngưỡng) - 10%.
        
        Negative Overload is evaluated based on:
        - Frequency of extreme negative emotions
        - Intensity of negative emotional displays
        - Duration of negative emotional states
        - Presence of anger, disgust, or intense sadness/fear
        
        Args:
            facial_data_list: List of FacialData objects from video
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Negative Overload score (0-1, higher = less overload)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - Detect excessive negative emotions
            - Penalize high-intensity negative displays
            - Reward absence of negative overload
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # 1. Count negative emotions by type and intensity
        extreme_negative_count = 0  # angry, disgust with high intensity
        moderate_negative_count = 0  # sad, fear with moderate intensity
        mild_negative_count = 0  # any negative with low intensity
        
        negative_intensity_sum = 0.0
        negative_duration_frames = 0
        consecutive_negative_frames = 0
        max_consecutive_negative = 0
        
        prev_is_negative = False
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_intensity = dominant_emotion
            
            is_negative = emotion_name in self.negative_emotions
            
            if is_negative and emotion_intensity > 0.3:
                negative_duration_frames += 1
                negative_intensity_sum += emotion_intensity
                
                # Track consecutive negative frames
                if prev_is_negative:
                    consecutive_negative_frames += 1
                else:
                    consecutive_negative_frames = 1
                
                max_consecutive_negative = max(max_consecutive_negative, consecutive_negative_frames)
                
                # Classify by severity
                if emotion_name in self.extreme_negative_emotions and emotion_intensity > 0.6:
                    extreme_negative_count += 1
                elif emotion_intensity > 0.5:
                    moderate_negative_count += 1
                else:
                    mild_negative_count += 1
            else:
                consecutive_negative_frames = 0
            
            prev_is_negative = is_negative and emotion_intensity > 0.3
        
        total_frames = len(facial_data_list)
        
        # 2. Calculate negative metrics
        negative_frequency = negative_duration_frames / total_frames if total_frames > 0 else 0.0
        avg_negative_intensity = negative_intensity_sum / negative_duration_frames if negative_duration_frames > 0 else 0.0
        max_consecutive_ratio = max_consecutive_negative / total_frames if total_frames > 0 else 0.0
        
        extreme_negative_ratio = extreme_negative_count / total_frames if total_frames > 0 else 0.0
        moderate_negative_ratio = moderate_negative_count / total_frames if total_frames > 0 else 0.0
        
        # 3. Calculate overload score (higher = less overload = better)
        # Thresholds:
        # - Extreme negative: >5% is concerning, >10% is severe
        # - Moderate negative: >15% is concerning, >25% is severe
        # - Max consecutive: >10% is concerning, >20% is severe
        
        # Extreme negative penalty (most severe)
        if extreme_negative_ratio <= 0.05:
            extreme_penalty = 0.0
        elif extreme_negative_ratio <= 0.10:
            extreme_penalty = (extreme_negative_ratio - 0.05) / 0.05 * 0.4  # Up to 40% penalty
        else:
            extreme_penalty = min(0.7, 0.4 + (extreme_negative_ratio - 0.10) * 3.0)  # Up to 70% penalty
        
        # Moderate negative penalty
        if moderate_negative_ratio <= 0.15:
            moderate_penalty = 0.0
        elif moderate_negative_ratio <= 0.25:
            moderate_penalty = (moderate_negative_ratio - 0.15) / 0.10 * 0.2  # Up to 20% penalty
        else:
            moderate_penalty = min(0.4, 0.2 + (moderate_negative_ratio - 0.25) * 2.0)  # Up to 40% penalty
        
        # Consecutive negative penalty (indicates sustained negativity)
        if max_consecutive_ratio <= 0.10:
            consecutive_penalty = 0.0
        elif max_consecutive_ratio <= 0.20:
            consecutive_penalty = (max_consecutive_ratio - 0.10) / 0.10 * 0.15  # Up to 15% penalty
        else:
            consecutive_penalty = min(0.3, 0.15 + (max_consecutive_ratio - 0.20) * 1.5)  # Up to 30% penalty
        
        # Intensity penalty (high average intensity of negative emotions)
        if avg_negative_intensity <= 0.5:
            intensity_penalty = 0.0
        elif avg_negative_intensity <= 0.7:
            intensity_penalty = (avg_negative_intensity - 0.5) / 0.2 * 0.1  # Up to 10% penalty
        else:
            intensity_penalty = min(0.2, 0.1 + (avg_negative_intensity - 0.7) * 0.5)  # Up to 20% penalty
        
        # 4. Calculate final score (1.0 = no overload, 0.0 = severe overload)
        total_penalty = extreme_penalty + moderate_penalty + consecutive_penalty + intensity_penalty
        negative_overload_score = max(0.0, 1.0 - total_penalty)
        
        # Supporting data
        supporting_data = {
            'extreme_negative_count': extreme_negative_count,
            'moderate_negative_count': moderate_negative_count,
            'mild_negative_count': mild_negative_count,
            'negative_duration_frames': negative_duration_frames,
            'negative_frequency': float(negative_frequency),
            'avg_negative_intensity': float(avg_negative_intensity),
            'max_consecutive_negative': max_consecutive_negative,
            'max_consecutive_ratio': float(max_consecutive_ratio),
            'extreme_negative_ratio': float(extreme_negative_ratio),
            'moderate_negative_ratio': float(moderate_negative_ratio),
            'extreme_penalty': float(extreme_penalty),
            'moderate_penalty': float(moderate_penalty),
            'consecutive_penalty': float(consecutive_penalty),
            'intensity_penalty': float(intensity_penalty),
            'total_penalty': float(total_penalty),
            'total_frames': total_frames
        }
        
        return negative_overload_score, supporting_data
