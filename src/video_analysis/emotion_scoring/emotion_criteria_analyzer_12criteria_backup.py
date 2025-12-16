"""
Emotion Criteria Analyzer for Recruitment Emotion Scoring System.

This module analyzes specific emotional criteria from facial features and expressions.
Each criterion is scored on a 0-10 scale based on detected indicators.

Requirements: 1.1, 1.2, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .models import FacialData, MicroExpression, EmotionState, ConversationContext


class EmotionCriteriaAnalyzer:
    """
    Analyzes specific emotional criteria from facial features.
    
    This class provides methods to evaluate 12 emotional criteria:
    1. Calmness - emotional stability and composure
    2. Expression Control - appropriate facial expression management
    3. Confidence - self-assurance through posture, gaze, and expressions
    4. Friendliness - warmth and approachability
    5. Enthusiasm - energy and engagement
    6. Professionalism - appropriate demeanor and emotional control
    7. Authenticity - genuineness of emotional expressions
    8. Eye Contact - appropriate gaze patterns
    9. Facial Interaction - responsiveness through expressions
    10. Contextual Appropriateness - expressions matching context
    11. Emotion Consistency - emotional stability over time
    12. Sentiment Alignment - emotional synchronization with interviewer
    
    Each method returns a raw analysis score (0-1) that will be scaled
    to 0-10 by the EmotionScoringEngine.
    """
    
    def __init__(self):
        """Initialize EmotionCriteriaAnalyzer."""
        # Emotion categories for analysis
        self.calm_emotions = {'neutral', 'happy'}
        self.anxious_emotions = {'fear', 'surprise'}
        self.negative_emotions = {'angry', 'disgust', 'sad', 'fear'}
        self.positive_emotions = {'happy'}
        self.warm_emotions = {'happy'}
        
        print("EmotionCriteriaAnalyzer initialized")
    
    def analyze_calmness(
        self,
        facial_data_list: List[FacialData]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze calmness using facial expression patterns.
        
        Calmness is evaluated based on:
        - Frequency of calm emotions (neutral, happy)
        - Absence of anxiety indicators (fear, surprise)
        - Stability of emotional expressions
        - Low intensity of negative emotions
        
        Args:
            facial_data_list: List of FacialData objects from video
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Calmness score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - 1.1: Detect facial expressions indicating calmness levels
            - 1.2: Assign score based on frequency and intensity of calm expressions
            - 1.4: Reduce score when anxiety or stress is detected
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # Track calm and anxious emotion frequencies
        calm_count = 0
        anxious_count = 0
        total_frames = len(facial_data_list)
        
        # Track emotion intensities
        calm_intensities = []
        anxious_intensities = []
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            
            # Get dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_prob = dominant_emotion
            
            # Count calm emotions
            if emotion_name in self.calm_emotions:
                calm_count += 1
                calm_intensities.append(emotion_prob)
            
            # Count anxious emotions
            if emotion_name in self.anxious_emotions:
                anxious_count += 1
                anxious_intensities.append(emotion_prob)
        
        # Calculate calm emotion frequency
        calm_frequency = calm_count / total_frames if total_frames > 0 else 0.0
        
        # Calculate anxious emotion frequency
        anxious_frequency = anxious_count / total_frames if total_frames > 0 else 0.0
        
        # Calculate average intensities
        avg_calm_intensity = np.mean(calm_intensities) if calm_intensities else 0.0
        avg_anxious_intensity = np.mean(anxious_intensities) if anxious_intensities else 0.0
        
        # Calculate calmness score
        # High calm frequency and low anxious frequency = high calmness
        base_score = calm_frequency * avg_calm_intensity if calm_intensities else calm_frequency
        
        # Penalty for anxiety
        anxiety_penalty = anxious_frequency * avg_anxious_intensity if anxious_intensities else anxious_frequency
        
        # Final score (0-1)
        calmness_score = max(0.0, base_score - anxiety_penalty)
        
        # Supporting data
        supporting_data = {
            'calm_frequency': calm_frequency,
            'anxious_frequency': anxious_frequency,
            'avg_calm_intensity': float(avg_calm_intensity),
            'avg_anxious_intensity': float(avg_anxious_intensity),
            'total_frames': total_frames,
            'calm_frames': calm_count,
            'anxious_frames': anxious_count
        }
        
        return calmness_score, supporting_data
    
    def analyze_expression_control(
        self,
        facial_data_list: List[FacialData]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze expression control using action unit variance.
        
        Expression control is evaluated based on:
        - Stability of facial action units (low variance = good control)
        - Appropriate range of expressions (not too flat, not too extreme)
        - Consistency of emotional displays
        - Absence of excessive or inappropriate expressions
        
        Args:
            facial_data_list: List of FacialData objects from video
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Expression control score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - 2.1: Evaluate appropriateness and control of facial expressions
            - 2.3: Assign lower score for excessive or inappropriate expressions
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # Collect action unit activations across frames
        au_timeseries = {}
        
        for facial_data in facial_data_list:
            for au_name, au_value in facial_data.action_units.items():
                if au_name not in au_timeseries:
                    au_timeseries[au_name] = []
                au_timeseries[au_name].append(au_value)
        
        # Calculate variance for each action unit
        au_variances = {}
        for au_name, values in au_timeseries.items():
            au_variances[au_name] = np.var(values)
        
        # Calculate average variance across all AUs
        avg_variance = np.mean(list(au_variances.values())) if au_variances else 0.0
        
        # Calculate expression range (how much expressions vary)
        emotion_changes = 0
        prev_emotion = None
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            if prev_emotion is not None and dominant_emotion != prev_emotion:
                emotion_changes += 1
            
            prev_emotion = dominant_emotion
        
        # Calculate emotion change rate
        emotion_change_rate = emotion_changes / len(facial_data_list) if len(facial_data_list) > 1 else 0.0
        
        # Good expression control:
        # - Moderate variance (not too flat, not too erratic)
        # - Moderate emotion change rate (not robotic, not chaotic)
        
        # Optimal variance is around 0.05-0.15
        variance_score = 1.0 - min(1.0, abs(avg_variance - 0.10) / 0.10)
        
        # Optimal emotion change rate is around 0.05-0.15 (5-15% of frames)
        change_rate_score = 1.0 - min(1.0, abs(emotion_change_rate - 0.10) / 0.10)
        
        # Combined score
        expression_control_score = (variance_score + change_rate_score) / 2.0
        
        # Supporting data
        supporting_data = {
            'avg_au_variance': float(avg_variance),
            'emotion_change_rate': float(emotion_change_rate),
            'emotion_changes': emotion_changes,
            'total_frames': len(facial_data_list),
            'variance_score': float(variance_score),
            'change_rate_score': float(change_rate_score)
        }
        
        return expression_control_score, supporting_data
    
    def analyze_confidence(
        self,
        facial_data_list: List[FacialData],
        eye_contact_stats: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze confidence combining posture, gaze, and expressions.
        
        Confidence is evaluated based on:
        - Eye contact patterns (steady gaze indicates confidence)
        - Head pose (upright posture indicates confidence)
        - Positive emotional expressions
        - Absence of hesitation indicators
        
        Args:
            facial_data_list: List of FacialData objects from video
            eye_contact_stats: Optional pre-computed eye contact statistics
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Confidence score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - 3.1: Detect confidence indicators including posture, eye contact, and expressions
            - 3.3: Reduce score when hesitation or uncertainty is detected
            - 3.4: Increase score for steady eye contact and upright posture
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # 1. Analyze eye contact (if not provided, calculate it)
        if eye_contact_stats is None:
            from .facial_data_extractor import FacialDataExtractor
            extractor = FacialDataExtractor()
            eye_contact_stats = extractor.track_eye_contact(facial_data_list)
        
        eye_contact_score = eye_contact_stats['eye_contact_percentage'] / 100.0
        
        # 2. Analyze head pose (upright = confident)
        head_poses = []
        for facial_data in facial_data_list:
            pitch, yaw, roll = facial_data.head_pose
            head_poses.append((pitch, yaw, roll))
        
        # Calculate average head pose
        avg_pitch = np.mean([pose[0] for pose in head_poses])
        avg_yaw = np.mean([pose[1] for pose in head_poses])
        avg_roll = np.mean([pose[2] for pose in head_poses])
        
        # Upright posture: pitch near 0, yaw near 0, roll near 0
        # Score based on deviation from neutral
        pitch_deviation = abs(avg_pitch) / 90.0  # Normalize to 0-1
        yaw_deviation = abs(avg_yaw) / 90.0
        roll_deviation = abs(avg_roll) / 180.0
        
        posture_score = 1.0 - (pitch_deviation + yaw_deviation + roll_deviation) / 3.0
        posture_score = max(0.0, posture_score)
        
        # 3. Analyze facial expressions (positive emotions = confident)
        positive_count = 0
        negative_count = 0
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_prob = dominant_emotion
            
            if emotion_name in self.positive_emotions and emotion_prob > 0.5:
                positive_count += 1
            elif emotion_name in self.negative_emotions and emotion_prob > 0.5:
                negative_count += 1
        
        expression_score = positive_count / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        
        # Penalty for negative emotions
        negative_penalty = negative_count / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        expression_score = max(0.0, expression_score - negative_penalty * 0.5)
        
        # 4. Combined confidence score
        # Weight: eye contact (40%), posture (30%), expressions (30%)
        confidence_score = (
            eye_contact_score * 0.4 +
            posture_score * 0.3 +
            expression_score * 0.3
        )
        
        # Supporting data
        supporting_data = {
            'eye_contact_percentage': eye_contact_stats['eye_contact_percentage'],
            'eye_contact_score': float(eye_contact_score),
            'avg_head_pitch': float(avg_pitch),
            'avg_head_yaw': float(avg_yaw),
            'avg_head_roll': float(avg_roll),
            'posture_score': float(posture_score),
            'positive_expression_count': positive_count,
            'negative_expression_count': negative_count,
            'expression_score': float(expression_score),
            'total_frames': len(facial_data_list)
        }
        
        return confidence_score, supporting_data
    
    def analyze_friendliness(
        self,
        facial_data_list: List[FacialData]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze friendliness using smile detection and warm expressions.
        
        Friendliness is evaluated based on:
        - Frequency of genuine smiles (Duchenne smiles)
        - Presence of warm emotional expressions
        - Smile authenticity (AU6 + AU12 activation)
        - Absence of cold or distant expressions
        
        Args:
            facial_data_list: List of FacialData objects from video
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Friendliness score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - 4.1: Detect friendliness indicators such as genuine smiles and warm expressions
            - 4.3: Assign lower score for cold or distant expressions
            - 4.4: Distinguish between genuine and forced smiles
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        from .facial_data_extractor import FacialDataExtractor
        extractor = FacialDataExtractor()
        
        # Track smile authenticity and warm expressions
        genuine_smile_count = 0
        forced_smile_count = 0
        warm_expression_count = 0
        cold_expression_count = 0
        
        smile_authenticity_scores = []
        
        for facial_data in facial_data_list:
            # Detect smile authenticity
            authenticity = extractor.detect_smile_authenticity(facial_data)
            
            # Check if smiling (AU12 > 0.3)
            au12 = facial_data.action_units.get('AU12', 0.0)
            
            if au12 > 0.3:
                smile_authenticity_scores.append(authenticity)
                
                # Classify as genuine or forced
                if authenticity > 0.6:
                    genuine_smile_count += 1
                else:
                    forced_smile_count += 1
            
            # Check for warm expressions
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_prob = dominant_emotion
            
            if emotion_name in self.warm_emotions and emotion_prob > 0.5:
                warm_expression_count += 1
            elif emotion_name in {'angry', 'disgust'} and emotion_prob > 0.5:
                cold_expression_count += 1
        
        # Calculate metrics
        total_frames = len(facial_data_list)
        genuine_smile_frequency = genuine_smile_count / total_frames if total_frames > 0 else 0.0
        warm_expression_frequency = warm_expression_count / total_frames if total_frames > 0 else 0.0
        
        # Average smile authenticity
        avg_smile_authenticity = np.mean(smile_authenticity_scores) if smile_authenticity_scores else 0.0
        
        # Penalty for cold expressions
        cold_penalty = cold_expression_count / total_frames if total_frames > 0 else 0.0
        
        # Calculate friendliness score
        # Weight: genuine smiles (50%), warm expressions (30%), smile authenticity (20%)
        base_score = (
            genuine_smile_frequency * 0.5 +
            warm_expression_frequency * 0.3 +
            avg_smile_authenticity * 0.2
        )
        
        # Apply cold expression penalty
        friendliness_score = max(0.0, base_score - cold_penalty * 0.5)
        
        # Supporting data
        supporting_data = {
            'genuine_smile_count': genuine_smile_count,
            'forced_smile_count': forced_smile_count,
            'genuine_smile_frequency': float(genuine_smile_frequency),
            'warm_expression_count': warm_expression_count,
            'cold_expression_count': cold_expression_count,
            'warm_expression_frequency': float(warm_expression_frequency),
            'avg_smile_authenticity': float(avg_smile_authenticity),
            'total_frames': total_frames,
            'cold_penalty': float(cold_penalty)
        }
        
        return friendliness_score, supporting_data
    
    def analyze_enthusiasm(
        self,
        facial_data_list: List[FacialData]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze enthusiasm using facial animation and energy levels.
        
        Enthusiasm is evaluated based on:
        - Facial animation (frequency of expression changes)
        - Energy levels (intensity of emotional expressions)
        - Presence of excited/happy emotions
        - Dynamic range of facial movements
        
        Args:
            facial_data_list: List of FacialData objects from video
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Enthusiasm score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - 5.1: Detect enthusiasm indicators including facial animation and energy levels
            - 5.3: Assign lower score for low energy or disinterest
            - 5.4: Increase score for genuine excitement
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # 1. Measure facial animation (expression changes)
        emotion_changes = 0
        prev_emotion = None
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            if prev_emotion is not None and dominant_emotion != prev_emotion:
                emotion_changes += 1
            
            prev_emotion = dominant_emotion
        
        # Animation score: more changes = more animated (but not too chaotic)
        animation_rate = emotion_changes / len(facial_data_list) if len(facial_data_list) > 1 else 0.0
        # Optimal animation rate: 0.1-0.3 (10-30% of frames show changes)
        if animation_rate < 0.1:
            animation_score = animation_rate / 0.1  # Scale up low animation
        elif animation_rate <= 0.3:
            animation_score = 1.0  # Optimal range
        else:
            animation_score = max(0.0, 1.0 - (animation_rate - 0.3) / 0.3)  # Penalize excessive
        
        # 2. Measure energy levels (emotion intensity)
        energy_levels = []
        excited_count = 0
        low_energy_count = 0
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            
            # Get dominant emotion and its intensity
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_intensity = dominant_emotion
            
            energy_levels.append(emotion_intensity)
            
            # Count excited/happy emotions
            if emotion_name in {'happy', 'surprise'} and emotion_intensity > 0.5:
                excited_count += 1
            
            # Count low energy (neutral with low intensity)
            if emotion_name == 'neutral' and emotion_intensity > 0.7:
                low_energy_count += 1
        
        # Calculate average energy
        avg_energy = np.mean(energy_levels) if energy_levels else 0.0
        
        # Energy score based on average intensity
        energy_score = avg_energy
        
        # 3. Calculate excitement frequency
        excitement_frequency = excited_count / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        
        # 4. Penalty for low energy
        low_energy_penalty = low_energy_count / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        
        # 5. Measure action unit variance (facial movement dynamics)
        au_variances = []
        for au_name in ['AU1', 'AU2', 'AU6', 'AU12']:  # Key expressive AUs
            au_values = [fd.action_units.get(au_name, 0.0) for fd in facial_data_list]
            if au_values:
                au_variances.append(np.var(au_values))
        
        avg_au_variance = np.mean(au_variances) if au_variances else 0.0
        # Higher variance = more dynamic expressions
        dynamics_score = min(1.0, avg_au_variance * 10)  # Scale variance to 0-1
        
        # 6. Combined enthusiasm score
        # Weight: animation (25%), energy (30%), excitement (25%), dynamics (20%)
        base_score = (
            animation_score * 0.25 +
            energy_score * 0.30 +
            excitement_frequency * 0.25 +
            dynamics_score * 0.20
        )
        
        # Apply low energy penalty
        enthusiasm_score = max(0.0, base_score - low_energy_penalty * 0.3)
        
        # Supporting data
        supporting_data = {
            'animation_rate': float(animation_rate),
            'animation_score': float(animation_score),
            'emotion_changes': emotion_changes,
            'avg_energy': float(avg_energy),
            'energy_score': float(energy_score),
            'excited_count': excited_count,
            'excitement_frequency': float(excitement_frequency),
            'low_energy_count': low_energy_count,
            'low_energy_penalty': float(low_energy_penalty),
            'avg_au_variance': float(avg_au_variance),
            'dynamics_score': float(dynamics_score),
            'total_frames': len(facial_data_list)
        }
        
        return enthusiasm_score, supporting_data
    
    def analyze_professionalism(
        self,
        facial_data_list: List[FacialData]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze professionalism combining composure and emotional control.
        
        Professionalism is evaluated based on:
        - Emotional composure (stable, controlled emotions)
        - Appropriate emotional displays (not too extreme)
        - Absence of inappropriate emotions (anger, disgust)
        - Emotional maturity (balanced expressions)
        
        Args:
            facial_data_list: List of FacialData objects from video
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Professionalism score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - 6.1: Evaluate professionalism through facial expressions, demeanor, and emotional control
            - 6.3: Reduce score for inappropriate emotional displays
            - 6.4: Increase score for emotional maturity
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # 1. Analyze emotional composure (stability)
        emotion_sequence = []
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_sequence.append(dominant_emotion[0])
        
        # Calculate emotion stability (fewer changes = more stable)
        emotion_changes = sum(1 for i in range(1, len(emotion_sequence)) 
                             if emotion_sequence[i] != emotion_sequence[i-1])
        change_rate = emotion_changes / len(emotion_sequence) if len(emotion_sequence) > 1 else 0.0
        
        # Composure score: lower change rate = higher composure
        # Professional range: 0-15% change rate
        if change_rate <= 0.15:
            composure_score = 1.0
        else:
            composure_score = max(0.0, 1.0 - (change_rate - 0.15) / 0.35)
        
        # 2. Analyze emotional appropriateness
        appropriate_count = 0
        inappropriate_count = 0
        extreme_count = 0
        
        professional_emotions = {'neutral', 'happy'}
        inappropriate_emotions = {'angry', 'disgust'}
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_intensity = dominant_emotion
            
            # Count appropriate emotions
            if emotion_name in professional_emotions:
                appropriate_count += 1
            
            # Count inappropriate emotions
            if emotion_name in inappropriate_emotions and emotion_intensity > 0.5:
                inappropriate_count += 1
            
            # Count extreme emotions (very high intensity)
            if emotion_intensity > 0.9 and emotion_name not in professional_emotions:
                extreme_count += 1
        
        # Appropriateness score
        appropriate_frequency = appropriate_count / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        inappropriate_frequency = inappropriate_count / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        extreme_frequency = extreme_count / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        
        appropriateness_score = appropriate_frequency
        
        # 3. Analyze emotional control (action unit variance)
        au_variances = []
        for au_name in ['AU1', 'AU2', 'AU4', 'AU6', 'AU12']:
            au_values = [fd.action_units.get(au_name, 0.0) for fd in facial_data_list]
            if au_values:
                au_variances.append(np.var(au_values))
        
        avg_au_variance = np.mean(au_variances) if au_variances else 0.0
        
        # Professional control: moderate variance (not too flat, not too erratic)
        # Optimal variance: 0.03-0.10
        if avg_au_variance < 0.03:
            control_score = avg_au_variance / 0.03  # Too flat
        elif avg_au_variance <= 0.10:
            control_score = 1.0  # Optimal
        else:
            control_score = max(0.0, 1.0 - (avg_au_variance - 0.10) / 0.15)  # Too erratic
        
        # 4. Analyze emotional maturity (balanced expressions)
        # Maturity = appropriate emotions + controlled expressions + no extremes
        maturity_score = (appropriate_frequency + control_score) / 2.0
        
        # 5. Combined professionalism score
        # Weight: composure (30%), appropriateness (30%), control (20%), maturity (20%)
        base_score = (
            composure_score * 0.30 +
            appropriateness_score * 0.30 +
            control_score * 0.20 +
            maturity_score * 0.20
        )
        
        # Penalties for inappropriate and extreme emotions
        inappropriate_penalty = inappropriate_frequency * 0.5
        extreme_penalty = extreme_frequency * 0.3
        
        professionalism_score = max(0.0, base_score - inappropriate_penalty - extreme_penalty)
        
        # Supporting data
        supporting_data = {
            'emotion_changes': emotion_changes,
            'change_rate': float(change_rate),
            'composure_score': float(composure_score),
            'appropriate_count': appropriate_count,
            'inappropriate_count': inappropriate_count,
            'extreme_count': extreme_count,
            'appropriate_frequency': float(appropriate_frequency),
            'inappropriate_frequency': float(inappropriate_frequency),
            'extreme_frequency': float(extreme_frequency),
            'appropriateness_score': float(appropriateness_score),
            'avg_au_variance': float(avg_au_variance),
            'control_score': float(control_score),
            'maturity_score': float(maturity_score),
            'inappropriate_penalty': float(inappropriate_penalty),
            'extreme_penalty': float(extreme_penalty),
            'total_frames': len(facial_data_list)
        }
        
        return professionalism_score, supporting_data
    
    def analyze_authenticity(
        self,
        facial_data_list: List[FacialData],
        micro_expressions: Optional[List[MicroExpression]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze authenticity using micro-expression consistency.
        
        Authenticity is evaluated based on:
        - Consistency between micro-expressions and overall emotions
        - Alignment of facial expressions with emotional context
        - Absence of contradictory emotional signals
        - Natural emotional flow without suppression
        
        Args:
            facial_data_list: List of FacialData objects from video
            micro_expressions: Optional pre-detected micro-expressions
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Authenticity score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - 7.1: Detect authenticity by analyzing micro-expressions and emotional consistency
            - 7.3: Reduce score when inconsistencies between verbal and non-verbal cues are detected
            - 7.4: Increase score when micro-expressions align with stated emotions
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # 1. Detect micro-expressions if not provided
        if micro_expressions is None:
            from .facial_data_extractor import FacialDataExtractor
            extractor = FacialDataExtractor()
            micro_expressions = extractor.detect_micro_expressions(facial_data_list)
        
        # 2. Analyze micro-expression consistency
        consistent_micro_expr = 0
        inconsistent_micro_expr = 0
        
        for micro_expr in micro_expressions:
            # Find the dominant emotion around the micro-expression
            # Look at frames before and after
            start_idx = max(0, micro_expr.start_frame - 5)
            end_idx = min(len(facial_data_list), micro_expr.end_frame + 5)
            
            surrounding_emotions = []
            for i in range(start_idx, end_idx):
                if i < len(facial_data_list):
                    emotions = facial_data_list[i].emotion_probabilities
                    dominant = max(emotions.items(), key=lambda x: x[1])[0]
                    surrounding_emotions.append(dominant)
            
            # Check if micro-expression matches surrounding emotions
            if surrounding_emotions:
                most_common = max(set(surrounding_emotions), key=surrounding_emotions.count)
                
                if micro_expr.expression_type == most_common:
                    consistent_micro_expr += 1
                else:
                    inconsistent_micro_expr += 1
        
        # Micro-expression consistency score
        total_micro_expr = len(micro_expressions)
        if total_micro_expr > 0:
            micro_expr_consistency = consistent_micro_expr / total_micro_expr
        else:
            # No micro-expressions detected = neutral (assume authentic)
            micro_expr_consistency = 0.7
        
        # 3. Analyze emotional flow (natural transitions)
        emotion_intensities = []
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_intensity = max(emotions.values())
            emotion_intensities.append(dominant_intensity)
        
        # Calculate intensity changes
        intensity_changes = []
        for i in range(1, len(emotion_intensities)):
            change = abs(emotion_intensities[i] - emotion_intensities[i-1])
            intensity_changes.append(change)
        
        # Natural flow: gradual changes (not abrupt)
        avg_intensity_change = np.mean(intensity_changes) if intensity_changes else 0.0
        
        # Optimal change rate: 0.05-0.15 (gradual, natural)
        if avg_intensity_change < 0.05:
            flow_score = 0.8  # Too flat (possibly suppressed)
        elif avg_intensity_change <= 0.15:
            flow_score = 1.0  # Natural
        else:
            flow_score = max(0.5, 1.0 - (avg_intensity_change - 0.15) / 0.20)  # Too abrupt
        
        # 4. Analyze smile authenticity (genuine vs forced)
        from .facial_data_extractor import FacialDataExtractor
        extractor = FacialDataExtractor()
        
        smile_authenticity_scores = []
        for facial_data in facial_data_list:
            au12 = facial_data.action_units.get('AU12', 0.0)
            if au12 > 0.3:  # Smiling
                authenticity = extractor.detect_smile_authenticity(facial_data)
                smile_authenticity_scores.append(authenticity)
        
        avg_smile_authenticity = np.mean(smile_authenticity_scores) if smile_authenticity_scores else 0.7
        
        # 5. Analyze emotional variance (authentic emotions vary naturally)
        emotion_counts = {}
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant = max(emotions.items(), key=lambda x: x[1])[0]
            emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1
        
        # Calculate emotion diversity (Shannon entropy)
        total_frames = len(facial_data_list)
        emotion_diversity = 0.0
        for count in emotion_counts.values():
            p = count / total_frames
            if p > 0:
                emotion_diversity -= p * np.log2(p)
        
        # Normalize diversity (max entropy for 7 emotions = log2(7) ≈ 2.807)
        max_entropy = np.log2(7)
        diversity_score = emotion_diversity / max_entropy if max_entropy > 0 else 0.0
        
        # Optimal diversity: 0.3-0.7 (some variety, but not chaotic)
        if diversity_score < 0.3:
            diversity_score = diversity_score / 0.3  # Too monotonous
        elif diversity_score <= 0.7:
            diversity_score = 1.0  # Optimal
        else:
            diversity_score = max(0.5, 1.0 - (diversity_score - 0.7) / 0.3)  # Too varied
        
        # 6. Combined authenticity score
        # Weight: micro-expr consistency (30%), flow (25%), smile authenticity (25%), diversity (20%)
        authenticity_score = (
            micro_expr_consistency * 0.30 +
            flow_score * 0.25 +
            avg_smile_authenticity * 0.25 +
            diversity_score * 0.20
        )
        
        # Penalty for inconsistent micro-expressions
        if total_micro_expr > 0:
            inconsistency_penalty = (inconsistent_micro_expr / total_micro_expr) * 0.2
            authenticity_score = max(0.0, authenticity_score - inconsistency_penalty)
        
        # Supporting data
        supporting_data = {
            'total_micro_expressions': total_micro_expr,
            'consistent_micro_expr': consistent_micro_expr,
            'inconsistent_micro_expr': inconsistent_micro_expr,
            'micro_expr_consistency': float(micro_expr_consistency),
            'avg_intensity_change': float(avg_intensity_change),
            'flow_score': float(flow_score),
            'smile_count': len(smile_authenticity_scores),
            'avg_smile_authenticity': float(avg_smile_authenticity),
            'emotion_diversity': float(emotion_diversity),
            'diversity_score': float(diversity_score),
            'total_frames': len(facial_data_list)
        }
        
        return authenticity_score, supporting_data
    
    def analyze_eye_contact(
        self,
        facial_data_list: List[FacialData],
        eye_contact_stats: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze eye contact using gaze tracking data.
        
        Eye contact is evaluated based on:
        - Percentage of time maintaining eye contact
        - Consistency of eye contact patterns
        - Appropriate duration of eye contact
        - Natural eye contact rhythm (not staring, not avoiding)
        
        Args:
            facial_data_list: List of FacialData objects from video
            eye_contact_stats: Optional pre-computed eye contact statistics
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Eye contact score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - 8.1: Track eye contact patterns and duration
            - 8.3: Reduce score when candidate avoids eye contact frequently
            - 8.4: Assign higher score when eye contact duration and frequency are within professional norms
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # 1. Calculate eye contact statistics if not provided
        if eye_contact_stats is None:
            from .facial_data_extractor import FacialDataExtractor
            extractor = FacialDataExtractor()
            eye_contact_stats = extractor.track_eye_contact(facial_data_list)
        
        # 2. Analyze eye contact percentage
        eye_contact_percentage = eye_contact_stats['eye_contact_percentage']
        
        # Professional eye contact: 50-70% of the time
        # Too little = avoidant, too much = staring
        if eye_contact_percentage < 30:
            percentage_score = eye_contact_percentage / 30  # Very low
        elif 50 <= eye_contact_percentage <= 70:
            percentage_score = 1.0  # Optimal
        elif eye_contact_percentage < 50:
            percentage_score = 0.7 + (eye_contact_percentage - 30) / 20 * 0.3  # Below optimal
        else:  # > 70
            percentage_score = max(0.5, 1.0 - (eye_contact_percentage - 70) / 30 * 0.5)  # Staring
        
        # 3. Analyze eye contact consistency (rhythm)
        # Track eye contact segments
        eye_contact_segments = []
        current_segment_length = 0
        in_eye_contact = False
        
        for facial_data in facial_data_list:
            gaze_x, gaze_y = facial_data.gaze_direction
            gaze_magnitude = np.sqrt(gaze_x**2 + gaze_y**2)
            
            is_eye_contact = gaze_magnitude <= 0.3
            
            if is_eye_contact:
                if not in_eye_contact:
                    if current_segment_length > 0:
                        eye_contact_segments.append(('away', current_segment_length))
                    current_segment_length = 1
                    in_eye_contact = True
                else:
                    current_segment_length += 1
            else:
                if in_eye_contact:
                    eye_contact_segments.append(('contact', current_segment_length))
                    current_segment_length = 1
                    in_eye_contact = False
                else:
                    current_segment_length += 1
        
        # Add final segment
        if current_segment_length > 0:
            segment_type = 'contact' if in_eye_contact else 'away'
            eye_contact_segments.append((segment_type, current_segment_length))
        
        # 4. Analyze segment patterns
        contact_segments = [length for seg_type, length in eye_contact_segments if seg_type == 'contact']
        away_segments = [length for seg_type, length in eye_contact_segments if seg_type == 'away']
        
        # Calculate average segment lengths
        avg_contact_length = np.mean(contact_segments) if contact_segments else 0.0
        avg_away_length = np.mean(away_segments) if away_segments else 0.0
        
        # Professional pattern: moderate segment lengths (not too brief, not too long)
        # Optimal contact segment: 30-90 frames (1-3 seconds at 30fps)
        if 30 <= avg_contact_length <= 90:
            contact_pattern_score = 1.0
        elif avg_contact_length < 30:
            contact_pattern_score = avg_contact_length / 30  # Too brief
        else:
            contact_pattern_score = max(0.5, 1.0 - (avg_contact_length - 90) / 90)  # Too long (staring)
        
        # Optimal away segment: 15-60 frames (0.5-2 seconds at 30fps)
        if 15 <= avg_away_length <= 60:
            away_pattern_score = 1.0
        elif avg_away_length < 15:
            away_pattern_score = 0.9  # Very brief looks away (good)
        else:
            away_pattern_score = max(0.3, 1.0 - (avg_away_length - 60) / 60)  # Too long (avoiding)
        
        # 5. Analyze gaze stability (low variance = steady gaze)
        gaze_x_values = [fd.gaze_direction[0] for fd in facial_data_list]
        gaze_y_values = [fd.gaze_direction[1] for fd in facial_data_list]
        
        gaze_x_variance = np.var(gaze_x_values)
        gaze_y_variance = np.var(gaze_y_values)
        avg_gaze_variance = (gaze_x_variance + gaze_y_variance) / 2
        
        # Lower variance = more stable (better)
        # Optimal variance: 0.05-0.15
        if avg_gaze_variance <= 0.15:
            stability_score = 1.0
        else:
            stability_score = max(0.5, 1.0 - (avg_gaze_variance - 0.15) / 0.20)
        
        # 6. Combined eye contact score
        # Weight: percentage (40%), contact pattern (25%), away pattern (20%), stability (15%)
        eye_contact_score = (
            percentage_score * 0.40 +
            contact_pattern_score * 0.25 +
            away_pattern_score * 0.20 +
            stability_score * 0.15
        )
        
        # Supporting data
        supporting_data = {
            'eye_contact_percentage': eye_contact_percentage,
            'eye_contact_frames': eye_contact_stats['eye_contact_frames'],
            'total_frames': eye_contact_stats['total_frames'],
            'percentage_score': float(percentage_score),
            'num_contact_segments': len(contact_segments),
            'num_away_segments': len(away_segments),
            'avg_contact_length': float(avg_contact_length),
            'avg_away_length': float(avg_away_length),
            'contact_pattern_score': float(contact_pattern_score),
            'away_pattern_score': float(away_pattern_score),
            'avg_gaze_variance': float(avg_gaze_variance),
            'stability_score': float(stability_score)
        }
        
        return eye_contact_score, supporting_data

    def analyze_facial_interaction(
        self,
        facial_data_list: List[FacialData],
        context: Optional[ConversationContext] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze facial interaction evaluating responsiveness.
        
        Facial interaction is evaluated based on:
        - Responsiveness to questions (facial reactions at question timestamps)
        - Appropriate facial reactions to conversation
        - Engagement through facial expressions
        - Dynamic facial responses (not static)
        
        Args:
            facial_data_list: List of FacialData objects from video
            context: Optional conversation context with question timestamps
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Facial interaction score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - 9.1: Evaluate facial responsiveness to questions and conversation
            - 9.3: Assign lower score for minimal facial responsiveness
            - 9.4: Increase score when facial expressions appropriately match conversation context
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # Use default context if not provided
        if context is None:
            context = ConversationContext.default()
        
        # 1. Analyze overall facial responsiveness (expression changes)
        emotion_changes = 0
        prev_emotion = None
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            
            if prev_emotion is not None and dominant_emotion != prev_emotion:
                emotion_changes += 1
            
            prev_emotion = dominant_emotion
        
        # Calculate responsiveness rate
        responsiveness_rate = emotion_changes / len(facial_data_list) if len(facial_data_list) > 1 else 0.0
        
        # Good responsiveness: 0.05-0.20 (5-20% of frames show changes)
        if responsiveness_rate < 0.05:
            responsiveness_score = responsiveness_rate / 0.05  # Too static
        elif responsiveness_rate <= 0.20:
            responsiveness_score = 1.0  # Optimal
        else:
            responsiveness_score = max(0.5, 1.0 - (responsiveness_rate - 0.20) / 0.30)  # Too reactive
        
        # 2. Analyze responsiveness to questions (if timestamps available)
        question_response_score = 0.7  # Default neutral score
        reactions_at_questions = 0
        
        if context.question_timestamps:
            # For each question, check if there's a facial reaction within 3 seconds
            for q_timestamp in context.question_timestamps:
                # Find frames within 3 seconds after question
                response_window_start = q_timestamp
                response_window_end = q_timestamp + 3.0
                
                # Get frames in this window
                window_frames = [
                    fd for fd in facial_data_list
                    if response_window_start <= fd.timestamp <= response_window_end
                ]
                
                if len(window_frames) >= 2:
                    # Check for emotion changes in window
                    window_emotions = [
                        max(fd.emotion_probabilities.items(), key=lambda x: x[1])[0]
                        for fd in window_frames
                    ]
                    
                    # Count unique emotions in window
                    unique_emotions = len(set(window_emotions))
                    
                    # Reaction detected if emotions change
                    if unique_emotions > 1:
                        reactions_at_questions += 1
            
            # Calculate question response score
            if len(context.question_timestamps) > 0:
                question_response_score = reactions_at_questions / len(context.question_timestamps)
        
        # 3. Analyze engagement through action unit activity
        au_activities = []
        for au_name in ['AU1', 'AU2', 'AU4', 'AU6', 'AU12']:  # Key expressive AUs
            au_values = [fd.action_units.get(au_name, 0.0) for fd in facial_data_list]
            if au_values:
                # Calculate activity as mean absolute deviation
                mean_au = np.mean(au_values)
                activity = np.mean([abs(v - mean_au) for v in au_values])
                au_activities.append(activity)
        
        avg_au_activity = np.mean(au_activities) if au_activities else 0.0
        
        # Higher activity = more engaged
        # Optimal activity: 0.05-0.15
        if avg_au_activity < 0.05:
            engagement_score = avg_au_activity / 0.05  # Too passive
        elif avg_au_activity <= 0.15:
            engagement_score = 1.0  # Optimal
        else:
            engagement_score = max(0.6, 1.0 - (avg_au_activity - 0.15) / 0.20)  # Too animated
        
        # 4. Analyze facial expression variety
        emotion_counts = {}
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant = max(emotions.items(), key=lambda x: x[1])[0]
            emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1
        
        # Calculate variety (number of different emotions shown)
        num_emotions = len(emotion_counts)
        
        # Good variety: 2-4 different emotions
        if num_emotions < 2:
            variety_score = 0.5  # Too monotonous
        elif 2 <= num_emotions <= 4:
            variety_score = 1.0  # Good variety
        else:
            variety_score = max(0.6, 1.0 - (num_emotions - 4) / 3)  # Too varied
        
        # 5. Combined facial interaction score
        # Weight: responsiveness (30%), question response (25%), engagement (25%), variety (20%)
        facial_interaction_score = (
            responsiveness_score * 0.30 +
            question_response_score * 0.25 +
            engagement_score * 0.25 +
            variety_score * 0.20
        )
        
        # Supporting data
        supporting_data = {
            'emotion_changes': emotion_changes,
            'responsiveness_rate': float(responsiveness_rate),
            'responsiveness_score': float(responsiveness_score),
            'num_questions': len(context.question_timestamps),
            'reactions_at_questions': reactions_at_questions,
            'question_response_score': float(question_response_score),
            'avg_au_activity': float(avg_au_activity),
            'engagement_score': float(engagement_score),
            'num_emotions_shown': num_emotions,
            'variety_score': float(variety_score),
            'total_frames': len(facial_data_list)
        }
        
        return facial_interaction_score, supporting_data

    def analyze_contextual_appropriateness(
        self,
        facial_data_list: List[FacialData],
        context: Optional[ConversationContext] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze contextual appropriateness matching expressions to context.
        
        Contextual appropriateness is evaluated based on:
        - Matching facial expressions to conversation context
        - Appropriate emotional responses for interview phase
        - Alignment with interviewer tone
        - Empathy and understanding through appropriate expressions
        
        Args:
            facial_data_list: List of FacialData objects from video
            context: Optional conversation context
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Contextual appropriateness score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - 10.1: Evaluate whether facial expressions match conversation context
            - 10.3: Reduce score when facial expressions are mismatched with context
            - 10.4: Increase score when candidate displays empathy and understanding
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # Use default context if not provided
        if context is None:
            context = ConversationContext.default()
        
        # 1. Analyze alignment with expected emotions for interview phase
        appropriate_count = 0
        inappropriate_count = 0
        
        # Define appropriate emotions for different interview contexts
        if context.interviewer_tone == 'formal':
            appropriate_emotions = {'neutral', 'happy'}
            inappropriate_emotions = {'angry', 'disgust', 'fear'}
        elif context.interviewer_tone == 'casual':
            appropriate_emotions = {'happy', 'neutral', 'surprise'}
            inappropriate_emotions = {'angry', 'disgust', 'sad'}
        else:  # friendly
            appropriate_emotions = {'happy', 'surprise'}
            inappropriate_emotions = {'angry', 'disgust', 'fear', 'sad'}
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_intensity = dominant_emotion
            
            if emotion_name in appropriate_emotions:
                appropriate_count += 1
            elif emotion_name in inappropriate_emotions and emotion_intensity > 0.5:
                inappropriate_count += 1
        
        # Calculate appropriateness frequency
        appropriate_frequency = appropriate_count / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        inappropriate_frequency = inappropriate_count / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        
        # Appropriateness score
        appropriateness_score = appropriate_frequency
        
        # 2. Analyze empathy indicators (mirroring positive emotions)
        empathy_count = 0
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            
            # Empathy indicators: happy, surprise (positive engagement)
            if emotions.get('happy', 0.0) > 0.4 or emotions.get('surprise', 0.0) > 0.4:
                empathy_count += 1
        
        empathy_frequency = empathy_count / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        
        # Good empathy: 20-60% of frames show positive engagement
        if empathy_frequency < 0.20:
            empathy_score = empathy_frequency / 0.20  # Too low
        elif empathy_frequency <= 0.60:
            empathy_score = 1.0  # Optimal
        else:
            empathy_score = max(0.7, 1.0 - (empathy_frequency - 0.60) / 0.40)  # Too much
        
        # 3. Analyze understanding indicators (nodding, attentive expressions)
        # Use head pose changes as proxy for nodding
        head_pitch_changes = []
        for i in range(1, len(facial_data_list)):
            prev_pitch = facial_data_list[i-1].head_pose[0]
            curr_pitch = facial_data_list[i].head_pose[0]
            pitch_change = abs(curr_pitch - prev_pitch)
            head_pitch_changes.append(pitch_change)
        
        # Count significant pitch changes (potential nods)
        nod_threshold = 5.0  # degrees
        nod_count = sum(1 for change in head_pitch_changes if change > nod_threshold)
        
        # Normalize nod frequency
        nod_frequency = nod_count / len(facial_data_list) if len(facial_data_list) > 1 else 0.0
        
        # Good nod frequency: 0.02-0.10 (2-10% of frames)
        if nod_frequency < 0.02:
            understanding_score = nod_frequency / 0.02  # Too few
        elif nod_frequency <= 0.10:
            understanding_score = 1.0  # Optimal
        else:
            understanding_score = max(0.6, 1.0 - (nod_frequency - 0.10) / 0.15)  # Too many
        
        # 4. Analyze emotional alignment with question types (if available)
        alignment_score = 0.7  # Default neutral score
        aligned_responses = 0
        
        if context.question_timestamps and context.question_types:
            for q_timestamp, q_type in zip(context.question_timestamps, context.question_types):
                # Find frames within 5 seconds after question
                response_window_start = q_timestamp
                response_window_end = q_timestamp + 5.0
                
                # Get frames in this window
                window_frames = [
                    fd for fd in facial_data_list
                    if response_window_start <= fd.timestamp <= response_window_end
                ]
                
                if window_frames:
                    # Get dominant emotions in window
                    window_emotions = []
                    for fd in window_frames:
                        emotions = fd.emotion_probabilities
                        dominant = max(emotions.items(), key=lambda x: x[1])[0]
                        window_emotions.append(dominant)
                    
                    # Check if emotions match expected for question type
                    expected = context.expected_emotions.get(q_type, ['neutral', 'happy'])
                    
                    # Count how many frames show expected emotions
                    matching_frames = sum(1 for e in window_emotions if e in expected)
                    
                    if matching_frames / len(window_frames) > 0.5:
                        aligned_responses += 1
            
            # Calculate alignment score
            if len(context.question_timestamps) > 0:
                alignment_score = aligned_responses / len(context.question_timestamps)
        
        # 5. Combined contextual appropriateness score
        # Weight: appropriateness (35%), empathy (25%), understanding (20%), alignment (20%)
        contextual_score = (
            appropriateness_score * 0.35 +
            empathy_score * 0.25 +
            understanding_score * 0.20 +
            alignment_score * 0.20
        )
        
        # Penalty for inappropriate emotions
        inappropriate_penalty = inappropriate_frequency * 0.3
        contextual_score = max(0.0, contextual_score - inappropriate_penalty)
        
        # Supporting data
        supporting_data = {
            'appropriate_count': appropriate_count,
            'inappropriate_count': inappropriate_count,
            'appropriate_frequency': float(appropriate_frequency),
            'inappropriate_frequency': float(inappropriate_frequency),
            'appropriateness_score': float(appropriateness_score),
            'empathy_count': empathy_count,
            'empathy_frequency': float(empathy_frequency),
            'empathy_score': float(empathy_score),
            'nod_count': nod_count,
            'nod_frequency': float(nod_frequency),
            'understanding_score': float(understanding_score),
            'aligned_responses': aligned_responses,
            'alignment_score': float(alignment_score),
            'interviewer_tone': context.interviewer_tone,
            'interview_phase': context.interview_phase,
            'inappropriate_penalty': float(inappropriate_penalty),
            'total_frames': len(facial_data_list)
        }
        
        return contextual_score, supporting_data

    def analyze_emotion_consistency(
        self,
        facial_data_list: List[FacialData]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze emotion consistency analyzing temporal stability.
        
        Emotion consistency is evaluated based on:
        - Emotional stability over time (no sudden unexplained shifts)
        - Smooth emotional transitions
        - Consistent baseline emotional state
        - Absence of erratic emotional changes
        
        Args:
            facial_data_list: List of FacialData objects from video
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Emotion consistency score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - 11.1: Analyze emotional stability and consistency over time
            - 11.3: Reduce score when sudden or unexplained emotional shifts are detected
            - 11.4: Maintain higher score when emotional transitions are smooth and contextually justified
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # 1. Build emotion timeline
        emotion_timeline = []
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_intensity = dominant_emotion
            
            emotion_state = EmotionState(
                frame_number=facial_data.frame_number,
                timestamp=facial_data.timestamp,
                emotion=emotion_name,
                intensity=emotion_intensity,
                emotion_probabilities=emotions
            )
            emotion_timeline.append(emotion_state)
        
        # 2. Analyze emotional stability (frequency of changes)
        emotion_changes = 0
        sudden_changes = 0
        
        for i in range(1, len(emotion_timeline)):
            prev_state = emotion_timeline[i-1]
            curr_state = emotion_timeline[i]
            
            # Count emotion changes
            if prev_state.emotion != curr_state.emotion:
                emotion_changes += 1
                
                # Check if change is sudden (large intensity difference)
                intensity_diff = abs(curr_state.intensity - prev_state.intensity)
                if intensity_diff > 0.4:
                    sudden_changes += 1
        
        # Calculate change rate
        change_rate = emotion_changes / len(emotion_timeline) if len(emotion_timeline) > 1 else 0.0
        sudden_change_rate = sudden_changes / len(emotion_timeline) if len(emotion_timeline) > 1 else 0.0
        
        # Stability score: lower change rate = more stable
        # Optimal change rate: 0-10%
        if change_rate <= 0.10:
            stability_score = 1.0
        else:
            stability_score = max(0.3, 1.0 - (change_rate - 0.10) / 0.30)
        
        # Penalty for sudden changes
        sudden_penalty = sudden_change_rate * 0.5
        
        # 3. Analyze smoothness of transitions
        intensity_changes = []
        for i in range(1, len(emotion_timeline)):
            prev_intensity = emotion_timeline[i-1].intensity
            curr_intensity = emotion_timeline[i].intensity
            intensity_change = abs(curr_intensity - prev_intensity)
            intensity_changes.append(intensity_change)
        
        # Calculate average intensity change
        avg_intensity_change = np.mean(intensity_changes) if intensity_changes else 0.0
        
        # Smooth transitions: gradual changes (0.02-0.10)
        if avg_intensity_change <= 0.10:
            smoothness_score = 1.0
        else:
            smoothness_score = max(0.4, 1.0 - (avg_intensity_change - 0.10) / 0.20)
        
        # 4. Analyze baseline emotional state consistency
        # Find most common emotion
        emotion_counts = {}
        for state in emotion_timeline:
            emotion_counts[state.emotion] = emotion_counts.get(state.emotion, 0) + 1
        
        # Get dominant baseline emotion
        baseline_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        baseline_frequency = emotion_counts[baseline_emotion] / len(emotion_timeline)
        
        # Good baseline: 40-70% of time in one emotion
        if baseline_frequency < 0.40:
            baseline_score = baseline_frequency / 0.40  # Too varied
        elif baseline_frequency <= 0.70:
            baseline_score = 1.0  # Optimal
        else:
            baseline_score = max(0.6, 1.0 - (baseline_frequency - 0.70) / 0.30)  # Too monotonous
        
        # 5. Analyze temporal variance (consistency across time segments)
        # Divide timeline into segments and check consistency
        segment_size = max(10, len(emotion_timeline) // 5)  # 5 segments
        segment_emotions = []
        
        for i in range(0, len(emotion_timeline), segment_size):
            segment = emotion_timeline[i:i+segment_size]
            if segment:
                # Get most common emotion in segment
                seg_emotion_counts = {}
                for state in segment:
                    seg_emotion_counts[state.emotion] = seg_emotion_counts.get(state.emotion, 0) + 1
                
                seg_dominant = max(seg_emotion_counts.items(), key=lambda x: x[1])[0]
                segment_emotions.append(seg_dominant)
        
        # Calculate segment consistency
        if len(segment_emotions) > 1:
            # Count how many segments match the baseline
            matching_segments = sum(1 for e in segment_emotions if e == baseline_emotion)
            segment_consistency = matching_segments / len(segment_emotions)
        else:
            segment_consistency = 1.0
        
        # 6. Analyze emotional variance (standard deviation of intensities)
        intensities = [state.intensity for state in emotion_timeline]
        intensity_variance = np.var(intensities)
        
        # Low variance = consistent intensity
        # Optimal variance: 0.01-0.05
        if intensity_variance <= 0.05:
            variance_score = 1.0
        else:
            variance_score = max(0.5, 1.0 - (intensity_variance - 0.05) / 0.10)
        
        # 7. Combined emotion consistency score
        # Weight: stability (30%), smoothness (25%), baseline (20%), segment consistency (15%), variance (10%)
        base_score = (
            stability_score * 0.30 +
            smoothness_score * 0.25 +
            baseline_score * 0.20 +
            segment_consistency * 0.15 +
            variance_score * 0.10
        )
        
        # Apply sudden change penalty
        consistency_score = max(0.0, base_score - sudden_penalty)
        
        # Supporting data
        supporting_data = {
            'emotion_changes': emotion_changes,
            'sudden_changes': sudden_changes,
            'change_rate': float(change_rate),
            'sudden_change_rate': float(sudden_change_rate),
            'stability_score': float(stability_score),
            'sudden_penalty': float(sudden_penalty),
            'avg_intensity_change': float(avg_intensity_change),
            'smoothness_score': float(smoothness_score),
            'baseline_emotion': baseline_emotion,
            'baseline_frequency': float(baseline_frequency),
            'baseline_score': float(baseline_score),
            'num_segments': len(segment_emotions),
            'segment_consistency': float(segment_consistency),
            'intensity_variance': float(intensity_variance),
            'variance_score': float(variance_score),
            'total_frames': len(emotion_timeline)
        }
        
        return consistency_score, supporting_data

    def analyze_sentiment_alignment(
        self,
        facial_data_list: List[FacialData],
        context: Optional[ConversationContext] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze sentiment alignment evaluating empathy and alignment.
        
        Sentiment alignment is evaluated based on:
        - Emotional responses aligned with interviewer tone
        - Empathy and emotional intelligence
        - Appropriate emotional mirroring
        - Synchronization with conversation sentiment
        
        Args:
            facial_data_list: List of FacialData objects from video
            context: Optional conversation context
        
        Returns:
            Tuple of (score, supporting_data):
            - score: Sentiment alignment score (0-1)
            - supporting_data: Dictionary with analysis details
        
        Requirements:
            - 12.1: Evaluate how well candidate's emotions align with interviewer's tone and context
            - 12.3: Reduce score when emotional responses are misaligned with conversation
            - 12.4: Increase score when candidate shows empathy and emotional intelligence
        """
        if len(facial_data_list) == 0:
            return 0.0, {'error': 'No facial data available'}
        
        # Use default context if not provided
        if context is None:
            context = ConversationContext.default()
        
        # 1. Analyze alignment with interviewer tone
        aligned_count = 0
        misaligned_count = 0
        
        # Define expected emotions based on interviewer tone
        if context.interviewer_tone == 'formal':
            expected_emotions = {'neutral', 'happy'}
            misaligned_emotions = {'angry', 'disgust', 'fear', 'sad'}
        elif context.interviewer_tone == 'casual':
            expected_emotions = {'happy', 'neutral', 'surprise'}
            misaligned_emotions = {'angry', 'disgust', 'fear'}
        else:  # friendly
            expected_emotions = {'happy', 'surprise'}
            misaligned_emotions = {'angry', 'disgust', 'fear', 'sad'}
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])
            emotion_name, emotion_intensity = dominant_emotion
            
            if emotion_name in expected_emotions:
                aligned_count += 1
            elif emotion_name in misaligned_emotions and emotion_intensity > 0.5:
                misaligned_count += 1
        
        # Calculate alignment frequency
        alignment_frequency = aligned_count / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        misalignment_frequency = misaligned_count / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        
        # Tone alignment score
        tone_alignment_score = alignment_frequency
        
        # 2. Analyze empathy indicators (positive emotional engagement)
        empathy_indicators = 0
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            
            # Empathy: happy, surprise (showing interest and engagement)
            happy_prob = emotions.get('happy', 0.0)
            surprise_prob = emotions.get('surprise', 0.0)
            
            if happy_prob > 0.3 or surprise_prob > 0.3:
                empathy_indicators += 1
        
        empathy_frequency = empathy_indicators / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        
        # Good empathy: 25-65% of frames
        if empathy_frequency < 0.25:
            empathy_score = empathy_frequency / 0.25  # Too low
        elif empathy_frequency <= 0.65:
            empathy_score = 1.0  # Optimal
        else:
            empathy_score = max(0.7, 1.0 - (empathy_frequency - 0.65) / 0.35)  # Too much
        
        # 3. Analyze emotional intelligence (appropriate emotional responses)
        # Track emotional responses to different question types
        intelligent_responses = 0
        total_questions = 0
        
        if context.question_timestamps and context.question_types:
            for q_timestamp, q_type in zip(context.question_timestamps, context.question_types):
                total_questions += 1
                
                # Find frames within 5 seconds after question
                response_window_start = q_timestamp
                response_window_end = q_timestamp + 5.0
                
                # Get frames in this window
                window_frames = [
                    fd for fd in facial_data_list
                    if response_window_start <= fd.timestamp <= response_window_end
                ]
                
                if window_frames:
                    # Get dominant emotions in window
                    window_emotions = []
                    for fd in window_frames:
                        emotions = fd.emotion_probabilities
                        dominant = max(emotions.items(), key=lambda x: x[1])[0]
                        window_emotions.append(dominant)
                    
                    # Check if emotions are appropriate for question type
                    expected = context.expected_emotions.get(q_type, ['neutral', 'happy'])
                    
                    # Count matching emotions
                    matching_count = sum(1 for e in window_emotions if e in expected)
                    
                    # If majority match, consider it intelligent
                    if matching_count / len(window_emotions) > 0.5:
                        intelligent_responses += 1
        
        # Calculate emotional intelligence score
        if total_questions > 0:
            ei_score = intelligent_responses / total_questions
        else:
            ei_score = 0.7  # Default neutral score
        
        # 4. Analyze emotional mirroring (subtle matching of positive emotions)
        # In interviews, candidates should show some positive mirroring
        positive_emotion_count = 0
        
        for facial_data in facial_data_list:
            emotions = facial_data.emotion_probabilities
            
            # Positive emotions that show engagement
            if emotions.get('happy', 0.0) > 0.4:
                positive_emotion_count += 1
        
        positive_frequency = positive_emotion_count / len(facial_data_list) if len(facial_data_list) > 0 else 0.0
        
        # Good mirroring: 20-50% positive emotions
        if positive_frequency < 0.20:
            mirroring_score = positive_frequency / 0.20  # Too low
        elif positive_frequency <= 0.50:
            mirroring_score = 1.0  # Optimal
        else:
            mirroring_score = max(0.7, 1.0 - (positive_frequency - 0.50) / 0.50)  # Too much
        
        # 5. Analyze sentiment synchronization (emotional flow matches conversation)
        # Calculate emotional variance in segments
        segment_size = max(10, len(facial_data_list) // 5)
        segment_variances = []
        
        for i in range(0, len(facial_data_list), segment_size):
            segment = facial_data_list[i:i+segment_size]
            if len(segment) > 1:
                # Calculate emotion intensity variance in segment
                intensities = []
                for fd in segment:
                    emotions = fd.emotion_probabilities
                    max_intensity = max(emotions.values())
                    intensities.append(max_intensity)
                
                segment_variance = np.var(intensities)
                segment_variances.append(segment_variance)
        
        # Calculate average segment variance
        avg_segment_variance = np.mean(segment_variances) if segment_variances else 0.0
        
        # Moderate variance = good synchronization (not flat, not chaotic)
        # Optimal variance: 0.02-0.08
        if avg_segment_variance < 0.02:
            sync_score = avg_segment_variance / 0.02  # Too flat
        elif avg_segment_variance <= 0.08:
            sync_score = 1.0  # Optimal
        else:
            sync_score = max(0.5, 1.0 - (avg_segment_variance - 0.08) / 0.12)  # Too chaotic
        
        # 6. Combined sentiment alignment score
        # Weight: tone alignment (30%), empathy (25%), EI (20%), mirroring (15%), sync (10%)
        base_score = (
            tone_alignment_score * 0.30 +
            empathy_score * 0.25 +
            ei_score * 0.20 +
            mirroring_score * 0.15 +
            sync_score * 0.10
        )
        
        # Penalty for misalignment
        misalignment_penalty = misalignment_frequency * 0.3
        sentiment_alignment_score = max(0.0, base_score - misalignment_penalty)
        
        # Supporting data
        supporting_data = {
            'aligned_count': aligned_count,
            'misaligned_count': misaligned_count,
            'alignment_frequency': float(alignment_frequency),
            'misalignment_frequency': float(misalignment_frequency),
            'tone_alignment_score': float(tone_alignment_score),
            'empathy_indicators': empathy_indicators,
            'empathy_frequency': float(empathy_frequency),
            'empathy_score': float(empathy_score),
            'intelligent_responses': intelligent_responses,
            'total_questions': total_questions,
            'ei_score': float(ei_score),
            'positive_emotion_count': positive_emotion_count,
            'positive_frequency': float(positive_frequency),
            'mirroring_score': float(mirroring_score),
            'avg_segment_variance': float(avg_segment_variance),
            'sync_score': float(sync_score),
            'interviewer_tone': context.interviewer_tone,
            'misalignment_penalty': float(misalignment_penalty),
            'total_frames': len(facial_data_list)
        }
        
        return sentiment_alignment_score, supporting_data
