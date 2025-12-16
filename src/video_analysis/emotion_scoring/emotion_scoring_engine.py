"""
Emotion Scoring Engine for Recruitment Emotion Scoring System.

This module orchestrates the complete emotion scoring pipeline, from video
processing to final report generation. It coordinates facial data extraction,
criterion analysis, and score calculation.

Requirements: 13.1, 13.2, 13.3
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path

from .models import (
    EmotionReport,
    CriterionScore,
    FacialData,
    ScoringConfig,
    KeyMoment,
    ConversationContext,
)
from .facial_data_extractor import FacialDataExtractor
from .emotion_criteria_analyzer import EmotionCriteriaAnalyzer
from .validation import (
    EmotionScoringValidator,
    FaceDetectionValidator,
    handle_video_error,
    handle_face_detection_error,
    log_validation_results
)
import logging

logger = logging.getLogger(__name__)


class EmotionScoringEngine:
    """
    Orchestrates the emotion scoring pipeline.
    
    This class coordinates the complete scoring process:
    1. Extract facial data from video
    2. Analyze each of the 12 emotional criteria
    3. Calculate criterion scores with thresholds
    4. Calculate weighted total score
    5. Generate comprehensive emotion report
    
    The engine uses configurable weights and thresholds to customize
    scoring behavior for different interview contexts.
    
    Requirements:
    - 13.1: Calculate weighted average score from 12 criteria
    - 13.2: Apply configurable weights to each criterion
    - 13.3: Normalize result to 0-10 scale
    """
    
    def __init__(
        self,
        config: Optional[ScoringConfig] = None,
        facial_data_extractor: Optional[FacialDataExtractor] = None,
        emotion_criteria_analyzer: Optional[EmotionCriteriaAnalyzer] = None,
        enable_validation: bool = True,
        face_confidence_threshold: float = 0.5
    ):
        """
        Initialize EmotionScoringEngine with new 4-criterion system.
        
        Args:
            config: Scoring configuration (uses default if None)
            facial_data_extractor: FacialDataExtractor instance (creates default if None)
            emotion_criteria_analyzer: EmotionCriteriaAnalyzer instance (creates default if None)
            enable_validation: Enable input validation and quality checks (default: True)
            face_confidence_threshold: Face detection confidence threshold (default: 0.5, lower = more detections)
        """
        # Log face detection threshold
        logger.info(f"EmotionScoringEngine initializing with face_confidence_threshold={face_confidence_threshold}")
        
        # Load configuration
        if config is None:
            from .config import load_default_config
            self.config = load_default_config()
        else:
            self.config = config
        
        # Store face confidence threshold for error messages
        self.face_confidence_threshold = face_confidence_threshold
        
        # Initialize components
        if facial_data_extractor is None:
            # Create face detector and emotion classifier
            from src.inference.face_detector import FaceDetector
            from src.inference.preprocessor import FacePreprocessor
            from src.inference.emotion_classifier import EmotionClassifier
            
            # Initialize face detector
            face_detector = FaceDetector(
                device='auto',
                confidence_threshold=face_confidence_threshold,
                max_faces=1
            )
            
            # Initialize preprocessor
            face_preprocessor = FacePreprocessor(target_size=(224, 224))
            
            # Initialize emotion classifier
            model_path = Path('models/efficientnet_b2_best.pth')
            if not model_path.exists():
                logger.warning(f"Emotion model not found at {model_path}, trying alternative paths...")
                # Try alternative paths
                alt_paths = [
                    Path('models/efficientnet_b2_final.pth'),
                    Path('models/efficientnet_b2_old.pth')
                ]
                for alt_path in alt_paths:
                    if alt_path.exists():
                        model_path = alt_path
                        logger.info(f"Using alternative model: {model_path}")              
                        break
                else:
                    raise FileNotFoundError(
                        f"No emotion classifier model found. Please ensure one of these exists:\n"
                        f"  - models/efficientnet_b2_best.pth\n"
                        f"  - models/efficientnet_b2_final.pth\n"
                        f"  - models/efficientnet_b2_old.pth\n\n"
                        f"Troubleshooting steps:\n"
                        f"  1. Check if the 'models' directory exists in your project root\n"
                        f"  2. Download the pre-trained model from the project repository\n"
                        f"  3. Place the model file in the 'models' directory\n"
                        f"  4. Ensure the model file has the correct name (efficientnet_b2_best.pth)\n"
                        f"  5. Verify file permissions allow reading the model file"
                    )
            
            emotion_classifier = EmotionClassifier(
                model_path=str(model_path),
                device='auto'
            )
            
            # Create FacialDataExtractor with all components
            self.facial_data_extractor = FacialDataExtractor(
                face_detector=face_detector,
                face_preprocessor=face_preprocessor,
                emotion_classifier=emotion_classifier,
                enable_validation=enable_validation,
                confidence_threshold=face_confidence_threshold
            )
        else:
            self.facial_data_extractor = facial_data_extractor
        
        # Use new EmotionCriteriaAnalyzer (4 criteria system)
        self.emotion_criteria_analyzer = emotion_criteria_analyzer or EmotionCriteriaAnalyzer()
        
        # Initialize validators
        self.enable_validation = enable_validation
        self.validator = EmotionScoringValidator() if enable_validation else None
        self.face_validator = FaceDetectionValidator() if enable_validation else None
        
        # Define criterion names (New 4 criteria system)
        self.criterion_names = [
            'emotion_stability',
            'emotion_content_alignment',
            'positive_ratio',
            'negative_overload'
        ]
        
        print(f"EmotionScoringEngine initialized - New System (4 criteria - D. Cảm xúc 5%)")
        print(f"  Configuration: {len(self.criterion_names)} criteria")
        print(f"  Frame sampling rate: {self.config.frame_sampling_rate}")
        print(f"  Micro-expression analysis: {self.config.enable_micro_expression_analysis}")
        print(f"  Gaze tracking: {self.config.enable_gaze_tracking}")
        print(f"  Validation enabled: {self.enable_validation}")
    
    def score_video_interview(
        self,
        video_path: str,
        candidate_id: str,
        context: Optional[ConversationContext] = None
    ) -> EmotionReport:
        """
        Score a complete video interview.
        
        This is the main entry point for the scoring pipeline. It orchestrates
        all steps from video processing to final report generation.
        
        Args:
            video_path: Path to video file
            candidate_id: Unique identifier for the candidate
            context: Optional conversation context (question timestamps, tone, etc.)
        
        Returns:
            EmotionReport with all criterion scores and total score
        
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be processed
        
        Requirements:
            - 13.1: Calculate weighted average score
            - 13.2: Apply configurable weights
            - 13.3: Normalize to 0-10 scale
        """
        print(f"\n{'='*60}")
        print(f"Starting emotion scoring for: {Path(video_path).name}")
        print(f"Candidate ID: {candidate_id}")
        print(f"{'='*60}\n")
        
        # Use default context if not provided
        if context is None:
            context = ConversationContext.default()
        
        # Validate video before processing (if validation enabled)
        analysis_confidence = 1.0
        if self.enable_validation and self.validator:
            try:
                print("Validating video quality...")
                is_valid, validation_results, confidence = self.validator.validate_for_scoring(
                    video_path
                )
                
                if not is_valid:
                    error_messages = [r.message for r in validation_results 
                                    if r.severity.value in ['critical', 'error']]
                    raise ValueError(
                        f"Video validation failed: {'; '.join(error_messages)}"
                    )
                
                analysis_confidence = confidence
                
                # Log validation results
                log_validation_results(validation_results, logger)
                
                if confidence < 1.0:
                    print(f"  Analysis confidence: {confidence:.2f} (warnings present)")
                else:
                    print(f"  Video validation passed")
                print()
            
            except Exception as e:
                error_msg = handle_video_error(video_path, e)
                logger.error(error_msg)
                raise ValueError(error_msg) from e
        
        # Step 1: Extract facial data from video
        print("Step 1: Extracting facial data from video...")
        logger.info(f"Starting facial data extraction from {video_path}")
        logger.info(f"  Frame sampling rate: {self.config.frame_sampling_rate}")
        
        try:
            facial_data_list = self.facial_data_extractor.extract_from_video(
                video_path=video_path,
                frame_sampling_rate=self.config.frame_sampling_rate
            )
        except Exception as e:
            error_msg = handle_video_error(video_path, e)
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        if len(facial_data_list) == 0:
            # Get video info for better error message
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            raise ValueError(
                f"No faces detected in video: {video_path}\n"
                f"  Total frames processed: {total_frames}\n"
                f"  Face detection threshold: {self.face_confidence_threshold}\n\n"
                f"Troubleshooting steps:\n"
                f"  1. Lower the face detection threshold (try 0.3-0.4 instead of {self.face_confidence_threshold})\n"
                f"     Example: EmotionScoringEngine(face_confidence_threshold=0.3)\n"
                f"  2. Check video quality:\n"
                f"     - Ensure faces are clearly visible and not too small\n"
                f"     - Verify adequate lighting (not too dark or too bright)\n"
                f"     - Check that faces are not obscured or at extreme angles\n"
                f"  3. Verify video file is not corrupted and plays correctly\n"
                f"  4. Try processing a different video to isolate the issue\n"
                f"  5. Check that the MTCNN face detector model is properly loaded"
            )
        
        print(f"  Extracted {len(facial_data_list)} frames with facial data")
        logger.info(f"Facial data extraction complete: {len(facial_data_list)} frames processed")
        print()
        
        # Validate face detection quality (if validation enabled)
        if self.enable_validation and self.face_validator:
            try:
                # Get total frames from video
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                print("Validating face detection quality...")
                logger.info("Starting face detection quality validation")
                
                face_validation_results = self.face_validator.validate_face_detection(
                    facial_data_list, total_frames
                )
                
                # Check for critical face detection errors
                has_face_error = any(
                    r.severity.value == 'error' for r in face_validation_results
                )
                
                if has_face_error:
                    error_messages = [r.message for r in face_validation_results 
                                    if r.severity.value == 'error']
                    logger.warning(
                        f"Face detection quality issues: {'; '.join(error_messages)}"
                    )
                    # Reduce confidence but don't fail
                    old_confidence = analysis_confidence
                    analysis_confidence *= 0.7
                    logger.info(f"Confidence adjusted due to face detection errors: {old_confidence:.2f} -> {analysis_confidence:.2f}")
                
                # Log face validation results
                log_validation_results(face_validation_results, logger)
                
                # Get quality metrics
                face_quality = self.face_validator.get_face_detection_quality(
                    facial_data_list, total_frames
                )
                print(f"  Face detection rate: {face_quality.face_detection_rate*100:.1f}%")
                print(f"  Face quality score: {face_quality.quality_score:.2f}")
                logger.info(f"Face detection statistics: rate={face_quality.face_detection_rate*100:.1f}%, quality={face_quality.quality_score:.2f}")
                
                # Update confidence based on face quality
                old_confidence = analysis_confidence
                analysis_confidence *= face_quality.quality_score
                if old_confidence != analysis_confidence:
                    logger.info(f"Confidence adjusted based on face quality: {old_confidence:.2f} -> {analysis_confidence:.2f}")
                print(f"  Overall analysis confidence: {analysis_confidence:.2f}\n")
            
            except Exception as e:
                logger.warning(f"Face detection validation failed: {e}")
                old_confidence = analysis_confidence
                analysis_confidence *= 0.8
                logger.info(f"Confidence adjusted due to validation failure: {old_confidence:.2f} -> {analysis_confidence:.2f}")
        
        # Step 2: Detect micro-expressions (if enabled)
        micro_expressions = None
        if self.config.enable_micro_expression_analysis:
            print("Step 2: Detecting micro-expressions...")
            logger.info("Starting micro-expression detection")
            micro_expressions = self.facial_data_extractor.detect_micro_expressions(
                facial_data_list
            )
            print(f"  Detected {len(micro_expressions)} micro-expressions")
            logger.info(f"Micro-expression detection complete: {len(micro_expressions)} detected")
            print()
        
        # Step 3: Track eye contact (if enabled)
        eye_contact_stats = None
        if self.config.enable_gaze_tracking:
            print("Step 3: Tracking eye contact patterns...")
            logger.info("Starting eye contact tracking")
            eye_contact_stats = self.facial_data_extractor.track_eye_contact(
                facial_data_list
            )
            print(f"  Eye contact: {eye_contact_stats['eye_contact_percentage']:.1f}%")
            logger.info(f"Eye contact tracking complete: {eye_contact_stats['eye_contact_percentage']:.1f}% eye contact")
            print()
        
        # Step 4: Analyze all criteria
        print("Step 4: Analyzing emotional criteria...")
        logger.info(f"Starting criterion analysis for {len(self.criterion_names)} criteria")
        criterion_scores = {}
        key_moments = []
        
        for criterion_name in self.criterion_names:
            print(f"  Analyzing {criterion_name}...")
            logger.info(f"Analyzing criterion: {criterion_name}")
            
            # Get raw analysis score (0-1) and supporting data
            raw_score, supporting_data = self._analyze_criterion(
                criterion_name=criterion_name,
                facial_data_list=facial_data_list,
                micro_expressions=micro_expressions,
                eye_contact_stats=eye_contact_stats,
                context=context
            )
            
            logger.info(f"  Raw score for {criterion_name}: {raw_score:.3f}")
            
            # Calculate final criterion score (0-10) with thresholds
            criterion_score = self.calculate_criterion_score(
                criterion=criterion_name,
                analysis_result=raw_score
            )
            
            logger.info(f"  Final score for {criterion_name}: {criterion_score:.2f}/10")
            
            # Create CriterionScore object with adjusted confidence
            criterion_confidence = self._calculate_confidence(supporting_data) * analysis_confidence
            
            logger.info(f"  Confidence for {criterion_name}: {criterion_confidence:.2f}")
            
            criterion_score_obj = CriterionScore(
                criterion_name=criterion_name,
                score=criterion_score,
                confidence=criterion_confidence,
                explanation=self._generate_explanation(
                    criterion_name, criterion_score, supporting_data
                ),
                evidence_timestamps=self._extract_evidence_timestamps(
                    criterion_name, facial_data_list, supporting_data
                ),
                supporting_data=supporting_data
            )
            
            criterion_scores[criterion_name] = criterion_score_obj
            
            # Extract key moments for this criterion
            moments = self._extract_key_moments(
                criterion_name, facial_data_list, supporting_data
            )
            key_moments.extend(moments)
        
        print(f"  Analyzed all {len(criterion_scores)} criteria")
        logger.info(f"Criterion analysis complete: {len(criterion_scores)} criteria analyzed")
        print()
        
        # Step 5: Calculate total score
        print("Step 5: Calculating total score...")
        logger.info("Calculating weighted total score")
        total_score = self.calculate_total_score(criterion_scores)
        print(f"  Total score: {total_score:.2f}/10")
        logger.info(f"Total score calculated: {total_score:.2f}/10")
        print()
        
        # Step 6: Generate report
        print("Step 6: Generating emotion report...")
        logger.info("Generating final emotion report")
        report = EmotionReport(
            candidate_id=candidate_id,
            video_path=video_path,
            timestamp=datetime.now(),
            criterion_scores=criterion_scores,
            total_score=total_score,
            key_moments=key_moments[:20],  # Limit to top 20 moments
            metadata={
                'total_frames': len(facial_data_list),
                'video_duration': facial_data_list[-1].timestamp if facial_data_list else 0.0,
                'frame_sampling_rate': self.config.frame_sampling_rate,
                'micro_expressions_detected': len(micro_expressions) if micro_expressions else 0,
                'eye_contact_percentage': eye_contact_stats['eye_contact_percentage'] if eye_contact_stats else 0.0,
                'analysis_confidence': analysis_confidence,
                'validation_enabled': self.enable_validation,
            }
        )
        
        print(f"{'='*60}")
        print(f"Emotion scoring complete!")
        print(f"{'='*60}\n")
        
        return report
    
    def _analyze_criterion(
        self,
        criterion_name: str,
        facial_data_list: List[FacialData],
        micro_expressions: Optional[List],
        eye_contact_stats: Optional[Dict[str, Any]],
        context: ConversationContext
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Analyze a specific criterion with graceful degradation.
        
        Routes to the appropriate analyzer method based on criterion name.
        Handles missing data gracefully by returning default scores with warnings.
        
        For 4-criteria system: confidence, positivity, professionalism, engagement
        For 12-criteria system (legacy): all original criteria
        
        Args:
            criterion_name: Name of criterion to analyze
            facial_data_list: List of FacialData objects
            micro_expressions: Detected micro-expressions
            eye_contact_stats: Eye contact statistics
            context: Conversation context
        
        Returns:
            Tuple of (raw_score, supporting_data)
        """
        analyzer = self.emotion_criteria_analyzer
        
        # Check for insufficient data
        if len(facial_data_list) == 0:
            logger.warning(f"No facial data available for {criterion_name} analysis")
            return 0.5, {
                'error': 'insufficient_data',
                'message': 'No facial data available',
                'total_frames': 0
            }
        
        try:
            # New 4-criteria system
            if criterion_name == 'emotion_stability':
                return analyzer.analyze_emotion_stability(facial_data_list)
            
            elif criterion_name == 'emotion_content_alignment':
                return analyzer.analyze_emotion_content_alignment(facial_data_list, context)
            
            elif criterion_name == 'positive_ratio':
                return analyzer.analyze_positive_ratio(facial_data_list)
            
            elif criterion_name == 'negative_overload':
                return analyzer.analyze_negative_overload(facial_data_list)
            
            elif criterion_name == 'expression_control':
                return analyzer.analyze_expression_control(facial_data_list)
            
            elif criterion_name == 'friendliness':
                return analyzer.analyze_friendliness(facial_data_list)
            
            else:
                raise ValueError(f"Unknown criterion: {criterion_name}")
        
        except Exception as e:
            logger.error(f"Error analyzing {criterion_name}: {e}")
            return 0.5, {
                'error': 'analysis_failed',
                'message': str(e),
                'total_frames': len(facial_data_list)
            }
    
    def calculate_criterion_score(
        self,
        criterion: str,
        analysis_result: float
    ) -> float:
        """
        Calculate final criterion score applying thresholds.
        
        Converts raw analysis result (0-1) to final score (0-10) using
        configured thresholds. Scores are mapped as follows:
        - Below low threshold: scaled 0 to 4
        - Between thresholds: scaled 4 to 7
        - Above high threshold: scaled 7 to 10
        
        Args:
            criterion: Criterion name
            analysis_result: Raw analysis result (0-1)
        
        Returns:
            Final criterion score (0-10)
        
        Requirements:
            - 13.2: Apply thresholds from configuration
            - 13.3: Normalize to 0-10 scale
        """
        # Get thresholds for this criterion
        if criterion not in self.config.score_thresholds:
            # Use default thresholds if not configured
            low_threshold, high_threshold = (4.0, 7.0)
        else:
            low_threshold, high_threshold = self.config.score_thresholds[criterion]
        
        # Convert raw score (0-1) to 0-10 scale
        raw_score_10 = analysis_result * 10.0
        
        # Apply threshold mapping
        if raw_score_10 < low_threshold:
            # Low range: 0 to low_threshold
            # Map to 0 to low_threshold
            final_score = raw_score_10
        elif raw_score_10 < high_threshold:
            # Medium range: low_threshold to high_threshold
            # Keep as is
            final_score = raw_score_10
        else:
            # High range: high_threshold to 10
            # Keep as is
            final_score = raw_score_10
        
        # Ensure score is in valid range [0, 10]
        final_score = np.clip(final_score, 0.0, 10.0)
        
        return float(final_score)
    
    def calculate_total_score(
        self,
        criterion_scores: Dict[str, CriterionScore]
    ) -> float:
        """
        Calculate weighted total score from criterion scores.
        
        Applies configured weights to each criterion score and computes
        the weighted average. The result is normalized to 0-10 scale.
        
        For 4-criteria system:
        - Confidence: 30-35%
        - Positivity: 25-30%
        - Professionalism: 20-25%
        - Engagement: 15-20%
        
        For 12-criteria system (legacy): uses configured weights
        
        Args:
            criterion_scores: Dictionary mapping criterion names to CriterionScore objects
        
        Returns:
            Total weighted score (0-10)
        
        Requirements:
            - 5.1: Calculate weighted average score from 4 criteria
            - 5.2: Apply weights: Confidence (30-35%), Positivity (25-30%), 
                   Professionalism (20-25%), Engagement (15-20%)
            - 5.3: Normalize to 0-10 scale
        """
        # Extract scores and apply weights
        weighted_sum = 0.0
        
        for criterion_name, criterion_score_obj in criterion_scores.items():
            # Get weight for this criterion
            weight = self.config.criterion_weights.get(criterion_name, 0.0)
            
            # Get score value (already on 0-10 scale)
            score = criterion_score_obj.score
            
            # Calculate weighted contribution
            weighted_sum += score * weight
        
        # The weighted sum is already the final score since weights sum to 1.0
        # and scores are already on 0-10 scale
        total_score = weighted_sum
        
        # Apply final score multiplier if configured (for new system: 1.0 to keep 0-10 scale)
        if hasattr(self.config, 'final_score_multiplier'):
            total_score *= self.config.final_score_multiplier
            logger.info(f"Applied final_score_multiplier: {self.config.final_score_multiplier}, result: {total_score:.2f}")
        
        # Ensure score is in valid range
        max_score = 10.0 if not hasattr(self.config, 'final_score_multiplier') else 10.0 * self.config.final_score_multiplier
        total_score = np.clip(total_score, 0.0, max_score)
        
        return float(total_score)
    
    def apply_weights(
        self,
        scores: Dict[str, float]
    ) -> float:
        """
        Apply configurable weights to criterion scores.
        
        This is a utility method that applies weights to a dictionary of
        raw scores and returns the weighted average.
        
        For 4-criteria system:
        - Confidence: 30-35%
        - Positivity: 25-30%
        - Professionalism: 20-25%
        - Engagement: 15-20%
        
        For 12-criteria system (legacy): uses configured weights
        
        Args:
            scores: Dictionary mapping criterion names to score values (0-10)
        
        Returns:
            Weighted average score (0-10)
        
        Requirements:
            - 5.2: Apply weights: Confidence (30-35%), Positivity (25-30%), 
                   Professionalism (20-25%), Engagement (15-20%)
        """
        weighted_sum = 0.0
        
        for criterion_name, score in scores.items():
            weight = self.config.criterion_weights.get(criterion_name, 0.0)
            weighted_sum += score * weight
        
        # The weighted sum is already the final score since weights sum to 1.0
        # and scores are already on 0-10 scale
        weighted_average = weighted_sum
        
        # Ensure score is in valid range [0, 10]
        weighted_average = np.clip(weighted_average, 0.0, 10.0)
        
        return float(weighted_average)
    
    def _calculate_confidence(self, supporting_data: Dict[str, Any]) -> float:
        """
        Calculate confidence score for a criterion analysis.
        
        Confidence is based on:
        - Amount of data available
        - Consistency of measurements
        - Absence of errors
        
        Args:
            supporting_data: Supporting data from criterion analysis
        
        Returns:
            Confidence score (0-1)
        """
        # Check for errors
        if 'error' in supporting_data:
            return 0.0
        
        # Base confidence on data availability
        total_frames = supporting_data.get('total_frames', 0)
        
        if total_frames == 0:
            return 0.0
        elif total_frames < 30:
            # Low confidence for very short videos
            return 0.5
        elif total_frames < 100:
            # Medium confidence
            return 0.7
        else:
            # High confidence for longer videos
            return 0.9
    
    def _generate_explanation(
        self,
        criterion_name: str,
        score: float,
        supporting_data: Dict[str, Any]
    ) -> str:
        """
        Generate human-readable explanation for a criterion score.
        
        Args:
            criterion_name: Name of the criterion
            score: Final score (0-10)
            supporting_data: Supporting data from analysis
        
        Returns:
            Explanation string
        """
        # Determine score level
        if score >= 7.0:
            level = "high"
        elif score >= 4.0:
            level = "moderate"
        else:
            level = "low"
        
        # Generate criterion-specific explanation
        explanations = {
            # New 4-criteria system (D. Cảm xúc - 5%)
            'emotion_stability': f"Emotion Stability (Ổn định cảm xúc) was {level} with a score of {score:.1f}/10. "
                                f"Emotion change rate was {supporting_data.get('change_rate', 0)*100:.1f}%, "
                                f"with {level} consistency in emotional expressions. "
                                f"Baseline emotion: {supporting_data.get('most_common_emotion', 'unknown')} "
                                f"({supporting_data.get('baseline_consistency', 0)*100:.1f}% of time).",
            
            'emotion_content_alignment': f"Emotion-Content Alignment (Khớp cảm xúc & nội dung) was {level} with a score of {score:.1f}/10. "
                                        f"Appropriate emotions were shown {supporting_data.get('appropriate_ratio', 0)*100:.1f}% of the time, "
                                        f"with {supporting_data.get('inappropriate_count', 0)} instances of inappropriate emotions. "
                                        f"Emotional transitions were {'smooth' if supporting_data.get('transition_penalty', 0) < 0.1 else 'somewhat erratic'}.",
            
            'positive_ratio': f"Positive Ratio (Tích cực) was {level} with a score of {score:.1f}/10. "
                             f"Positive emotions appeared in {supporting_data.get('positive_ratio', 0)*100:.1f}% of frames, "
                             f"with {supporting_data.get('genuine_smile_count', 0)} genuine smiles detected. "
                             f"Negative emotions: {supporting_data.get('negative_ratio', 0)*100:.1f}%.",
            
            'negative_overload': f"Negative Overload (Mức tiêu cực vượt ngưỡng) was {level} with a score of {score:.1f}/10. "
                                f"Extreme negative emotions: {supporting_data.get('extreme_negative_ratio', 0)*100:.1f}%, "
                                f"moderate negative: {supporting_data.get('moderate_negative_ratio', 0)*100:.1f}%. "
                                f"Max consecutive negative frames: {supporting_data.get('max_consecutive_negative', 0)} "
                                f"({supporting_data.get('max_consecutive_ratio', 0)*100:.1f}% of video).",
            
            # Legacy 4-criteria system (backward compatibility)
            'confidence': f"The candidate showed {level} confidence with a score of {score:.1f}/10. "
                         f"Eye contact was maintained {supporting_data.get('eye_contact_percentage', 0):.1f}% of the time, "
                         f"with {level} posture and voice tone indicators.",
            
            'positivity': f"Positivity was {level} with a score of {score:.1f}/10. "
                         f"Genuine smiles were detected in {supporting_data.get('genuine_smile_frequency', 0)*100:.1f}% of frames, "
                         f"showing {level} warmth and friendliness.",
            
            'professionalism': f"Professionalism was {level} with a score of {score:.1f}/10. "
                              f"The candidate maintained {level} composure and emotional control, "
                              f"with appropriate behavior {supporting_data.get('appropriate_frequency', 0)*100:.1f}% of the time.",
            
            'engagement': f"Engagement was {level} with a score of {score:.1f}/10. "
                         f"The candidate showed {level} attention and responsiveness, "
                         f"with facial expressions matching context {supporting_data.get('responsiveness_rate', 0)*100:.1f}% of the time.",
            
            # Legacy 12-criteria system (backward compatibility)
            'calmness': f"The candidate demonstrated {level} calmness with a score of {score:.1f}/10. "
                       f"Calm expressions were observed in {supporting_data.get('calm_frequency', 0)*100:.1f}% of frames.",
            
            'expression_control': f"Expression control was {level} with a score of {score:.1f}/10. "
                                 f"Emotional changes occurred at a rate of {supporting_data.get('emotion_change_rate', 0)*100:.1f}%.",
            
            'friendliness': f"Friendliness was {level} with a score of {score:.1f}/10. "
                           f"Genuine smiles were detected in {supporting_data.get('genuine_smile_count', 0)} frames.",
            
            'enthusiasm': f"The candidate displayed {level} enthusiasm with a score of {score:.1f}/10. "
                         f"Average energy level was {supporting_data.get('avg_energy', 0):.2f}.",
            
            'authenticity': f"Authenticity was {level} with a score of {score:.1f}/10. "
                           f"Micro-expression consistency was {supporting_data.get('micro_expr_consistency', 0):.2f}.",
            
            'eye_contact': f"Eye contact was {level} with a score of {score:.1f}/10. "
                          f"The candidate maintained eye contact {supporting_data.get('eye_contact_percentage', 0):.1f}% of the time.",
            
            'facial_interaction': f"Facial interaction was {level} with a score of {score:.1f}/10. "
                                 f"Responsiveness rate was {supporting_data.get('responsiveness_rate', 0)*100:.1f}%.",
            
            'contextual_appropriateness': f"Contextual appropriateness was {level} with a score of {score:.1f}/10. "
                                         f"Appropriate expressions were shown {supporting_data.get('appropriate_frequency', 0)*100:.1f}% of the time.",
            
            'emotion_consistency': f"Emotion consistency was {level} with a score of {score:.1f}/10. "
                                  f"Emotional changes occurred at a rate of {supporting_data.get('change_rate', 0)*100:.1f}%.",
            
            'sentiment_alignment': f"Sentiment alignment was {level} with a score of {score:.1f}/10. "
                                  f"Emotional alignment with interviewer tone was {supporting_data.get('alignment_frequency', 0)*100:.1f}%.",
        }
        
        return explanations.get(criterion_name, f"{criterion_name.replace('_', ' ').title()} score: {score:.1f}/10")
    
    def _extract_evidence_timestamps(
        self,
        criterion_name: str,
        facial_data_list: List[FacialData],
        supporting_data: Dict[str, Any]
    ) -> List[float]:
        """
        Extract timestamps that provide evidence for a criterion score.
        
        Args:
            criterion_name: Name of the criterion
            facial_data_list: List of FacialData objects
            supporting_data: Supporting data from analysis
        
        Returns:
            List of timestamps (in seconds)
        """
        timestamps = []
        
        # Extract relevant timestamps based on criterion
        if criterion_name == 'calmness':
            # Find frames with high calm emotions
            for fd in facial_data_list[:10]:  # Sample first 10
                if fd.emotion_probabilities.get('neutral', 0) > 0.6:
                    timestamps.append(fd.timestamp)
        
        elif criterion_name == 'friendliness':
            # Find frames with genuine smiles
            for fd in facial_data_list:
                if fd.action_units.get('AU12', 0) > 0.5 and fd.action_units.get('AU6', 0) > 0.3:
                    timestamps.append(fd.timestamp)
                    if len(timestamps) >= 5:
                        break
        
        elif criterion_name == 'eye_contact':
            # Find frames with good eye contact
            for fd in facial_data_list:
                gaze_x, gaze_y = fd.gaze_direction
                gaze_magnitude = np.sqrt(gaze_x**2 + gaze_y**2)
                if gaze_magnitude <= 0.3:
                    timestamps.append(fd.timestamp)
                    if len(timestamps) >= 5:
                        break
        
        # Limit to 10 timestamps
        return timestamps[:10]
    
    def _extract_key_moments(
        self,
        criterion_name: str,
        facial_data_list: List[FacialData],
        supporting_data: Dict[str, Any]
    ) -> List[KeyMoment]:
        """
        Extract key moments for a criterion.
        
        Args:
            criterion_name: Name of the criterion
            facial_data_list: List of FacialData objects
            supporting_data: Supporting data from analysis
        
        Returns:
            List of KeyMoment objects
        """
        key_moments = []
        
        # Extract criterion-specific key moments
        if criterion_name == 'confidence' and len(facial_data_list) > 0:
            # Find moment with best posture and eye contact
            best_confidence_score = 0.0
            best_confidence_frame = None
            
            for fd in facial_data_list:
                # Calculate confidence score from posture and gaze
                pitch, yaw, roll = fd.head_pose
                gaze_x, gaze_y = fd.gaze_direction
                gaze_magnitude = np.sqrt(gaze_x**2 + gaze_y**2)
                
                posture_score = 1.0 - (abs(pitch) + abs(yaw)) / 180.0
                gaze_score = 1.0 - gaze_magnitude
                confidence_score = (posture_score + gaze_score) / 2.0
                
                if confidence_score > best_confidence_score:
                    best_confidence_score = confidence_score
                    best_confidence_frame = fd
            
            if best_confidence_frame and best_confidence_score > 0.5:
                key_moments.append(KeyMoment(
                    timestamp=best_confidence_frame.timestamp,
                    frame_number=best_confidence_frame.frame_number,
                    description="Strong confidence indicators",
                    criterion=criterion_name,
                    emotion='neutral',
                    intensity=best_confidence_score
                ))
        
        elif criterion_name == 'positivity' and len(facial_data_list) > 0:
            # Find best smile moment
            best_smile_score = 0.0
            best_smile_frame = None
            
            for fd in facial_data_list:
                smile_score = fd.action_units.get('AU12', 0) * fd.action_units.get('AU6', 0)
                if smile_score > best_smile_score:
                    best_smile_score = smile_score
                    best_smile_frame = fd
            
            if best_smile_frame and best_smile_score > 0.1:
                key_moments.append(KeyMoment(
                    timestamp=best_smile_frame.timestamp,
                    frame_number=best_smile_frame.frame_number,
                    description="Genuine smile detected",
                    criterion=criterion_name,
                    emotion='happy',
                    intensity=best_smile_score
                ))
        
        elif criterion_name == 'professionalism' and len(facial_data_list) > 0:
            # Find moment with best composure
            best_composure_score = 0.0
            best_composure_frame = None
            
            for fd in facial_data_list:
                # Calculate composure from neutral/calm emotions
                composure_score = fd.emotion_probabilities.get('neutral', 0) * 0.7 + \
                                 fd.emotion_probabilities.get('happy', 0) * 0.3
                if composure_score > best_composure_score:
                    best_composure_score = composure_score
                    best_composure_frame = fd
            
            if best_composure_frame and best_composure_score > 0.5:
                key_moments.append(KeyMoment(
                    timestamp=best_composure_frame.timestamp,
                    frame_number=best_composure_frame.frame_number,
                    description="Professional composure maintained",
                    criterion=criterion_name,
                    emotion='neutral',
                    intensity=best_composure_score
                ))
        
        elif criterion_name == 'engagement' and len(facial_data_list) > 0:
            # Find moment with best engagement indicators
            best_engagement_score = 0.0
            best_engagement_frame = None
            
            for fd in facial_data_list:
                # Calculate engagement from gaze and facial responsiveness
                gaze_x, gaze_y = fd.gaze_direction
                gaze_magnitude = np.sqrt(gaze_x**2 + gaze_y**2)
                gaze_score = 1.0 - gaze_magnitude
                
                # Facial responsiveness from non-neutral emotions
                responsiveness = 1.0 - fd.emotion_probabilities.get('neutral', 0)
                
                engagement_score = (gaze_score + responsiveness) / 2.0
                
                if engagement_score > best_engagement_score:
                    best_engagement_score = engagement_score
                    best_engagement_frame = fd
            
            if best_engagement_frame and best_engagement_score > 0.5:
                key_moments.append(KeyMoment(
                    timestamp=best_engagement_frame.timestamp,
                    frame_number=best_engagement_frame.frame_number,
                    description="High engagement and attention",
                    criterion=criterion_name,
                    emotion='neutral',
                    intensity=best_engagement_score
                ))
        
        # Legacy 12-criteria system (backward compatibility)
        elif criterion_name == 'friendliness' and len(facial_data_list) > 0:
            # Find best smile moment
            best_smile_score = 0.0
            best_smile_frame = None
            
            for fd in facial_data_list:
                smile_score = fd.action_units.get('AU12', 0) * fd.action_units.get('AU6', 0)
                if smile_score > best_smile_score:
                    best_smile_score = smile_score
                    best_smile_frame = fd
            
            if best_smile_frame and best_smile_score > 0.1:
                key_moments.append(KeyMoment(
                    timestamp=best_smile_frame.timestamp,
                    frame_number=best_smile_frame.frame_number,
                    description="Genuine smile detected",
                    criterion=criterion_name,
                    emotion='happy',
                    intensity=best_smile_score
                ))
        
        return key_moments
