"""
Facial Data Extractor for Recruitment Emotion Scoring System (4-Criteria).

This module extracts comprehensive facial features from video frames for the
4-criteria emotion scoring system, including:
- Gaze tracking for confidence and engagement
- Smile authenticity detection for positivity
- Posture data extraction for confidence
- Attention indicators extraction for engagement
- Facial landmarks, emotion probabilities, action units, and head pose

Requirements: 1.1, 2.1, 4.1
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from src.inference.face_detector import FaceDetector
from src.inference.preprocessor import FacePreprocessor
from src.inference.emotion_classifier import EmotionClassifier
from .models import FacialData, BoundingBox, MicroExpression, PostureData, AttentionData
from .validation import VideoValidator, handle_video_error
import logging

logger = logging.getLogger(__name__)


class FacialDataExtractor:
    """
    Extracts comprehensive facial data from video frames for 4-criteria emotion scoring.
    
    This class interfaces with existing FaceDetector, FacePreprocessor, and
    EmotionClassifier to extract facial features, emotions, action units,
    gaze direction, head pose, posture data, and attention indicators.
    
    Features for 4-Criteria System:
    - Gaze tracking for confidence and engagement assessment
    - Smile authenticity detection for positivity assessment
    - Posture data extraction for confidence assessment
    - Attention indicators extraction for engagement assessment
    - Extract facial landmarks and bounding boxes
    - Detect emotions with probability distributions
    - Extract head pose (pitch, yaw, roll)
    - Detect micro-expressions
    
    Requirements:
    - 1.1: Detect confidence indicators including voice tone, posture, eye contact, and facial expressions
    - 2.1: Detect positivity indicators such as genuine smiles, warm expressions, and friendly demeanor
    - 4.1: Detect engagement indicators including eye contact, head movements, facial responsiveness, and attention
    """
    
    def __init__(
        self,
        face_detector: Optional[FaceDetector] = None,
        face_preprocessor: Optional[FacePreprocessor] = None,
        emotion_classifier: Optional[EmotionClassifier] = None,
        device: str = 'auto',
        enable_validation: bool = True,
        confidence_threshold: float = 0.7,
        enable_posture_extraction: bool = True,
        enable_attention_extraction: bool = True
    ):
        """
        Initialize FacialDataExtractor.
        
        Args:
            face_detector: FaceDetector instance (creates default if None)
            face_preprocessor: FacePreprocessor instance (creates default if None)
            emotion_classifier: EmotionClassifier instance (creates default if None)
            device: Device for processing ('cuda', 'cpu', or 'auto')
            enable_validation: Enable input validation (default: True)
            confidence_threshold: Face detection confidence threshold (default: 0.7, lower = more detections)
            enable_posture_extraction: Enable posture data extraction for confidence (default: True)
            enable_attention_extraction: Enable attention indicators extraction for engagement (default: True)
        """
        # Initialize components
        self.face_detector = face_detector or FaceDetector(
            device=device,
            confidence_threshold=confidence_threshold,
            max_faces=1
        )
        
        self.face_preprocessor = face_preprocessor or FacePreprocessor(
            target_size=(224, 224)
        )
        
        # Emotion classifier requires a model path
        if emotion_classifier is None:
            model_path = Path('models/efficientnet_b2_best.pth')
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Emotion classifier model not found: {model_path}. "
                    f"Please provide an emotion_classifier instance or ensure model exists."
                )
            self.emotion_classifier = EmotionClassifier(
                model_path=model_path,
                device=device
            )
        else:
            self.emotion_classifier = emotion_classifier
        
        # Micro-expression detection parameters
        self.micro_expression_threshold = 0.1  # Minimum intensity change
        self.micro_expression_max_duration = 0.5  # Maximum 500ms
        
        # Feature extraction flags
        self.enable_posture_extraction = enable_posture_extraction
        self.enable_attention_extraction = enable_attention_extraction
        
        # Validation
        self.enable_validation = enable_validation
        self.video_validator = VideoValidator() if enable_validation else None
        
        # Previous frame data for temporal analysis
        self.previous_facial_data: Optional[FacialData] = None
        
        print(f"FacialDataExtractor initialized (confidence_threshold: {self.face_detector.confidence_threshold})")
        print(f"  Posture extraction: {'enabled' if enable_posture_extraction else 'disabled'}")
        print(f"  Attention extraction: {'enabled' if enable_attention_extraction else 'disabled'}")
    
    def extract_from_video(
        self,
        video_path: str,
        frame_sampling_rate: int = 1
    ) -> List[FacialData]:
        """
        Extract facial data from all frames in a video for 4-criteria emotion scoring.
        
        Processes video frame by frame, detecting faces and extracting comprehensive
        facial features including:
        - Gaze tracking for confidence and engagement
        - Smile authenticity detection for positivity
        - Posture data extraction for confidence
        - Attention indicators extraction for engagement
        
        Args:
            video_path: Path to video file
            frame_sampling_rate: Process every Nth frame (1 = all frames)
        
        Returns:
            List of FacialData objects, one per processed frame
        
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        
        Requirements:
            - 1.1: Extract confidence indicators (voice tone, posture, eye contact, facial expressions)
            - 2.1: Extract positivity indicators (smiles, warm expressions, friendly demeanor)
            - 4.1: Extract engagement indicators (eye contact, head movements, facial responsiveness, attention)
        """
        video_path = Path(video_path)
        
        # Validate video if enabled
        if self.enable_validation and self.video_validator:
            try:
                validation_results = self.video_validator.validate_video(str(video_path))
                
                # Check for critical errors
                critical_errors = [r for r in validation_results 
                                 if r.severity.value in ['critical', 'error']]
                
                if critical_errors:
                    error_messages = [r.message for r in critical_errors]
                    raise ValueError(
                        f"Video validation failed: {'; '.join(error_messages)}"
                    )
                
                # Log warnings
                warnings = [r for r in validation_results 
                           if r.severity.value == 'warning']
                for warning in warnings:
                    logger.warning(str(warning))
            
            except Exception as e:
                error_msg = handle_video_error(str(video_path), e)
                logger.error(error_msg)
                raise ValueError(error_msg) from e
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path.name}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Sampling rate: every {frame_sampling_rate} frame(s)")
        
        facial_data_list = []
        frame_number = 0
        frames_with_faces = 0
        frames_without_faces = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # Skip frames based on sampling rate
                if (frame_number - 1) % frame_sampling_rate != 0:
                    continue
                
                # Calculate timestamp
                timestamp = frame_number / fps if fps > 0 else frame_number / 30.0
                
                # Extract facial data from frame
                facial_data = self._extract_from_frame(frame, frame_number, timestamp)
                
                if facial_data is not None:
                    facial_data_list.append(facial_data)
                    frames_with_faces += 1
                else:
                    frames_without_faces += 1
                
                # Progress indicator
                if frame_number % 100 == 0:
                    print(f"  Processed {frame_number}/{total_frames} frames... (Faces detected: {frames_with_faces}, No faces: {frames_without_faces})")
        
        finally:
            cap.release()
        
        # Calculate detection rate
        detection_rate = (frames_with_faces / max(frame_number, 1)) * 100
        
        print(f"Extraction complete:")
        print(f"  Total frames processed: {frame_number}")
        print(f"  Frames with faces detected: {frames_with_faces}")
        print(f"  Frames without faces: {frames_without_faces}")
        print(f"  Detection rate: {detection_rate:.1f}%")
        
        # Warn if no faces detected
        if len(facial_data_list) == 0:
            error_msg = (
                f"No faces detected in video {video_path}. "
                f"Processed {frame_number} frames with 0 successful detections. "
                f"\nPossible causes and solutions: "
                f"\n1) Face detection confidence threshold too high (current: {self.face_detector.confidence_threshold}) - try lowering to 0.3-0.4"
                f"\n2) Poor video quality or lighting - ensure adequate lighting and clear visibility"
                f"\n3) Faces too small or not visible - face should be at least 48x48 pixels"
                f"\n4) Face detection model not loaded properly - verify model file exists at models/efficientnet_b2_best.pth"
            )
            logger.warning(error_msg)
            print(f"\n❌ ERROR: {error_msg}")
        # Warn if detection rate is low (< 30%)
        elif detection_rate < 30.0:
            logger.warning(
                f"Low face detection rate ({detection_rate:.1f}%) in video {video_path}. "
                f"Detected faces in only {frames_with_faces}/{frame_number} frames. "
                f"This may affect emotion scoring accuracy."
            )
            print(f"\n⚠️  WARNING: Low detection rate ({detection_rate:.1f}%)")
            print(f"  Troubleshooting suggestions:")
            print(f"  1. Lower the confidence threshold (current: {self.face_detector.confidence_threshold}, try 0.3-0.4)")
            print(f"  2. Check video quality - ensure good lighting and clear face visibility")
            print(f"  3. Verify face is not too small in frame (should be at least 48x48 pixels)")
            print(f"  4. Check if face is partially occluded or at extreme angles")
            print(f"  5. Ensure video resolution is adequate (minimum 480p recommended)")
        
        return facial_data_list
    
    def _extract_from_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float
    ) -> Optional[FacialData]:
        """
        Extract facial data from a single frame.
        
        Args:
            frame: Video frame (BGR format)
            frame_number: Frame number in video
            timestamp: Timestamp in seconds
        
        Returns:
            FacialData object if face detected, None otherwise
        """
        # Detect faces
        detections = self.face_detector.detect_faces(frame)
        
        if len(detections) == 0:
            self.previous_facial_data = None
            return None
        
        # Use first (most confident) detection
        detection = detections[0]
        
        # Extract bounding box
        x, y, w, h = detection.bbox
        face_bbox = BoundingBox(x=x, y=y, width=w, height=h)
        
        # Extract landmarks
        landmarks = detection.landmarks if detection.landmarks is not None else np.zeros((5, 2))
        
        # Preprocess face for emotion classification
        face_tensor = self.face_preprocessor.preprocess(frame, detection)
        
        # Predict emotion
        emotion_prediction = self.emotion_classifier.predict(face_tensor)
        
        # Extract action units (facial muscle activations)
        action_units = self._extract_action_units(frame, detection)
        
        # Estimate gaze direction
        gaze_direction = self._estimate_gaze_direction(landmarks)
        
        # Estimate head pose
        head_pose = self._estimate_head_pose(landmarks, frame.shape)
        
        # Extract posture data for confidence assessment
        posture_data = None
        if self.enable_posture_extraction:
            posture_data = self._extract_posture_data(landmarks, head_pose, face_bbox, frame.shape)
        
        # Extract attention indicators for engagement assessment
        attention_data = None
        if self.enable_attention_extraction:
            attention_data = self._extract_attention_data(
                gaze_direction, 
                head_pose, 
                emotion_prediction.probabilities,
                timestamp
            )
        
        # Create FacialData object
        facial_data = FacialData(
            frame_number=frame_number,
            timestamp=timestamp,
            face_bbox=face_bbox,
            landmarks=landmarks,
            emotion_probabilities=emotion_prediction.probabilities,
            action_units=action_units,
            gaze_direction=gaze_direction,
            head_pose=head_pose,
            posture_data=posture_data,
            attention_data=attention_data
        )
        
        # Store for temporal analysis
        self.previous_facial_data = facial_data
        
        return facial_data
    
    def _extract_action_units(
        self,
        frame: np.ndarray,
        detection
    ) -> dict:
        """
        Extract facial action units (AU) from detected face.
        
        Action units represent specific facial muscle activations and are
        used for smile authenticity detection and micro-expression analysis.
        
        Args:
            frame: Video frame
            detection: Face detection with landmarks
        
        Returns:
            Dictionary mapping action unit names to activation levels (0-1)
        
        Requirements:
            - 4.4: Distinguish between genuine and forced smiles using AU
        """
        # Initialize action units
        action_units = {
            'AU6': 0.0,   # Cheek raiser (genuine smile indicator)
            'AU12': 0.0,  # Lip corner puller (smile)
            'AU1': 0.0,   # Inner brow raiser
            'AU2': 0.0,   # Outer brow raiser
            'AU4': 0.0,   # Brow lowerer
            'AU5': 0.0,   # Upper lid raiser
            'AU7': 0.0,   # Lid tightener
            'AU9': 0.0,   # Nose wrinkler
            'AU10': 0.0,  # Upper lip raiser
            'AU15': 0.0,  # Lip corner depressor
            'AU17': 0.0,  # Chin raiser
            'AU20': 0.0,  # Lip stretcher
            'AU23': 0.0,  # Lip tightener
            'AU24': 0.0,  # Lip pressor
            'AU25': 0.0,  # Lips part
            'AU26': 0.0,  # Jaw drop
            'AU27': 0.0,  # Mouth stretch
        }
        
        # If no landmarks, return zeros
        if detection.landmarks is None or len(detection.landmarks) < 5:
            return action_units
        
        landmarks = detection.landmarks
        
        # Extract key landmark points (5-point landmarks from MTCNN)
        # Points: left_eye, right_eye, nose, left_mouth, right_mouth
        if len(landmarks) >= 5:
            left_eye = landmarks[0]
            right_eye = landmarks[1]
            nose = landmarks[2]
            left_mouth = landmarks[3]
            right_mouth = landmarks[4]
            
            # Calculate distances for AU estimation
            eye_distance = np.linalg.norm(right_eye - left_eye)
            mouth_width = np.linalg.norm(right_mouth - left_mouth)
            
            # Estimate AU12 (smile) from mouth width relative to eye distance
            # Wider mouth relative to eyes indicates smiling
            if eye_distance > 0:
                mouth_ratio = mouth_width / eye_distance
                action_units['AU12'] = min(1.0, max(0.0, (mouth_ratio - 0.5) * 2.0))
            
            # Estimate AU6 (cheek raiser) from eye-mouth vertical distance
            # Eyes appear "squinted" when cheeks are raised (genuine smile)
            eye_center = (left_eye + right_eye) / 2
            mouth_center = (left_mouth + right_mouth) / 2
            eye_mouth_distance = np.linalg.norm(mouth_center - eye_center)
            
            # Normalized distance (smaller = more cheek raising)
            if eye_distance > 0:
                vertical_ratio = eye_mouth_distance / eye_distance
                # Inverse relationship: smaller distance = higher AU6
                action_units['AU6'] = min(1.0, max(0.0, (2.0 - vertical_ratio) / 2.0))
            
            # Estimate AU25 (lips part) from mouth opening
            # This is a simplified estimation
            action_units['AU25'] = action_units['AU12'] * 0.5
            
            # Estimate AU1/AU2 (brow raisers) from eye-nose distance
            # Higher eyebrows = larger distance
            eye_nose_distance = np.linalg.norm(eye_center - nose)
            if eye_distance > 0:
                brow_ratio = eye_nose_distance / eye_distance
                action_units['AU1'] = min(1.0, max(0.0, (brow_ratio - 0.8) * 2.0))
                action_units['AU2'] = action_units['AU1'] * 0.8
        
        return action_units
    
    def detect_smile_authenticity(self, facial_data: FacialData) -> float:
        """
        Detect smile authenticity using Duchenne smile markers.
        
        A genuine (Duchenne) smile involves both AU6 (cheek raiser) and AU12
        (lip corner puller). A forced smile typically only involves AU12.
        
        Args:
            facial_data: FacialData object with action units
        
        Returns:
            Authenticity score (0-1), where 1 is most authentic
        
        Requirements:
            - 4.4: Distinguish between genuine and forced smiles
        """
        action_units = facial_data.action_units
        
        # Get relevant action units
        au6 = action_units.get('AU6', 0.0)  # Cheek raiser (genuine smile marker)
        au12 = action_units.get('AU12', 0.0)  # Lip corner puller (smile)
        
        # No smile detected
        if au12 < 0.3:
            return 0.0
        
        # Duchenne smile: both AU6 and AU12 are active
        # Authenticity is based on the presence of AU6 relative to AU12
        if au6 > 0.3 and au12 > 0.3:
            # Genuine smile: high AU6 activation
            authenticity = min(1.0, (au6 + au12) / 2.0)
        else:
            # Forced smile: AU12 without AU6
            authenticity = au6 / max(au12, 0.1)  # Ratio of AU6 to AU12
        
        return authenticity
    
    def _estimate_gaze_direction(self, landmarks: np.ndarray) -> Tuple[float, float]:
        """
        Estimate gaze direction from facial landmarks.
        
        Uses eye landmarks to estimate where the person is looking.
        Returns normalized gaze direction (x, y) where:
        - x: -1 (left) to 1 (right)
        - y: -1 (down) to 1 (up)
        - (0, 0) indicates looking at camera
        
        Args:
            landmarks: Facial landmarks array
        
        Returns:
            Tuple of (gaze_x, gaze_y) normalized to [-1, 1]
        
        Requirements:
            - 8.1: Track eye contact patterns and duration
        """
        # If insufficient landmarks, assume looking at camera
        if landmarks is None or len(landmarks) < 5:
            return (0.0, 0.0)
        
        # Extract eye landmarks (5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth)
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        
        # Calculate eye center
        eye_center = (left_eye + right_eye) / 2
        
        # Calculate horizontal gaze from eye-nose relationship
        # If eyes are centered relative to nose, gaze is forward
        eye_nose_vector = nose - eye_center
        eye_distance = np.linalg.norm(right_eye - left_eye)
        
        if eye_distance > 0:
            # Horizontal gaze: positive = looking right, negative = looking left
            gaze_x = eye_nose_vector[0] / (eye_distance / 2)
            gaze_x = np.clip(gaze_x, -1.0, 1.0)
            
            # Vertical gaze: positive = looking up, negative = looking down
            gaze_y = -eye_nose_vector[1] / (eye_distance / 2)
            gaze_y = np.clip(gaze_y, -1.0, 1.0)
        else:
            gaze_x = 0.0
            gaze_y = 0.0
        
        return (float(gaze_x), float(gaze_y))
    
    def _estimate_head_pose(
        self,
        landmarks: np.ndarray,
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[float, float, float]:
        """
        Estimate head pose (pitch, yaw, roll) from facial landmarks.
        
        Args:
            landmarks: Facial landmarks array
            frame_shape: Frame shape (height, width, channels)
        
        Returns:
            Tuple of (pitch, yaw, roll) in degrees
            - pitch: up/down rotation (-90 to 90)
            - yaw: left/right rotation (-90 to 90)
            - roll: tilt rotation (-180 to 180)
        """
        # If insufficient landmarks, assume neutral pose
        if landmarks is None or len(landmarks) < 5:
            return (0.0, 0.0, 0.0)
        
        # Extract key points
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        
        # Calculate roll (head tilt) from eye alignment
        eye_vector = right_eye - left_eye
        roll = np.degrees(np.arctan2(eye_vector[1], eye_vector[0]))
        
        # Calculate yaw (left/right turn) from eye-nose relationship
        eye_center = (left_eye + right_eye) / 2
        eye_distance = np.linalg.norm(right_eye - left_eye)
        
        if eye_distance > 0:
            # Horizontal offset of nose from eye center
            nose_offset_x = (nose[0] - eye_center[0]) / (eye_distance / 2)
            yaw = np.degrees(np.arctan(nose_offset_x)) * 2  # Scale for visibility
            yaw = np.clip(yaw, -90, 90)
        else:
            yaw = 0.0
        
        # Calculate pitch (up/down tilt) from eye-mouth relationship
        mouth_center = (left_mouth + right_mouth) / 2
        eye_mouth_vector = mouth_center - eye_center
        
        if eye_distance > 0:
            # Vertical offset normalized by eye distance
            vertical_ratio = eye_mouth_vector[1] / eye_distance
            pitch = np.degrees(np.arctan(vertical_ratio - 1.5)) * 2  # Scale and offset
            pitch = np.clip(pitch, -90, 90)
        else:
            pitch = 0.0
        
        return (float(pitch), float(yaw), float(roll))
    
    def detect_micro_expressions(
        self,
        facial_data_list: List[FacialData],
        fps: float = 30.0
    ) -> List[MicroExpression]:
        """
        Detect micro-expressions from a sequence of facial data.
        
        Micro-expressions are brief (<500ms) involuntary facial expressions
        that may reveal hidden emotions.
        
        Args:
            facial_data_list: List of FacialData objects in temporal order
            fps: Video frame rate for duration calculation
        
        Returns:
            List of detected MicroExpression objects
        """
        micro_expressions = []
        
        if len(facial_data_list) < 3:
            return micro_expressions
        
        # Analyze emotion changes across frames
        for i in range(1, len(facial_data_list) - 1):
            prev_data = facial_data_list[i - 1]
            curr_data = facial_data_list[i]
            next_data = facial_data_list[i + 1]
            
            # Get dominant emotions
            prev_emotion = max(prev_data.emotion_probabilities.items(), key=lambda x: x[1])
            curr_emotion = max(curr_data.emotion_probabilities.items(), key=lambda x: x[1])
            next_emotion = max(next_data.emotion_probabilities.items(), key=lambda x: x[1])
            
            # Detect brief emotion change (micro-expression pattern)
            if (curr_emotion[0] != prev_emotion[0] and 
                curr_emotion[0] != next_emotion[0] and
                curr_emotion[1] > 0.5):  # Sufficient confidence
                
                # Calculate intensity change
                intensity_change = abs(curr_emotion[1] - prev_emotion[1])
                
                if intensity_change > self.micro_expression_threshold:
                    # Calculate duration
                    duration = (curr_data.timestamp - prev_data.timestamp)
                    
                    # Check if duration is within micro-expression range
                    if duration <= self.micro_expression_max_duration:
                        micro_expr = MicroExpression(
                            start_frame=prev_data.frame_number,
                            end_frame=curr_data.frame_number,
                            duration=duration,
                            expression_type=curr_emotion[0],
                            intensity=curr_emotion[1],
                            confidence=min(curr_emotion[1], 1.0 - duration / self.micro_expression_max_duration)
                        )
                        micro_expressions.append(micro_expr)
        
        return micro_expressions
    
    def _extract_posture_data(
        self,
        landmarks: np.ndarray,
        head_pose: Tuple[float, float, float],
        face_bbox: BoundingBox,
        frame_shape: Tuple[int, int, int]
    ) -> PostureData:
        """
        Extract posture indicators for confidence assessment.
        
        Analyzes head position, alignment, and orientation to assess posture quality.
        Good posture (upright, centered, facing camera) indicates confidence.
        
        Args:
            landmarks: Facial landmarks array
            head_pose: Head pose (pitch, yaw, roll) in degrees
            face_bbox: Face bounding box
            frame_shape: Frame shape (height, width, channels)
        
        Returns:
            PostureData object with posture indicators
        
        Requirements:
            - 1.1: Extract posture indicators for confidence assessment
        """
        pitch, yaw, roll = head_pose
        frame_height, frame_width = frame_shape[:2]
        
        # Head tilt: absolute roll angle (0 = upright, higher = more tilted)
        head_tilt = abs(roll)
        
        # Shoulder alignment: estimated from head roll and face position
        # Lower roll indicates better shoulder alignment
        shoulder_alignment = max(0.0, 1.0 - (abs(roll) / 45.0))  # Normalize to 0-1
        
        # Body orientation: based on yaw (facing camera vs. turned away)
        # Yaw close to 0 = facing camera directly
        body_orientation = max(0.0, 1.0 - (abs(yaw) / 45.0))  # Normalize to 0-1
        
        # Face vertical position: centered is better for confidence
        # Calculate face center Y position relative to frame
        face_center_y = face_bbox.y + face_bbox.height / 2
        vertical_position = face_center_y / frame_height
        
        # Ideal vertical position is slightly above center (0.4-0.5)
        # Too low suggests slouching, too high suggests leaning back
        vertical_alignment = 1.0 - abs(vertical_position - 0.45) * 2.0
        vertical_alignment = max(0.0, min(1.0, vertical_alignment))
        
        # Overall posture confidence: weighted combination of indicators
        posture_confidence = (
            shoulder_alignment * 0.3 +
            body_orientation * 0.4 +
            vertical_alignment * 0.3
        )
        
        return PostureData(
            head_tilt=head_tilt,
            shoulder_alignment=shoulder_alignment,
            body_orientation=body_orientation,
            posture_confidence=posture_confidence
        )
    
    def _extract_attention_data(
        self,
        gaze_direction: Tuple[float, float],
        head_pose: Tuple[float, float, float],
        emotion_probabilities: Dict[str, float],
        timestamp: float
    ) -> AttentionData:
        """
        Extract attention indicators for engagement assessment.
        
        Analyzes gaze, head movement, and facial responsiveness to assess attention level.
        High attention (steady gaze, responsive expressions) indicates engagement.
        
        Args:
            gaze_direction: Gaze direction (x, y) normalized to [-1, 1]
            head_pose: Head pose (pitch, yaw, roll) in degrees
            emotion_probabilities: Emotion probability distribution
            timestamp: Current timestamp in seconds
        
        Returns:
            AttentionData object with attention indicators
        
        Requirements:
            - 4.1: Extract attention indicators for engagement assessment
        """
        gaze_x, gaze_y = gaze_direction
        pitch, yaw, roll = head_pose
        
        # Eye contact score: based on gaze direction
        # Gaze close to (0, 0) indicates looking at camera
        gaze_magnitude = np.sqrt(gaze_x**2 + gaze_y**2)
        eye_contact_score = max(0.0, 1.0 - gaze_magnitude)  # Normalize to 0-1
        
        # Head movement frequency: calculated from temporal changes
        # This is a placeholder - actual frequency calculated in aggregate analysis
        head_movement_frequency = 0.0
        if self.previous_facial_data is not None:
            prev_pitch, prev_yaw, prev_roll = self.previous_facial_data.head_pose
            time_delta = timestamp - self.previous_facial_data.timestamp
            
            if time_delta > 0:
                # Calculate angular change
                pitch_change = abs(pitch - prev_pitch)
                yaw_change = abs(yaw - prev_yaw)
                roll_change = abs(roll - prev_roll)
                total_change = pitch_change + yaw_change + roll_change
                
                # Frequency = change per second
                head_movement_frequency = total_change / time_delta
        
        # Facial responsiveness: based on emotion distribution entropy
        # High entropy = varied expressions = responsive
        # Low entropy = flat affect = less responsive
        emotion_values = list(emotion_probabilities.values())
        if len(emotion_values) > 0 and sum(emotion_values) > 0:
            # Normalize probabilities
            total = sum(emotion_values)
            probs = [v / total for v in emotion_values]
            
            # Calculate entropy
            entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
            max_entropy = np.log(len(emotion_values))
            
            # Normalize to 0-1
            facial_responsiveness = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            facial_responsiveness = 0.0
        
        # Overall attention score: weighted combination of indicators
        attention_score = (
            eye_contact_score * 0.5 +
            min(1.0, head_movement_frequency / 10.0) * 0.2 +  # Normalize frequency
            facial_responsiveness * 0.3
        )
        
        return AttentionData(
            eye_contact_score=eye_contact_score,
            head_movement_frequency=head_movement_frequency,
            facial_responsiveness=facial_responsiveness,
            attention_score=attention_score
        )
    
    def track_eye_contact(
        self,
        facial_data_list: List[FacialData],
        threshold: float = 0.3
    ) -> dict:
        """
        Track eye contact patterns throughout video.
        
        Analyzes gaze direction to determine when the person is making
        eye contact (looking at camera) vs. looking away.
        
        Args:
            facial_data_list: List of FacialData objects
            threshold: Maximum gaze deviation for eye contact (0-1)
        
        Returns:
            Dictionary with eye contact statistics:
            - total_duration: Total video duration in seconds
            - eye_contact_duration: Duration of eye contact in seconds
            - eye_contact_percentage: Percentage of time making eye contact
            - eye_contact_frames: Number of frames with eye contact
            - total_frames: Total number of frames
        
        Requirements:
            - 8.1: Track eye contact patterns and duration
        """
        if len(facial_data_list) == 0:
            return {
                'total_duration': 0.0,
                'eye_contact_duration': 0.0,
                'eye_contact_percentage': 0.0,
                'eye_contact_frames': 0,
                'total_frames': 0
            }
        
        eye_contact_frames = 0
        total_frames = len(facial_data_list)
        
        for facial_data in facial_data_list:
            gaze_x, gaze_y = facial_data.gaze_direction
            
            # Calculate gaze magnitude (distance from center)
            gaze_magnitude = np.sqrt(gaze_x**2 + gaze_y**2)
            
            # Eye contact if gaze is within threshold of center
            if gaze_magnitude <= threshold:
                eye_contact_frames += 1
        
        # Calculate durations
        total_duration = facial_data_list[-1].timestamp - facial_data_list[0].timestamp
        eye_contact_duration = (eye_contact_frames / total_frames) * total_duration if total_frames > 0 else 0.0
        eye_contact_percentage = (eye_contact_frames / total_frames * 100) if total_frames > 0 else 0.0
        
        return {
            'total_duration': total_duration,
            'eye_contact_duration': eye_contact_duration,
            'eye_contact_percentage': eye_contact_percentage,
            'eye_contact_frames': eye_contact_frames,
            'total_frames': total_frames
        }
