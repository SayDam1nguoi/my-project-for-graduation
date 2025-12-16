"""
Validation and Error Handling for Recruitment Emotion Scoring System.

This module provides comprehensive input validation, quality checks, and
error handling utilities for the emotion scoring pipeline.

Task 10: Add error handling and validation
"""

import cv2
import os
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

if TYPE_CHECKING:
    from .models import FacialData


# Configure logging
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        """String representation of validation result."""
        severity_str = self.severity.value.upper()
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"[{severity_str}] {self.message} ({details_str})"
        return f"[{severity_str}] {self.message}"


@dataclass
class VideoQualityMetrics:
    """Quality metrics for video analysis."""
    resolution: Tuple[int, int]
    fps: float
    duration: float
    total_frames: int
    codec: str
    file_size_mb: float
    is_valid_format: bool
    is_sufficient_resolution: bool
    is_sufficient_duration: bool
    is_sufficient_fps: bool
    quality_score: float  # 0-1
    warnings: List[str]
    
    def __str__(self) -> str:
        """String representation of quality metrics."""
        return (
            f"VideoQualityMetrics(\n"
            f"  Resolution: {self.resolution[0]}x{self.resolution[1]}\n"
            f"  FPS: {self.fps:.2f}\n"
            f"  Duration: {self.duration:.2f}s\n"
            f"  Total Frames: {self.total_frames}\n"
            f"  Codec: {self.codec}\n"
            f"  File Size: {self.file_size_mb:.2f} MB\n"
            f"  Quality Score: {self.quality_score:.2f}\n"
            f"  Warnings: {len(self.warnings)}\n"
            f")"
        )


@dataclass
class FaceDetectionQuality:
    """Quality metrics for face detection."""
    total_frames: int
    frames_with_faces: int
    face_detection_rate: float
    avg_face_confidence: float
    avg_face_size: float  # Relative to frame size
    face_stability: float  # How stable face position is
    quality_score: float  # 0-1
    warnings: List[str]
    
    def __str__(self) -> str:
        """String representation of face detection quality."""
        return (
            f"FaceDetectionQuality(\n"
            f"  Detection Rate: {self.face_detection_rate*100:.1f}%\n"
            f"  Avg Confidence: {self.avg_face_confidence:.2f}\n"
            f"  Avg Face Size: {self.avg_face_size*100:.1f}%\n"
            f"  Face Stability: {self.face_stability:.2f}\n"
            f"  Quality Score: {self.quality_score:.2f}\n"
            f"  Warnings: {len(self.warnings)}\n"
            f")"
        )


class VideoValidator:
    """
    Validates video files for emotion scoring.
    
    Checks:
    - File format and codec
    - Resolution requirements
    - Duration requirements
    - Frame rate requirements
    - File accessibility
    """
    
    # Validation thresholds
    MIN_RESOLUTION = (320, 240)  # Minimum width x height (lowered for mobile videos)
    RECOMMENDED_RESOLUTION = (1280, 720)  # Recommended width x height
    MIN_DURATION = 5.0  # Minimum 5 seconds
    MAX_DURATION = 3600.0  # Maximum 1 hour
    MIN_FPS = 10.0  # Minimum 10 fps (lowered for compatibility)
    RECOMMENDED_FPS = 30.0  # Recommended 30 fps
    MAX_FILE_SIZE_MB = 2000.0  # Maximum 2GB
    
    # Supported video formats
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    RECOMMENDED_CODECS = {'h264', 'h265', 'hevc', 'vp8', 'vp9'}
    
    def __init__(self):
        """Initialize VideoValidator."""
        logger.info("VideoValidator initialized")
    
    def validate_video(self, video_path: str) -> List[ValidationResult]:
        """
        Validate a video file for emotion scoring.
        
        Performs comprehensive validation checks on the video file including:
        - File existence and accessibility
        - File format and codec
        - Resolution requirements
        - Duration requirements
        - Frame rate requirements
        - File size limits
        
        Args:
            video_path: Path to video file
        
        Returns:
            List of ValidationResult objects (empty if all checks pass)
        
        Example:
            >>> validator = VideoValidator()
            >>> results = validator.validate_video("interview.mp4")
            >>> if any(r.severity == ValidationSeverity.ERROR for r in results):
            >>>     print("Video validation failed!")
        """
        results = []
        video_path = Path(video_path)
        
        # Check file existence
        if not video_path.exists():
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Video file not found: {video_path}",
                details={'path': str(video_path)}
            ))
            return results
        
        # Check file accessibility
        if not os.access(video_path, os.R_OK):
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Video file is not readable: {video_path}",
                details={'path': str(video_path)}
            ))
            return results
        
        # Check file format
        file_ext = video_path.suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Unsupported video format: {file_ext}",
                details={
                    'format': file_ext,
                    'supported_formats': list(self.SUPPORTED_FORMATS)
                }
            ))
            return results
        
        # Open video to check properties
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Cannot open video file: {video_path}",
                details={'path': str(video_path)}
            ))
            return results
        
        try:
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            # Calculate duration
            duration = total_frames / fps if fps > 0 else 0.0
            
            # Get codec name
            codec = self._fourcc_to_codec(fourcc)
            
            # Get file size
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            
            # Validate resolution
            if width < self.MIN_RESOLUTION[0] or height < self.MIN_RESOLUTION[1]:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Video resolution too low: {width}x{height}",
                    details={
                        'resolution': (width, height),
                        'minimum': self.MIN_RESOLUTION,
                        'recommended': self.RECOMMENDED_RESOLUTION
                    }
                ))
            elif width < self.RECOMMENDED_RESOLUTION[0] or height < self.RECOMMENDED_RESOLUTION[1]:
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    message=f"Video resolution below recommended: {width}x{height}",
                    details={
                        'resolution': (width, height),
                        'recommended': self.RECOMMENDED_RESOLUTION
                    }
                ))
            
            # Validate duration
            if duration < self.MIN_DURATION:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Video too short: {duration:.1f}s",
                    details={
                        'duration': duration,
                        'minimum': self.MIN_DURATION
                    }
                ))
            elif duration > self.MAX_DURATION:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Video too long: {duration:.1f}s",
                    details={
                        'duration': duration,
                        'maximum': self.MAX_DURATION
                    }
                ))
            
            # Validate FPS
            if fps < self.MIN_FPS:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Frame rate too low: {fps:.1f} fps",
                    details={
                        'fps': fps,
                        'minimum': self.MIN_FPS,
                        'recommended': self.RECOMMENDED_FPS
                    }
                ))
            elif fps < self.RECOMMENDED_FPS:
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.WARNING,
                    message=f"Frame rate below recommended: {fps:.1f} fps",
                    details={
                        'fps': fps,
                        'recommended': self.RECOMMENDED_FPS
                    }
                ))
            
            # Validate codec
            if codec.lower() not in self.RECOMMENDED_CODECS:
                results.append(ValidationResult(
                    is_valid=True,
                    severity=ValidationSeverity.INFO,
                    message=f"Video codec not optimal: {codec}",
                    details={
                        'codec': codec,
                        'recommended_codecs': list(self.RECOMMENDED_CODECS)
                    }
                ))
            
            # Validate file size
            if file_size_mb > self.MAX_FILE_SIZE_MB:
                results.append(ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Video file too large: {file_size_mb:.1f} MB",
                    details={
                        'file_size_mb': file_size_mb,
                        'maximum': self.MAX_FILE_SIZE_MB
                    }
                ))
            
            # Log validation results
            if len(results) == 0:
                logger.info(f"Video validation passed: {video_path.name}")
            else:
                for result in results:
                    if result.severity == ValidationSeverity.ERROR:
                        logger.error(str(result))
                    elif result.severity == ValidationSeverity.WARNING:
                        logger.warning(str(result))
                    else:
                        logger.info(str(result))
        
        finally:
            cap.release()
        
        return results
    
    def get_video_quality_metrics(self, video_path: str) -> VideoQualityMetrics:
        """
        Get comprehensive quality metrics for a video.
        
        Args:
            video_path: Path to video file
        
        Returns:
            VideoQualityMetrics object with detailed quality information
        
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        try:
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            
            duration = total_frames / fps if fps > 0 else 0.0
            codec = self._fourcc_to_codec(fourcc)
            file_size_mb = video_path.stat().st_size / (1024 * 1024)
            
            # Determine validity flags
            is_valid_format = video_path.suffix.lower() in self.SUPPORTED_FORMATS
            is_sufficient_resolution = (width >= self.MIN_RESOLUTION[0] and 
                                       height >= self.MIN_RESOLUTION[1])
            is_sufficient_duration = (self.MIN_DURATION <= duration <= self.MAX_DURATION)
            is_sufficient_fps = fps >= self.MIN_FPS
            
            # Calculate quality score (0-1)
            quality_score = self._calculate_quality_score(
                width, height, fps, duration, codec
            )
            
            # Generate warnings
            warnings = []
            if not is_valid_format:
                warnings.append(f"Unsupported format: {video_path.suffix}")
            if not is_sufficient_resolution:
                warnings.append(f"Low resolution: {width}x{height}")
            if not is_sufficient_duration:
                warnings.append(f"Duration out of range: {duration:.1f}s")
            if not is_sufficient_fps:
                warnings.append(f"Low frame rate: {fps:.1f} fps")
            if file_size_mb > self.MAX_FILE_SIZE_MB:
                warnings.append(f"File too large: {file_size_mb:.1f} MB")
            
            return VideoQualityMetrics(
                resolution=(width, height),
                fps=fps,
                duration=duration,
                total_frames=total_frames,
                codec=codec,
                file_size_mb=file_size_mb,
                is_valid_format=is_valid_format,
                is_sufficient_resolution=is_sufficient_resolution,
                is_sufficient_duration=is_sufficient_duration,
                is_sufficient_fps=is_sufficient_fps,
                quality_score=quality_score,
                warnings=warnings
            )
        
        finally:
            cap.release()
    
    def _fourcc_to_codec(self, fourcc: int) -> str:
        """Convert FourCC code to codec name."""
        try:
            codec_bytes = fourcc.to_bytes(4, byteorder='little')
            codec = codec_bytes.decode('ascii', errors='ignore').strip()
            return codec if codec else 'unknown'
        except:
            return 'unknown'
    
    def _calculate_quality_score(
        self,
        width: int,
        height: int,
        fps: float,
        duration: float,
        codec: str
    ) -> float:
        """
        Calculate overall quality score (0-1).
        
        Considers resolution, fps, duration, and codec quality.
        """
        # Resolution score
        resolution_score = min(1.0, (width * height) / (self.RECOMMENDED_RESOLUTION[0] * self.RECOMMENDED_RESOLUTION[1]))
        
        # FPS score
        fps_score = min(1.0, fps / self.RECOMMENDED_FPS)
        
        # Duration score (optimal around 5-30 minutes)
        if duration < self.MIN_DURATION:
            duration_score = duration / self.MIN_DURATION
        elif duration > 1800:  # 30 minutes
            duration_score = max(0.5, 1.0 - (duration - 1800) / 1800)
        else:
            duration_score = 1.0
        
        # Codec score
        codec_score = 1.0 if codec.lower() in self.RECOMMENDED_CODECS else 0.7
        
        # Weighted average
        quality_score = (
            resolution_score * 0.3 +
            fps_score * 0.3 +
            duration_score * 0.2 +
            codec_score * 0.2
        )
        
        return quality_score



class FaceDetectionValidator:
    """
    Validates face detection quality for emotion scoring.
    
    Checks:
    - Face detection rate
    - Face confidence levels
    - Face size and visibility
    - Face stability across frames
    """
    
    # Quality thresholds
    MIN_DETECTION_RATE = 0.5  # At least 50% of frames should have faces
    RECOMMENDED_DETECTION_RATE = 0.8  # Recommended 80%+
    MIN_FACE_CONFIDENCE = 0.7  # Minimum average confidence
    MIN_FACE_SIZE = 0.1  # Minimum 10% of frame area
    RECOMMENDED_FACE_SIZE = 0.2  # Recommended 20%+
    
    def __init__(self):
        """Initialize FaceDetectionValidator."""
        logger.info("FaceDetectionValidator initialized")
    
    def validate_face_detection(
        self,
        facial_data_list: List['FacialData'],
        total_frames: int
    ) -> List[ValidationResult]:
        """
        Validate face detection quality.
        
        Args:
            facial_data_list: List of FacialData objects from video
            total_frames: Total number of frames in video
        
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        if total_frames == 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message="No frames in video",
                details={'total_frames': 0}
            ))
            return results
        
        # Calculate detection rate
        frames_with_faces = len(facial_data_list)
        detection_rate = frames_with_faces / total_frames
        
        if detection_rate < self.MIN_DETECTION_RATE:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Face detection rate too low: {detection_rate*100:.1f}%",
                details={
                    'detection_rate': detection_rate,
                    'frames_with_faces': frames_with_faces,
                    'total_frames': total_frames,
                    'minimum': self.MIN_DETECTION_RATE
                }
            ))
        elif detection_rate < self.RECOMMENDED_DETECTION_RATE:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Face detection rate below recommended: {detection_rate*100:.1f}%",
                details={
                    'detection_rate': detection_rate,
                    'recommended': self.RECOMMENDED_DETECTION_RATE
                }
            ))
        
        if len(facial_data_list) == 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message="No faces detected in video",
                details={'frames_with_faces': 0}
            ))
            return results
        
        # Calculate average face confidence (using emotion probabilities as proxy)
        avg_confidence = self._calculate_avg_confidence(facial_data_list)
        
        if avg_confidence < self.MIN_FACE_CONFIDENCE:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Low average face confidence: {avg_confidence:.2f}",
                details={
                    'avg_confidence': avg_confidence,
                    'minimum': self.MIN_FACE_CONFIDENCE
                }
            ))
        
        # Calculate average face size
        avg_face_size = self._calculate_avg_face_size(facial_data_list)
        
        if avg_face_size < self.MIN_FACE_SIZE:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Face size too small: {avg_face_size*100:.1f}% of frame",
                details={
                    'avg_face_size': avg_face_size,
                    'minimum': self.MIN_FACE_SIZE,
                    'recommended': self.RECOMMENDED_FACE_SIZE
                }
            ))
        elif avg_face_size < self.RECOMMENDED_FACE_SIZE:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message=f"Face size below recommended: {avg_face_size*100:.1f}% of frame",
                details={
                    'avg_face_size': avg_face_size,
                    'recommended': self.RECOMMENDED_FACE_SIZE
                }
            ))
        
        # Calculate face stability
        face_stability = self._calculate_face_stability(facial_data_list)
        
        if face_stability < 0.7:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Face position unstable: {face_stability:.2f}",
                details={'face_stability': face_stability}
            ))
        
        # Log results
        if len(results) == 0:
            logger.info("Face detection validation passed")
        else:
            for result in results:
                if result.severity == ValidationSeverity.ERROR:
                    logger.error(str(result))
                elif result.severity == ValidationSeverity.WARNING:
                    logger.warning(str(result))
                else:
                    logger.info(str(result))
        
        return results
    
    def get_face_detection_quality(
        self,
        facial_data_list: List['FacialData'],
        total_frames: int
    ) -> FaceDetectionQuality:
        """
        Get comprehensive face detection quality metrics.
        
        Args:
            facial_data_list: List of FacialData objects
            total_frames: Total number of frames in video
        
        Returns:
            FaceDetectionQuality object
        """
        frames_with_faces = len(facial_data_list)
        detection_rate = frames_with_faces / total_frames if total_frames > 0 else 0.0
        
        if len(facial_data_list) == 0:
            return FaceDetectionQuality(
                total_frames=total_frames,
                frames_with_faces=0,
                face_detection_rate=0.0,
                avg_face_confidence=0.0,
                avg_face_size=0.0,
                face_stability=0.0,
                quality_score=0.0,
                warnings=["No faces detected in video"]
            )
        
        avg_confidence = self._calculate_avg_confidence(facial_data_list)
        avg_face_size = self._calculate_avg_face_size(facial_data_list)
        face_stability = self._calculate_face_stability(facial_data_list)
        
        # Calculate quality score
        quality_score = self._calculate_detection_quality_score(
            detection_rate, avg_confidence, avg_face_size, face_stability
        )
        
        # Generate warnings
        warnings = []
        if detection_rate < self.MIN_DETECTION_RATE:
            warnings.append(f"Low detection rate: {detection_rate*100:.1f}%")
        if avg_confidence < self.MIN_FACE_CONFIDENCE:
            warnings.append(f"Low confidence: {avg_confidence:.2f}")
        if avg_face_size < self.MIN_FACE_SIZE:
            warnings.append(f"Small face size: {avg_face_size*100:.1f}%")
        if face_stability < 0.7:
            warnings.append(f"Unstable face position: {face_stability:.2f}")
        
        return FaceDetectionQuality(
            total_frames=total_frames,
            frames_with_faces=frames_with_faces,
            face_detection_rate=detection_rate,
            avg_face_confidence=avg_confidence,
            avg_face_size=avg_face_size,
            face_stability=face_stability,
            quality_score=quality_score,
            warnings=warnings
        )
    
    def _calculate_avg_confidence(self, facial_data_list: List['FacialData']) -> float:
        """Calculate average face detection confidence."""
        if len(facial_data_list) == 0:
            return 0.0
        
        # Use max emotion probability as confidence proxy
        confidences = []
        for fd in facial_data_list:
            max_prob = max(fd.emotion_probabilities.values()) if fd.emotion_probabilities else 0.0
            confidences.append(max_prob)
        
        return sum(confidences) / len(confidences)
    
    def _calculate_avg_face_size(self, facial_data_list: List['FacialData']) -> float:
        """Calculate average face size relative to frame."""
        if len(facial_data_list) == 0:
            return 0.0
        
        # Assume standard frame size for relative calculation
        # In practice, this should use actual frame dimensions
        frame_area = 1920 * 1080  # Assume HD
        
        face_sizes = []
        for fd in facial_data_list:
            bbox = fd.face_bbox
            face_area = bbox.width * bbox.height
            relative_size = face_area / frame_area
            face_sizes.append(relative_size)
        
        return sum(face_sizes) / len(face_sizes)
    
    def _calculate_face_stability(self, facial_data_list: List['FacialData']) -> float:
        """Calculate face position stability across frames."""
        if len(facial_data_list) < 2:
            return 1.0
        
        # Calculate variance in face center positions
        centers = []
        for fd in facial_data_list:
            bbox = fd.face_bbox
            center_x = bbox.x + bbox.width / 2
            center_y = bbox.y + bbox.height / 2
            centers.append((center_x, center_y))
        
        # Calculate standard deviation of positions
        centers_array = np.array(centers)
        std_x = np.std(centers_array[:, 0])
        std_y = np.std(centers_array[:, 1])
        
        # Normalize by frame size (assume HD)
        normalized_std = (std_x / 1920 + std_y / 1080) / 2
        
        # Convert to stability score (lower std = higher stability)
        stability = max(0.0, 1.0 - normalized_std * 10)
        
        return stability
    
    def _calculate_detection_quality_score(
        self,
        detection_rate: float,
        avg_confidence: float,
        avg_face_size: float,
        face_stability: float
    ) -> float:
        """Calculate overall detection quality score (0-1)."""
        # Weighted average of quality factors
        quality_score = (
            detection_rate * 0.3 +
            avg_confidence * 0.3 +
            min(1.0, avg_face_size / self.RECOMMENDED_FACE_SIZE) * 0.2 +
            face_stability * 0.2
        )
        
        return quality_score


class EmotionScoringValidator:
    """
    Comprehensive validator for emotion scoring pipeline.
    
    Combines video validation and face detection validation with
    graceful degradation and confidence scoring.
    """
    
    def __init__(self):
        """Initialize EmotionScoringValidator."""
        self.video_validator = VideoValidator()
        self.face_validator = FaceDetectionValidator()
        logger.info("EmotionScoringValidator initialized")
    
    def validate_for_scoring(
        self,
        video_path: str,
        facial_data_list: Optional[List['FacialData']] = None,
        total_frames: Optional[int] = None
    ) -> Tuple[bool, List[ValidationResult], float]:
        """
        Comprehensive validation for emotion scoring.
        
        Args:
            video_path: Path to video file
            facial_data_list: Optional list of FacialData (for face validation)
            total_frames: Optional total frame count
        
        Returns:
            Tuple of (is_valid, validation_results, confidence_score)
            - is_valid: Whether video can be processed
            - validation_results: List of all validation issues
            - confidence_score: Overall confidence in analysis (0-1)
        """
        all_results = []
        
        # Validate video
        video_results = self.video_validator.validate_video(video_path)
        all_results.extend(video_results)
        
        # Check for critical errors
        has_critical_error = any(
            r.severity == ValidationSeverity.CRITICAL for r in video_results
        )
        has_error = any(
            r.severity == ValidationSeverity.ERROR for r in video_results
        )
        
        if has_critical_error or has_error:
            return False, all_results, 0.0
        
        # Validate face detection if data provided
        if facial_data_list is not None and total_frames is not None:
            face_results = self.face_validator.validate_face_detection(
                facial_data_list, total_frames
            )
            all_results.extend(face_results)
            
            # Check for face detection errors
            has_face_error = any(
                r.severity == ValidationSeverity.ERROR for r in face_results
            )
            
            if has_face_error:
                return False, all_results, 0.0
        
        # Calculate confidence score based on warnings
        confidence_score = self._calculate_confidence_score(all_results)
        
        return True, all_results, confidence_score
    
    def _calculate_confidence_score(self, results: List[ValidationResult]) -> float:
        """
        Calculate confidence score based on validation results.
        
        Args:
            results: List of validation results
        
        Returns:
            Confidence score (0-1)
        """
        if len(results) == 0:
            return 1.0
        
        # Start with perfect confidence
        confidence = 1.0
        
        # Reduce confidence based on severity
        for result in results:
            if result.severity == ValidationSeverity.WARNING:
                confidence -= 0.1
            elif result.severity == ValidationSeverity.INFO:
                confidence -= 0.05
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def get_quality_report(
        self,
        video_path: str,
        facial_data_list: Optional[List['FacialData']] = None,
        total_frames: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive quality report.
        
        Args:
            video_path: Path to video file
            facial_data_list: Optional list of FacialData
            total_frames: Optional total frame count
        
        Returns:
            Dictionary with quality metrics and validation results
        """
        report = {
            'video_path': video_path,
            'timestamp': datetime.now().isoformat(),
            'video_quality': None,
            'face_detection_quality': None,
            'validation_results': [],
            'is_valid': False,
            'confidence_score': 0.0,
            'recommendations': []
        }
        
        try:
            # Get video quality metrics
            video_quality = self.video_validator.get_video_quality_metrics(video_path)
            report['video_quality'] = {
                'resolution': video_quality.resolution,
                'fps': video_quality.fps,
                'duration': video_quality.duration,
                'total_frames': video_quality.total_frames,
                'codec': video_quality.codec,
                'file_size_mb': video_quality.file_size_mb,
                'quality_score': video_quality.quality_score,
                'warnings': video_quality.warnings
            }
            
            # Get face detection quality if data provided
            if facial_data_list is not None and total_frames is not None:
                face_quality = self.face_validator.get_face_detection_quality(
                    facial_data_list, total_frames
                )
                report['face_detection_quality'] = {
                    'detection_rate': face_quality.face_detection_rate,
                    'avg_confidence': face_quality.avg_face_confidence,
                    'avg_face_size': face_quality.avg_face_size,
                    'face_stability': face_quality.face_stability,
                    'quality_score': face_quality.quality_score,
                    'warnings': face_quality.warnings
                }
            
            # Perform validation
            is_valid, validation_results, confidence_score = self.validate_for_scoring(
                video_path, facial_data_list, total_frames
            )
            
            report['is_valid'] = is_valid
            report['confidence_score'] = confidence_score
            report['validation_results'] = [str(r) for r in validation_results]
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                video_quality,
                face_quality if facial_data_list else None,
                validation_results
            )
            report['recommendations'] = recommendations
        
        except Exception as e:
            logger.error(f"Error generating quality report: {e}")
            report['error'] = str(e)
        
        return report
    
    def _generate_recommendations(
        self,
        video_quality: VideoQualityMetrics,
        face_quality: Optional[FaceDetectionQuality],
        validation_results: List[ValidationResult]
    ) -> List[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []
        
        # Video quality recommendations
        if video_quality.quality_score < 0.7:
            recommendations.append("Consider re-recording video with better quality")
        
        if not video_quality.is_sufficient_resolution:
            recommendations.append(
                f"Increase video resolution to at least "
                f"{VideoValidator.MIN_RESOLUTION[0]}x{VideoValidator.MIN_RESOLUTION[1]}"
            )
        
        if not video_quality.is_sufficient_fps:
            recommendations.append(
                f"Increase frame rate to at least {VideoValidator.MIN_FPS} fps"
            )
        
        # Face detection recommendations
        if face_quality:
            if face_quality.face_detection_rate < 0.8:
                recommendations.append(
                    "Ensure face is visible and well-lit throughout the video"
                )
            
            if face_quality.avg_face_size < FaceDetectionValidator.RECOMMENDED_FACE_SIZE:
                recommendations.append(
                    "Position camera closer to capture larger face area"
                )
            
            if face_quality.face_stability < 0.7:
                recommendations.append(
                    "Keep camera and subject position stable during recording"
                )
        
        # Validation-based recommendations
        has_warnings = any(
            r.severity == ValidationSeverity.WARNING for r in validation_results
        )
        if has_warnings:
            recommendations.append(
                "Review validation warnings and consider addressing them for better results"
            )
        
        return recommendations


# Utility functions for error handling

def handle_video_error(video_path: str, error: Exception) -> str:
    """
    Generate user-friendly error message for video processing errors.
    
    Args:
        video_path: Path to video file
        error: Exception that occurred
    
    Returns:
        User-friendly error message
    """
    error_type = type(error).__name__
    
    if isinstance(error, FileNotFoundError):
        return f"Video file not found: {video_path}. Please check the file path."
    
    elif isinstance(error, PermissionError):
        return f"Permission denied accessing video: {video_path}. Check file permissions."
    
    elif isinstance(error, ValueError):
        return f"Invalid video file: {video_path}. {str(error)}"
    
    elif "codec" in str(error).lower():
        return (
            f"Video codec not supported: {video_path}. "
            f"Please convert to a supported format (MP4/H264 recommended)."
        )
    
    else:
        return (
            f"Error processing video: {video_path}. "
            f"{error_type}: {str(error)}"
        )


def handle_face_detection_error(error: Exception) -> str:
    """
    Generate user-friendly error message for face detection errors.
    
    Args:
        error: Exception that occurred
    
    Returns:
        User-friendly error message
    """
    error_type = type(error).__name__
    
    if "model" in str(error).lower():
        return (
            "Face detection model error. "
            "Please ensure face detection models are properly installed."
        )
    
    elif "memory" in str(error).lower() or "cuda" in str(error).lower():
        return (
            "Insufficient memory for face detection. "
            "Try reducing video resolution or frame sampling rate."
        )
    
    else:
        return f"Face detection error: {error_type}: {str(error)}"


def log_validation_results(results: List[ValidationResult], logger_instance=None):
    """
    Log validation results with appropriate severity levels.
    
    Args:
        results: List of ValidationResult objects
        logger_instance: Optional logger instance (uses module logger if None)
    """
    log = logger_instance or logger
    
    for result in results:
        if result.severity == ValidationSeverity.CRITICAL:
            log.critical(str(result))
        elif result.severity == ValidationSeverity.ERROR:
            log.error(str(result))
        elif result.severity == ValidationSeverity.WARNING:
            log.warning(str(result))
        else:
            log.info(str(result))
