"""
Video Preprocessing Pipeline Orchestrator

This module provides the main pipeline orchestrator that coordinates all
preprocessing steps.
"""

import cv2
import numpy as np
import time
from typing import Optional, List, Tuple
from pathlib import Path
import logging

from .config import PreprocessingConfig, load_preprocessing_config
from .normalizer import VideoNormalizer
from .quality_filter import FrameQualityFilter
from .stabilizer import FaceStabilizer
from .lighting_corrector import LightingCorrector
from .aligner import FaceAligner
from .temporal_smoother import TemporalSmoother
from .data_models import (
    FrameMetadata, FaceData, ProcessedFrame, EmotionPrediction,
    PipelineStatistics, VideoAnalysisResult, InsufficientQualityError
)

logger = logging.getLogger(__name__)


class VideoPreprocessingPipeline:
    """
    Main preprocessing pipeline orchestrator.
    
    Coordinates all preprocessing steps and manages the flow of frames
    through the pipeline.
    """
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """
        Initialize preprocessing pipeline.
        
        Args:
            config: Preprocessing configuration. If None, loads from default path.
        """
        if config is None:
            config = load_preprocessing_config()
        
        self.config = config
        self.enabled = config.enabled
        
        # Initialize preprocessing modules
        self.normalizer = None
        self.quality_filter = None
        self.stabilizer = None
        self.lighting_corrector = None
        self.aligner = None
        self.temporal_smoother = None
        
        if self.enabled:
            self._initialize_modules()
        
        # Statistics
        self.stats = PipelineStatistics(
            total_frames=0,
            valid_frames=0,
            filtered_frames=0
        )
        
        logger.info(f"VideoPreprocessingPipeline initialized (enabled={self.enabled})")
    
    def _initialize_modules(self):
        """Initialize all preprocessing modules based on configuration."""
        # Step 1: Video Normalizer
        if self.config.normalization.enabled:
            self.normalizer = VideoNormalizer(
                target_fps=self.config.normalization.target_fps,
                target_resolution=tuple(self.config.normalization.target_resolution),
                enable_sharpening=self.config.normalization.sharpening['enabled'],
                sharpen_strength=self.config.normalization.sharpening['strength'],
                enable_denoising=self.config.normalization.denoising['enabled'],
                denoise_strength=self.config.normalization.denoising['strength'],
                denoise_method=self.config.normalization.denoising['method']
            )
            logger.info("VideoNormalizer initialized")
        
        # Step 4: Frame Quality Filter
        if self.config.quality_filter.enabled:
            self.quality_filter = FrameQualityFilter(
                blur_threshold=self.config.quality_filter.blur_threshold,
                confidence_threshold=self.config.quality_filter.confidence_threshold,
                max_pose_angle=self.config.quality_filter.max_pose_angle,
                check_occlusion=self.config.quality_filter.check_occlusion,
                min_face_size=tuple(self.config.quality_filter.min_face_size),
                min_fps_after_filter=self.config.quality_filter.min_fps_after_filter
            )
            logger.info("FrameQualityFilter initialized")
        
        # Step 2: Face Stabilizer
        if self.config.stabilization.enabled:
            self.stabilizer = FaceStabilizer(
                window_size=self.config.stabilization.window_size,
                method=self.config.stabilization.method,
                smoothing_factor=self.config.stabilization.smoothing_factor
            )
            logger.info("FaceStabilizer initialized")
        
        # Step 3: Lighting Corrector
        if self.config.lighting.enabled:
            self.lighting_corrector = LightingCorrector(
                enable_clahe=self.config.lighting.clahe['enabled'],
                clahe_clip_limit=self.config.lighting.clahe['clip_limit'],
                clahe_tile_size=tuple(self.config.lighting.clahe['tile_size']),
                enable_white_balance=self.config.lighting.white_balance['enabled'],
                white_balance_method=self.config.lighting.white_balance['method'],
                enable_contrast=self.config.lighting.contrast['enabled'],
                contrast_factor=self.config.lighting.contrast['factor']
            )
            logger.info("LightingCorrector initialized")
        
        # Step 5: Face Aligner
        if self.config.alignment.enabled:
            self.aligner = FaceAligner(
                method=self.config.alignment.method,
                crop_margin=self.config.alignment.crop_margin,
                target_size=tuple(self.config.alignment.target_size),
                correct_rotation=self.config.alignment.correct_rotation,
                correct_perspective=self.config.alignment.correct_perspective
            )
            logger.info("FaceAligner initialized")
        
        # Step 6: Temporal Smoother
        if self.config.temporal_smoothing.enabled:
            self.temporal_smoother = TemporalSmoother(
                method=self.config.temporal_smoothing.method,
                window_size=self.config.temporal_smoothing.window_size,
                ema_alpha=self.config.temporal_smoothing.ema_alpha,
                min_confidence_for_smoothing=self.config.temporal_smoothing.min_confidence_for_smoothing
            )
            logger.info("TemporalSmoother initialized")
    
    def process_frame(
        self,
        frame: np.ndarray,
        frame_metadata: FrameMetadata,
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        face_landmarks: Optional[np.ndarray] = None,
        face_confidence: Optional[float] = None
    ) -> ProcessedFrame:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input frame
            frame_metadata: Frame metadata
            face_bbox: Optional face bounding box
            face_landmarks: Optional facial landmarks
            face_confidence: Optional face detection confidence
            
        Returns:
            ProcessedFrame object
        """
        start_time = time.time()
        applied_steps = []
        
        if not self.enabled:
            # Pipeline disabled, return original frame
            return ProcessedFrame(
                frame=frame,
                frame_metadata=frame_metadata,
                face_data=None,
                is_valid=True,
                preprocessing_time_ms=0.0
            )
        
        processed_frame = frame.copy()
        
        try:
            # Step 1: Video Normalization
            if self.normalizer is not None:
                processed_frame = self.normalizer.normalize_frame(
                    processed_frame,
                    source_fps=frame_metadata.fps
                )
                applied_steps.append("normalization")
            
            # Create face data object
            face_data = None
            if face_bbox is not None and face_landmarks is not None:
                face_data = FaceData(
                    bbox=face_bbox,
                    landmarks=face_landmarks,
                    confidence=face_confidence or 1.0
                )
            
            # Step 4: Frame Quality Filter
            if self.quality_filter is not None and face_data is not None:
                is_valid, rejection_reason = self.quality_filter.is_frame_valid(
                    processed_frame,
                    face_bbox=face_data.bbox,
                    face_confidence=face_data.confidence,
                    face_landmarks=face_data.landmarks
                )
                
                if not is_valid:
                    self.stats.filtered_frames += 1
                    processing_time = (time.time() - start_time) * 1000
                    
                    return ProcessedFrame(
                        frame=processed_frame,
                        frame_metadata=frame_metadata,
                        face_data=face_data,
                        is_valid=False,
                        rejection_reason=rejection_reason,
                        preprocessing_time_ms=processing_time,
                        applied_steps=applied_steps
                    )
                
                applied_steps.append("quality_filter")
            
            # Step 2: Face Stabilization
            if self.stabilizer is not None and face_data is not None:
                processed_frame = self.stabilizer.stabilize(
                    processed_frame,
                    face_data.landmarks
                )
                applied_steps.append("stabilization")
            
            # Step 3: Lighting Correction
            if self.lighting_corrector is not None:
                processed_frame = self.lighting_corrector.correct_lighting(
                    processed_frame,
                    face_bbox=face_data.bbox if face_data else None
                )
                applied_steps.append("lighting_correction")
            
            # Step 5: Face Alignment
            if self.aligner is not None and face_data is not None:
                processed_frame = self.aligner.align_face(
                    processed_frame,
                    face_data.landmarks,
                    face_bbox=face_data.bbox
                )
                applied_steps.append("alignment")
            
            # Update statistics
            self.stats.valid_frames += 1
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessedFrame(
                frame=processed_frame,
                frame_metadata=frame_metadata,
                face_data=face_data,
                is_valid=True,
                preprocessing_time_ms=processing_time,
                applied_steps=applied_steps
            )
        
        except Exception as e:
            logger.error(f"Error processing frame {frame_metadata.frame_number}: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessedFrame(
                frame=frame,
                frame_metadata=frame_metadata,
                face_data=None,
                is_valid=False,
                rejection_reason=f"processing_error: {str(e)}",
                preprocessing_time_ms=processing_time,
                applied_steps=applied_steps
            )
    
    def smooth_emotion_prediction(
        self,
        probabilities: np.ndarray,
        label: str,
        confidence: float
    ) -> Tuple[np.ndarray, str, float]:
        """
        Apply temporal smoothing to emotion prediction.
        
        Args:
            probabilities: Emotion probabilities
            label: Predicted emotion label
            confidence: Prediction confidence
            
        Returns:
            Tuple of (smoothed_probabilities, smoothed_label, smoothed_confidence)
        """
        if self.temporal_smoother is None or not self.enabled:
            return probabilities, label, confidence
        
        return self.temporal_smoother.smooth_prediction(
            probabilities, label, confidence
        )
    
    def get_statistics(self) -> PipelineStatistics:
        """
        Get pipeline statistics.
        
        Returns:
            PipelineStatistics object
        """
        # Update statistics from quality filter if available
        if self.quality_filter is not None:
            filter_stats = self.quality_filter.get_statistics()
            self.stats.filter_reasons = filter_stats['filter_reasons']
        
        return self.stats
    
    def reset(self):
        """Reset pipeline state for new video."""
        if self.normalizer:
            self.normalizer.reset()
        if self.stabilizer:
            self.stabilizer.reset()
        if self.temporal_smoother:
            self.temporal_smoother.reset()
        if self.quality_filter:
            self.quality_filter.reset()
        
        self.stats = PipelineStatistics(
            total_frames=0,
            valid_frames=0,
            filtered_frames=0
        )
        
        logger.info("Pipeline reset")
    
    def validate_configuration(self) -> bool:
        """
        Validate pipeline configuration.
        
        Returns:
            True if configuration is valid
        """
        try:
            self.config.validate()
            return True
        except ValueError as e:
            logger.error(f"Invalid configuration: {e}")
            return False
