"""
Dual Person Coordinator for Dual Person Comparison System.

This module coordinates the analysis of two people simultaneously - one from
camera and one from screen capture. It manages parallel processing, frame
synchronization, and provides unified results.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 5.1, 7.1, 7.2
"""

import time
import cv2
import numpy as np
from typing import Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, Future
from threading import Lock
import uuid
from datetime import datetime

from src.inference.face_detector import FaceDetector
from src.inference.preprocessor import FacePreprocessor
from src.inference.emotion_classifier import EmotionClassifier
from src.video_analysis.attention_detector import AttentionDetector
from src.utils.screen_capture import ScreenCapture
from src.video_analysis.person_analyzer import PersonAnalyzer
from src.video_analysis.comparison_engine import ComparisonEngine
from src.video_analysis.score_calculator import ScoreCalculator
from src.video_analysis.dual_person_models import (
    DualPersonResult,
    ComparisonResult,
    SessionReport
)
from src.video_analysis.appearance.config import AppearanceConfig
from src.video_analysis.appearance.coordinator import AppearanceAssessmentCoordinator


class DualPersonCoordinator:
    """Coordinate dual person comparison analysis.
    
    Manages parallel processing of two video sources (camera and screen capture),
    coordinates PersonAnalyzers for each source, performs comparisons, and
    provides unified statistics and results.
    
    Features:
    - Parallel processing with ThreadPoolExecutor
    - Frame synchronization between sources
    - Screen capture retry logic (3 attempts)
    - Automatic fallback to single mode on screen capture failure
    - FPS monitoring and optimization
    - Statistics collection and comparison
    
    Requirements:
    - 1.1: Initialize both camera and screen capture sources
    - 1.2: Display split view with two video feeds
    - 1.3: Handle screen capture failures with retry and fallback
    - 1.4: Maintain >= 15 FPS
    - 1.5: Cleanup resources on stop
    - 5.1: Generate comparison reports
    - 7.1: Detect faces in both sources
    - 7.2: Track largest face in secondary source
    """
    
    # Performance targets
    TARGET_FPS = 15
    MAX_LATENCY_MS = 100
    LOW_FPS_WARNING_THRESHOLD = 10
    LOW_FPS_WARNING_DURATION = 5.0  # seconds
    
    # Screen capture retry settings
    SCREEN_CAPTURE_RETRIES = 3
    SCREEN_CAPTURE_RETRY_DELAY = 0.5  # seconds
    
    # Performance optimization settings
    MAX_SECONDARY_FRAME_SKIP = 3
    MIN_SECONDARY_FRAME_SKIP = 1
    SCREEN_CAPTURE_SCALE_FACTOR = 0.5  # Scale down to 50% if performance issues
    
    def __init__(
        self,
        face_detector: FaceDetector,
        preprocessor: FacePreprocessor,
        emotion_classifier: EmotionClassifier,
        attention_detector: AttentionDetector,
        screen_capture: ScreenCapture,
        camera_device_id: int = 0,
        comparison_update_interval: float = 10.0,
        enable_optimization: bool = True,
        appearance_config: Optional[AppearanceConfig] = None
    ):
        """Initialize DualPersonCoordinator.
        
        Args:
            face_detector: FaceDetector instance (shared between analyzers)
            preprocessor: FacePreprocessor instance (shared)
            emotion_classifier: EmotionClassifier instance (shared)
            attention_detector: AttentionDetector instance (shared)
            screen_capture: ScreenCapture instance for secondary source
            camera_device_id: Camera device ID for primary source
            comparison_update_interval: Interval for comparison updates (seconds)
            enable_optimization: Enable performance optimizations
            appearance_config: Optional AppearanceConfig for appearance assessment
        """
        # Shared components
        self.face_detector = face_detector
        self.preprocessor = preprocessor
        self.emotion_classifier = emotion_classifier
        self.attention_detector = attention_detector
        self.screen_capture = screen_capture
        
        # Configuration
        self.camera_device_id = camera_device_id
        self.comparison_update_interval = comparison_update_interval
        self.enable_optimization = enable_optimization
        self.appearance_config = appearance_config
        
        # Appearance assessment coordinator
        self.appearance_coordinator: Optional[AppearanceAssessmentCoordinator] = None
        if appearance_config is not None and (appearance_config.lighting_enabled or appearance_config.clothing_enabled):
            self.appearance_coordinator = AppearanceAssessmentCoordinator(appearance_config)
            print("✓ Appearance assessment coordinator initialized")
        else:
            print("  Appearance assessment disabled")
        
        # Video sources
        self.camera = None
        self.camera_active = False
        self.screen_capture_active = False
        
        # Analyzers
        self.primary_analyzer: Optional[PersonAnalyzer] = None
        self.secondary_analyzer: Optional[PersonAnalyzer] = None
        
        # Comparison engine
        self.comparison_engine = ComparisonEngine(
            update_interval=comparison_update_interval
        )
        
        # Thread pool for parallel processing
        self.executor: Optional[ThreadPoolExecutor] = None
        
        # Frame synchronization
        self.frame_lock = Lock()
        self.primary_frame: Optional[np.ndarray] = None
        self.secondary_frame: Optional[np.ndarray] = None
        
        # Performance tracking
        self.frame_count = 0
        self.fps = 0.0
        self.fps_start_time = 0.0
        self.fps_frame_count = 0
        self.processing_times = []
        
        # Screen capture failure tracking
        self.screen_capture_failures = 0
        self.screen_capture_consecutive_failures = 0
        self.fallback_to_single_mode = False
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.session_start_time: Optional[datetime] = None
        self.session_end_time: Optional[datetime] = None
        
        # Comparison history tracking
        self.comparison_history = []
        
        # Frame skip for optimization
        self.secondary_frame_skip = 1
        self.secondary_frame_counter = 0
        
        # Resolution scaling for screen capture
        self.screen_capture_scale_factor = 1.0
        self.screen_capture_scaled = False
        
        # Face detection caching
        self.face_detection_cache = {}
        self.cache_ttl = 0.1  # Cache valid for 100ms
        
        # Low FPS tracking
        self.low_fps_start_time = None
        self.low_fps_warning_shown = False
        
        # Performance metrics for debug mode
        self.performance_metrics = {
            'frame_times': [],
            'detection_times': [],
            'emotion_times': [],
            'attention_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        print("DualPersonCoordinator initialized")
        print(f"  Camera device: {camera_device_id}")
        print(f"  Comparison interval: {comparison_update_interval}s")
        print(f"  Optimization enabled: {enable_optimization}")
    
    def start(self) -> bool:
        """Start dual person comparison mode.
        
        Initializes camera capture and screen capture sources, creates
        PersonAnalyzers, and starts the thread pool for parallel processing.
        
        Returns:
            True if successful, False if initialization failed
        
        Requirements:
        - 1.1: Initialize both camera and screen capture sources
        """
        print("Starting dual person comparison mode...")
        
        # Initialize camera
        try:
            self.camera = cv2.VideoCapture(self.camera_device_id)
            if not self.camera.isOpened():
                print(f"Error: Could not open camera {self.camera_device_id}")
                return False
            
            # Set camera properties for performance
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            self.camera_active = True
            print("✓ Camera initialized")
        
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
        
        # Test screen capture
        try:
            test_frame = self.screen_capture.capture_current()
            if test_frame is None:
                print("Error: Screen capture returned None")
                self.camera.release()
                self.camera_active = False
                return False
            
            self.screen_capture_active = True
            print("✓ Screen capture initialized")
        
        except Exception as e:
            print(f"Error initializing screen capture: {e}")
            self.camera.release()
            self.camera_active = False
            return False
        
        # Create score calculators (separate for each person)
        primary_score_calculator = ScoreCalculator(window_seconds=60)
        secondary_score_calculator = ScoreCalculator(window_seconds=60)
        
        # Create PersonAnalyzers
        self.primary_analyzer = PersonAnalyzer(
            person_id="primary",
            face_detector=self.face_detector,
            preprocessor=self.preprocessor,
            emotion_classifier=self.emotion_classifier,
            attention_detector=self.attention_detector,
            score_calculator=primary_score_calculator,
            appearance_coordinator=self.appearance_coordinator
        )
        
        self.secondary_analyzer = PersonAnalyzer(
            person_id="secondary",
            face_detector=self.face_detector,
            preprocessor=self.preprocessor,
            emotion_classifier=self.emotion_classifier,
            attention_detector=self.attention_detector,
            score_calculator=secondary_score_calculator,
            appearance_coordinator=self.appearance_coordinator
        )
        
        # Enable face detection caching for performance
        if self.enable_optimization:
            self.primary_analyzer.enable_detection_caching(cache_ttl=0.1)
            self.secondary_analyzer.enable_detection_caching(cache_ttl=0.1)
            print("✓ Face detection caching enabled")
        
        print("✓ PersonAnalyzers created")
        
        # Create thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)
        print("✓ Thread pool created")
        
        # Initialize session
        self.session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now()
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        
        # Reset comparison engine
        self.comparison_engine.reset()
        
        print("✓ Dual person comparison mode started successfully")
        return True
    
    def stop(self):
        """Stop dual person comparison mode and cleanup resources.
        
        Releases camera, stops thread pool, and cleans up all resources.
        
        Requirements:
        - 1.5: Cleanup resources on stop
        """
        print("Stopping dual person comparison mode...")
        
        # Mark session end time
        self.session_end_time = datetime.now()
        
        # Stop camera
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.camera_active = False
            print("✓ Camera released")
        
        # Screen capture doesn't need explicit cleanup (uses context manager)
        self.screen_capture_active = False
        
        # Shutdown thread pool
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None
            print("✓ Thread pool shutdown")
        
        # Reset analyzers
        if self.primary_analyzer is not None:
            self.primary_analyzer = None
        if self.secondary_analyzer is not None:
            self.secondary_analyzer = None
        
        print("✓ Dual person comparison mode stopped")
    
    def process_frame(self) -> Optional[DualPersonResult]:
        """Process one frame from both sources.
        
        Captures frames from camera and screen, processes them in parallel
        using PersonAnalyzers, and returns combined results with comparison.
        
        Returns:
            DualPersonResult with results from both sources and comparison,
            or None if processing failed
        
        Requirements:
        - 1.4: Maintain >= 15 FPS
        - 7.1: Detect faces in both sources
        - 7.2: Track largest face in secondary source
        """
        if not self.camera_active:
            return None
        
        start_time = time.perf_counter()
        
        # Capture primary frame (camera)
        ret, primary_frame = self.camera.read()
        if not ret or primary_frame is None:
            print("Warning: Failed to capture camera frame")
            return None
        
        # Capture secondary frame (screen) with retry logic
        secondary_frame = self._capture_screen_with_retry()
        
        # Apply resolution scaling if enabled
        if secondary_frame is not None and self.screen_capture_scaled:
            secondary_frame = self._scale_frame(secondary_frame, self.screen_capture_scale_factor)
        
        # Check if we should fallback to single mode
        if secondary_frame is None and self.fallback_to_single_mode:
            print("Warning: Operating in single mode (screen capture failed)")
            # Continue with only primary frame
        
        # Store frames for synchronization
        with self.frame_lock:
            self.primary_frame = primary_frame.copy()
            self.secondary_frame = secondary_frame.copy() if secondary_frame is not None else None
        
        # Process frames in parallel
        if secondary_frame is not None and not self.fallback_to_single_mode:
            # Parallel processing
            primary_future = self.executor.submit(
                self.primary_analyzer.process_frame,
                primary_frame
            )
            
            # Apply frame skip for secondary if optimization enabled
            process_secondary = True
            if self.enable_optimization:
                self.secondary_frame_counter += 1
                if self.secondary_frame_counter % self.secondary_frame_skip != 0:
                    process_secondary = False
            
            if process_secondary:
                secondary_future = self.executor.submit(
                    self.secondary_analyzer.process_frame,
                    secondary_frame
                )
            else:
                secondary_future = None
            
            # Wait for results
            try:
                primary_result = primary_future.result(timeout=0.2)
                
                if secondary_future is not None:
                    secondary_result = secondary_future.result(timeout=0.2)
                else:
                    # Use last result if skipping frame
                    secondary_result = self._get_last_secondary_result()
            
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                return None
        
        else:
            # Single mode - only process primary
            primary_result = self.primary_analyzer.process_frame(primary_frame)
            secondary_result = self._get_empty_secondary_result()
        
        # Update frame count
        self.frame_count += 1
        
        # Calculate comparison
        comparison = self._calculate_comparison(primary_result, secondary_result)
        
        # Track comparison history if a new comparison was generated
        if comparison is not None and (
            not self.comparison_history or 
            comparison.timestamp != self.comparison_history[-1].timestamp
        ):
            self.comparison_history.append(comparison)
        
        # Create combined result
        result = DualPersonResult(
            frame_number=self.frame_count,
            timestamp=time.time(),
            primary=primary_result,
            secondary=secondary_result,
            comparison=comparison
        )
        
        # Track processing time
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000
        self.processing_times.append(processing_time_ms)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
        
        # Update FPS
        self._update_fps()
        
        # Check performance and optimize if needed
        if self.enable_optimization:
            self._optimize_performance()
        
        return result
    
    def _capture_screen_with_retry(self) -> Optional[np.ndarray]:
        """Capture screen with retry logic.
        
        Attempts to capture screen up to SCREEN_CAPTURE_RETRIES times
        with delay between attempts.
        
        Returns:
            Captured frame or None if all attempts failed
        
        Requirements:
        - 1.3: Handle screen capture failures with retry
        """
        for attempt in range(self.SCREEN_CAPTURE_RETRIES):
            try:
                frame = self.screen_capture.capture_current()
                
                if frame is not None:
                    # Success - reset failure counters
                    self.screen_capture_consecutive_failures = 0
                    return frame
            
            except Exception as e:
                print(f"Screen capture attempt {attempt + 1} failed: {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.SCREEN_CAPTURE_RETRIES - 1:
                time.sleep(self.SCREEN_CAPTURE_RETRY_DELAY)
        
        # All attempts failed
        self.screen_capture_failures += 1
        self.screen_capture_consecutive_failures += 1
        
        # Fallback to single mode after 5 consecutive failures
        if self.screen_capture_consecutive_failures >= 5:
            print("Warning: Too many screen capture failures, falling back to single mode")
            self.fallback_to_single_mode = True
        
        return None
    
    def _calculate_comparison(
        self,
        primary_result,
        secondary_result
    ) -> Optional[ComparisonResult]:
        """Calculate comparison between primary and secondary results.
        
        Args:
            primary_result: PersonResult for primary user
            secondary_result: PersonResult for secondary user
        
        Returns:
            ComparisonResult or None if comparison not ready
        
        Requirements:
        - 5.1: Generate comparison reports
        """
        return self.comparison_engine.compare(
            primary_emotion_score=primary_result.emotion_score,
            primary_attention_score=primary_result.attention_score_avg,
            secondary_emotion_score=secondary_result.emotion_score,
            secondary_attention_score=secondary_result.attention_score_avg
        )
    
    def _update_fps(self):
        """Update FPS calculation."""
        self.fps_frame_count += 1
        
        current_time = time.time()
        elapsed = current_time - self.fps_start_time
        
        if elapsed >= 1.0:  # Update every second
            self.fps = self.fps_frame_count / elapsed
            self.fps_start_time = current_time
            self.fps_frame_count = 0
    
    def _optimize_performance(self):
        """Optimize performance based on FPS.
        
        Implements multiple optimization strategies:
        1. Adjust secondary frame skip
        2. Scale down screen capture resolution
        3. Track low FPS warnings
        
        Requirements:
        - 1.4: Maintain >= 15 FPS
        """
        current_time = time.time()
        
        # Check for low FPS
        if self.fps < self.LOW_FPS_WARNING_THRESHOLD and self.fps > 0:
            if self.low_fps_start_time is None:
                self.low_fps_start_time = current_time
            
            low_fps_duration = current_time - self.low_fps_start_time
            
            if low_fps_duration >= self.LOW_FPS_WARNING_DURATION and not self.low_fps_warning_shown:
                print(f"⚠ Warning: Low FPS detected ({self.fps:.1f} FPS) for {low_fps_duration:.1f}s")
                self.low_fps_warning_shown = True
        else:
            # Reset low FPS tracking
            self.low_fps_start_time = None
            self.low_fps_warning_shown = False
        
        # Optimization strategy based on FPS
        if self.fps < self.TARGET_FPS and self.fps > 0:
            # Strategy 1: Increase frame skip for secondary source
            if self.secondary_frame_skip < self.MAX_SECONDARY_FRAME_SKIP:
                self.secondary_frame_skip += 1
                print(f"⚡ Performance optimization: secondary_frame_skip = {self.secondary_frame_skip}")
            
            # Strategy 2: Scale down screen capture resolution
            elif not self.screen_capture_scaled:
                self.screen_capture_scaled = True
                self.screen_capture_scale_factor = self.SCREEN_CAPTURE_SCALE_FACTOR
                print(f"⚡ Performance optimization: screen capture scaled to {self.screen_capture_scale_factor * 100:.0f}%")
        
        elif self.fps > self.TARGET_FPS + 5:
            # We have headroom - restore quality
            
            # First, restore resolution if scaled
            if self.screen_capture_scaled:
                self.screen_capture_scaled = False
                self.screen_capture_scale_factor = 1.0
                print(f"⚡ Performance restored: screen capture at 100%")
            
            # Then, decrease frame skip
            elif self.secondary_frame_skip > self.MIN_SECONDARY_FRAME_SKIP:
                self.secondary_frame_skip -= 1
                print(f"⚡ Performance restored: secondary_frame_skip = {self.secondary_frame_skip}")
    
    def _get_last_secondary_result(self):
        """Get last secondary result (for frame skipping)."""
        # Create a dummy result with current scores
        from src.video_analysis.dual_person_models import PersonResult
        
        return PersonResult(
            person_id="secondary",
            frame_number=self.frame_count,
            timestamp=time.time(),
            face_detected=False,
            emotion_score=self.secondary_analyzer.calculate_emotion_score() if self.secondary_analyzer else 0.0,
            attention_score_avg=self.secondary_analyzer.calculate_attention_score() if self.secondary_analyzer else 0.0
        )
    
    def _get_empty_secondary_result(self):
        """Get empty secondary result (for single mode)."""
        from src.video_analysis.dual_person_models import PersonResult
        
        return PersonResult(
            person_id="secondary",
            frame_number=self.frame_count,
            timestamp=time.time(),
            face_detected=False,
            emotion_score=0.0,
            attention_score_avg=0.0
        )
    
    def get_statistics(self) -> Dict:
        """Get current statistics for both users.
        
        Returns:
            Dictionary with statistics for primary and secondary users
        """
        stats = {
            'session_id': self.session_id,
            'frame_count': self.frame_count,
            'fps': self.fps,
            'fallback_mode': self.fallback_to_single_mode,
            'screen_capture_failures': self.screen_capture_failures,
            'average_processing_time_ms': (
                sum(self.processing_times) / len(self.processing_times)
                if self.processing_times else 0.0
            ),
            'performance_optimizations': {
                'secondary_frame_skip': self.secondary_frame_skip,
                'screen_capture_scaled': self.screen_capture_scaled,
                'screen_capture_scale_factor': self.screen_capture_scale_factor
            }
        }
        
        if self.primary_analyzer is not None:
            stats['primary'] = self.primary_analyzer.get_statistics()
            stats['primary']['cache_stats'] = self.primary_analyzer.get_cache_stats()
        
        if self.secondary_analyzer is not None:
            stats['secondary'] = self.secondary_analyzer.get_statistics()
            stats['secondary']['cache_stats'] = self.secondary_analyzer.get_cache_stats()
        
        # Add appearance assessment statistics if enabled
        if self.appearance_coordinator is not None:
            stats['appearance_assessment'] = {
                'enabled': True,
                'lighting_enabled': self.appearance_config.lighting_enabled if self.appearance_config else False,
                'clothing_enabled': self.appearance_config.clothing_enabled if self.appearance_config else False
            }
            
            # Add appearance metrics from analyzers if available
            if self.primary_analyzer is not None and hasattr(self.primary_analyzer, 'appearance_history'):
                if self.primary_analyzer.appearance_history:
                    # Calculate average appearance scores
                    appearance_scores = [a.overall_score for a in self.primary_analyzer.appearance_history if a is not None]
                    if appearance_scores:
                        stats['primary']['appearance'] = {
                            'average_overall_score': sum(appearance_scores) / len(appearance_scores),
                            'min_score': min(appearance_scores),
                            'max_score': max(appearance_scores),
                            'sample_count': len(appearance_scores)
                        }
                        
                        # Add lighting-specific stats if enabled
                        if self.appearance_config and self.appearance_config.lighting_enabled:
                            lighting_scores = [a.lighting.score for a in self.primary_analyzer.appearance_history 
                                             if a is not None and a.lighting is not None]
                            if lighting_scores:
                                stats['primary']['appearance']['lighting'] = {
                                    'average_score': sum(lighting_scores) / len(lighting_scores),
                                    'min_score': min(lighting_scores),
                                    'max_score': max(lighting_scores)
                                }
                        
                        # Add clothing-specific stats if enabled
                        if self.appearance_config and self.appearance_config.clothing_enabled:
                            clothing_scores = [a.clothing.score for a in self.primary_analyzer.appearance_history 
                                             if a is not None and a.clothing is not None]
                            if clothing_scores:
                                stats['primary']['appearance']['clothing'] = {
                                    'average_score': sum(clothing_scores) / len(clothing_scores),
                                    'min_score': min(clothing_scores),
                                    'max_score': max(clothing_scores)
                                }
            
            if self.secondary_analyzer is not None and hasattr(self.secondary_analyzer, 'appearance_history'):
                if self.secondary_analyzer.appearance_history:
                    # Calculate average appearance scores
                    appearance_scores = [a.overall_score for a in self.secondary_analyzer.appearance_history if a is not None]
                    if appearance_scores:
                        stats['secondary']['appearance'] = {
                            'average_overall_score': sum(appearance_scores) / len(appearance_scores),
                            'min_score': min(appearance_scores),
                            'max_score': max(appearance_scores),
                            'sample_count': len(appearance_scores)
                        }
                        
                        # Add lighting-specific stats if enabled
                        if self.appearance_config and self.appearance_config.lighting_enabled:
                            lighting_scores = [a.lighting.score for a in self.secondary_analyzer.appearance_history 
                                             if a is not None and a.lighting is not None]
                            if lighting_scores:
                                stats['secondary']['appearance']['lighting'] = {
                                    'average_score': sum(lighting_scores) / len(lighting_scores),
                                    'min_score': min(lighting_scores),
                                    'max_score': max(lighting_scores)
                                }
                        
                        # Add clothing-specific stats if enabled
                        if self.appearance_config and self.appearance_config.clothing_enabled:
                            clothing_scores = [a.clothing.score for a in self.secondary_analyzer.appearance_history 
                                             if a is not None and a.clothing is not None]
                            if clothing_scores:
                                stats['secondary']['appearance']['clothing'] = {
                                    'average_score': sum(clothing_scores) / len(clothing_scores),
                                    'min_score': min(clothing_scores),
                                    'max_score': max(clothing_scores)
                                }
        else:
            stats['appearance_assessment'] = {
                'enabled': False
            }
        
        return stats
    
    def get_comparison(self) -> Optional[ComparisonResult]:
        """Get latest comparison results.
        
        Returns:
            Latest ComparisonResult or None if no comparison available
        """
        return self.comparison_engine.get_last_comparison()
    
    def reset_statistics(self):
        """Reset all statistics.
        
        Clears statistics for both analyzers and resets counters.
        Useful when starting a new session.
        """
        print("Resetting statistics...")
        
        # Reset analyzers
        if self.primary_analyzer is not None:
            self.primary_analyzer.reset()
        
        if self.secondary_analyzer is not None:
            self.secondary_analyzer.reset()
        
        # Reset comparison engine
        self.comparison_engine.reset()
        
        # Reset counters
        self.frame_count = 0
        self.fps = 0.0
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.processing_times.clear()
        self.screen_capture_failures = 0
        self.screen_capture_consecutive_failures = 0
        self.fallback_to_single_mode = False
        
        # Reset performance optimization state
        self.secondary_frame_skip = 1
        self.secondary_frame_counter = 0
        self.screen_capture_scale_factor = 1.0
        self.screen_capture_scaled = False
        self.low_fps_start_time = None
        self.low_fps_warning_shown = False
        
        # Clear caches
        self.face_detection_cache.clear()
        
        # Reset performance metrics
        self.performance_metrics = {
            'frame_times': [],
            'detection_times': [],
            'emotion_times': [],
            'attention_times': [],
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Clear comparison history
        self.comparison_history.clear()
        
        # New session
        self.session_id = str(uuid.uuid4())
        self.session_start_time = datetime.now()
        
        print("✓ Statistics reset")
    
    def get_session_report(self) -> SessionReport:
        """Generate session report for export.
        
        Returns:
            SessionReport with complete session data
        
        Requirements:
        - 5.1: Generate comparison reports
        """
        # Calculate duration
        if self.session_start_time and self.session_end_time:
            duration = (self.session_end_time - self.session_start_time).total_seconds()
        elif self.session_start_time:
            duration = (datetime.now() - self.session_start_time).total_seconds()
        else:
            duration = 0.0
        
        # Get statistics
        primary_stats = self.primary_analyzer.get_statistics() if self.primary_analyzer else {}
        secondary_stats = self.secondary_analyzer.get_statistics() if self.secondary_analyzer else {}
        
        # Get timelines
        primary_emotion_timeline = (
            self.primary_analyzer.emotion_history if self.primary_analyzer else []
        )
        primary_attention_timeline = (
            self.primary_analyzer.attention_history if self.primary_analyzer else []
        )
        secondary_emotion_timeline = (
            self.secondary_analyzer.emotion_history if self.secondary_analyzer else []
        )
        secondary_attention_timeline = (
            self.secondary_analyzer.attention_history if self.secondary_analyzer else []
        )
        
        # Get comparison history
        comparison_history = self.comparison_history.copy()
        
        # Calculate overall winners based on comparison history
        if comparison_history:
            # Count wins for each user
            emotion_wins = {'primary': 0, 'secondary': 0, 'tie': 0}
            attention_wins = {'primary': 0, 'secondary': 0, 'tie': 0}
            total_emotion_diff = 0.0
            total_attention_diff = 0.0
            
            for comp in comparison_history:
                emotion_wins[comp.emotion_winner] += 1
                attention_wins[comp.attention_winner] += 1
                total_emotion_diff += comp.emotion_difference
                total_attention_diff += comp.attention_difference
            
            # Determine overall winners
            overall_emotion_winner = max(emotion_wins.items(), key=lambda x: x[1])[0]
            overall_attention_winner = max(attention_wins.items(), key=lambda x: x[1])[0]
            
            # Calculate average differences
            average_emotion_difference = total_emotion_diff / len(comparison_history)
            average_attention_difference = total_attention_diff / len(comparison_history)
        else:
            # No comparison history - use last comparison if available
            last_comparison = self.get_comparison()
            overall_emotion_winner = last_comparison.emotion_winner if last_comparison else "tie"
            overall_attention_winner = last_comparison.attention_winner if last_comparison else "tie"
            average_emotion_difference = last_comparison.emotion_difference if last_comparison else 0.0
            average_attention_difference = last_comparison.attention_difference if last_comparison else 0.0
        
        return SessionReport(
            session_id=self.session_id,
            start_time=self.session_start_time or datetime.now(),
            end_time=self.session_end_time or datetime.now(),
            duration_seconds=duration,
            primary_stats=primary_stats,
            primary_emotion_timeline=primary_emotion_timeline,
            primary_attention_timeline=primary_attention_timeline,
            secondary_stats=secondary_stats,
            secondary_emotion_timeline=secondary_emotion_timeline,
            secondary_attention_timeline=secondary_attention_timeline,
            comparison_history=comparison_history,
            overall_emotion_winner=overall_emotion_winner,
            overall_attention_winner=overall_attention_winner,
            average_emotion_difference=average_emotion_difference,
            average_attention_difference=average_attention_difference
        )
    
    def get_current_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get current frames from both sources.
        
        Returns:
            Tuple of (primary_frame, secondary_frame)
        """
        with self.frame_lock:
            return (
                self.primary_frame.copy() if self.primary_frame is not None else None,
                self.secondary_frame.copy() if self.secondary_frame is not None else None
            )
    
    def is_active(self) -> bool:
        """Check if coordinator is active.
        
        Returns:
            True if both sources are active
        """
        return self.camera_active and (self.screen_capture_active or self.fallback_to_single_mode)
    
    def get_fps(self) -> float:
        """Get current FPS.
        
        Returns:
            Current frames per second
        """
        return self.fps
    
    def set_comparison_interval(self, interval: float):
        """Set comparison update interval.
        
        Args:
            interval: New interval in seconds
        """
        self.comparison_engine.set_update_interval(interval)
        print(f"Comparison interval updated to {interval}s")
    
    def _scale_frame(self, frame: np.ndarray, scale_factor: float) -> np.ndarray:
        """Scale frame for performance optimization.
        
        Args:
            frame: Input frame
            scale_factor: Scale factor (0.5 = 50% size)
        
        Returns:
            Scaled frame
        """
        if scale_factor == 1.0:
            return frame
        
        height, width = frame.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def _get_cached_face_detection(
        self,
        frame: np.ndarray,
        person_id: str
    ) -> Optional[list]:
        """Get cached face detection results if available.
        
        Args:
            frame: Input frame
            person_id: Person identifier for cache key
        
        Returns:
            Cached detection results or None if cache miss
        """
        current_time = time.time()
        cache_key = f"{person_id}_{self.frame_count}"
        
        # Check if cache entry exists and is still valid
        if cache_key in self.face_detection_cache:
            cached_time, cached_result = self.face_detection_cache[cache_key]
            
            if current_time - cached_time < self.cache_ttl:
                self.performance_metrics['cache_hits'] += 1
                return cached_result
        
        self.performance_metrics['cache_misses'] += 1
        return None
    
    def _cache_face_detection(
        self,
        person_id: str,
        detection_result: list
    ):
        """Cache face detection results.
        
        Args:
            person_id: Person identifier for cache key
            detection_result: Detection results to cache
        """
        current_time = time.time()
        cache_key = f"{person_id}_{self.frame_count}"
        
        # Store in cache with timestamp
        self.face_detection_cache[cache_key] = (current_time, detection_result)
        
        # Clean old cache entries (keep only last 10 frames)
        if len(self.face_detection_cache) > 10:
            oldest_key = min(self.face_detection_cache.keys())
            del self.face_detection_cache[oldest_key]
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for debug mode.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'fps': self.fps,
            'target_fps': self.TARGET_FPS,
            'frame_count': self.frame_count,
            'secondary_frame_skip': self.secondary_frame_skip,
            'screen_capture_scaled': self.screen_capture_scaled,
            'screen_capture_scale_factor': self.screen_capture_scale_factor,
            'low_fps_warning': self.low_fps_warning_shown,
            'fallback_mode': self.fallback_to_single_mode,
            'cache_hit_rate': (
                self.performance_metrics['cache_hits'] / 
                (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
                if (self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses']) > 0
                else 0.0
            )
        }
        
        # Add timing statistics if available
        if self.processing_times:
            metrics['avg_processing_time_ms'] = sum(self.processing_times) / len(self.processing_times)
            metrics['max_processing_time_ms'] = max(self.processing_times)
            metrics['min_processing_time_ms'] = min(self.processing_times)
        
        return metrics
    
    def print_performance_metrics(self):
        """Print performance metrics to console (debug mode)."""
        metrics = self.get_performance_metrics()
        
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        print(f"FPS: {metrics['fps']:.1f} / {metrics['target_fps']} (target)")
        print(f"Frame Count: {metrics['frame_count']}")
        print(f"Secondary Frame Skip: {metrics['secondary_frame_skip']}")
        print(f"Screen Capture Scaled: {metrics['screen_capture_scaled']} ({metrics['screen_capture_scale_factor']*100:.0f}%)")
        print(f"Low FPS Warning: {metrics['low_fps_warning']}")
        print(f"Fallback Mode: {metrics['fallback_mode']}")
        print(f"Cache Hit Rate: {metrics['cache_hit_rate']*100:.1f}%")
        
        if 'avg_processing_time_ms' in metrics:
            print(f"\nProcessing Times:")
            print(f"  Average: {metrics['avg_processing_time_ms']:.2f} ms")
            print(f"  Min: {metrics['min_processing_time_ms']:.2f} ms")
            print(f"  Max: {metrics['max_processing_time_ms']:.2f} ms")
        
        print("="*60 + "\n")
