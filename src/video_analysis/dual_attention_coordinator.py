"""
Dual Attention Coordinator

Coordinates all components of the dual attention detection system:
- ScreenMonitor: Monitors screen for face presence
- ScoreFusion: Combines camera and screen scores
- DistractionTracker: Tracks distraction duration
- AlertManager: Manages alert display

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import threading
import tracemalloc
from collections import deque
from datetime import datetime

from src.video_analysis.attention_detector import AttentionDetector
from src.video_analysis.screen_monitor import ScreenMonitor, ScreenMonitorResult
from src.video_analysis.score_fusion import ScoreFusion, FusedScore
from src.video_analysis.distraction_tracker import DistractionTracker, DistractionState
from src.video_analysis.alert_manager import AlertManager
from src.inference.face_detector import FaceDetector


@dataclass
class DualAttentionResult:
    """
    Complete result from dual attention detection system.
    
    Attributes:
        fused_score: Combined score from camera and screen
        distraction_state: Current distraction tracking state
        should_show_alert: Whether alert should be displayed
        alert_message: Alert message text (if applicable)
        timestamp: Time when result was generated
    """
    fused_score: FusedScore
    distraction_state: DistractionState
    should_show_alert: bool
    alert_message: Optional[str]
    timestamp: float


class DualAttentionCoordinator:
    """
    Coordinates all components of the dual attention detection system.
    
    This is the main entry point for dual attention detection. It:
    1. Manages lifecycle of all sub-components
    2. Coordinates per-frame processing
    3. Aggregates statistics from all components
    4. Handles errors gracefully
    
    Processing Flow:
        Camera Frame → Camera Score → Screen Result → Score Fusion →
        Distraction Tracking → Alert Management → Result
    
    Example:
        >>> coordinator = DualAttentionCoordinator(
        ...     camera_detector=attention_detector,
        ...     face_detector=face_detector,
        ...     enable_screen_monitor=True
        ... )
        >>> coordinator.start()
        >>> result = coordinator.process_frame(frame, landmarks)
        >>> if result.should_show_alert:
        ...     frame = alert_manager.draw_alert(frame, result.distraction_state.distraction_duration)
        >>> coordinator.stop()
    """
    
    def __init__(
        self,
        camera_detector: AttentionDetector,
        face_detector: FaceDetector,
        enable_screen_monitor: bool = True,
        camera_weight: float = 0.6,
        screen_weight: float = 0.4,
        distraction_threshold: float = 7.0,  # Thang 0-10 (thay vì 60.0 trên thang 0-100)
        alert_threshold: float = 5.0,
        alert_cooldown: float = 10.0,
        enable_performance_monitoring: bool = True,
        fps_threshold: float = 20.0,
        fps_check_duration: float = 5.0,
        memory_threshold_mb: float = 150.0,
        memory_check_duration: float = 10.0,
        enable_profiling: bool = False
    ):
        """
        Initialize DualAttentionCoordinator.
        
        Args:
            camera_detector: AttentionDetector for camera-based attention detection
            face_detector: FaceDetector for detecting faces in screen captures
            enable_screen_monitor: Whether to enable screen monitoring
            camera_weight: Weight for camera score in fusion (default: 0.6)
            screen_weight: Weight for screen score in fusion (default: 0.4)
            distraction_threshold: Score below this is considered distracted (default: 60.0)
            alert_threshold: Seconds of distraction before alert (default: 5.0)
            alert_cooldown: Seconds between alerts (default: 10.0)
            enable_performance_monitoring: Enable performance monitoring (default: True)
            fps_threshold: Minimum FPS before auto-disable (default: 20.0)
            fps_check_duration: Duration to check FPS before disable (default: 5.0s)
            memory_threshold_mb: Maximum memory usage in MB (default: 150.0)
            memory_check_duration: Duration to check memory before disable (default: 10.0s)
            enable_profiling: Enable detailed profiling mode (default: False)
        """
        self.camera_detector = camera_detector
        self.face_detector = face_detector
        self.enable_screen_monitor = enable_screen_monitor
        
        # Initialize sub-components
        self.screen_monitor: Optional[ScreenMonitor] = None
        if enable_screen_monitor:
            try:
                self.screen_monitor = ScreenMonitor(
                    face_detector=face_detector,
                    capture_interval=0.5,
                    confidence_threshold=0.85,
                    monitor_number=1
                )
                print("ScreenMonitor initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize ScreenMonitor: {e}")
                print("Continuing with camera-only mode")
                self.screen_monitor = None
        
        self.score_fusion = ScoreFusion(
            camera_weight=camera_weight,
            screen_weight=screen_weight
        )
        
        self.distraction_tracker = DistractionTracker(
            distraction_threshold=distraction_threshold,
            alert_threshold=alert_threshold,
            alert_cooldown=alert_cooldown
        )
        
        self.alert_manager = AlertManager(
            alert_duration=0.0,  # Dismiss immediately when refocused
            banner_size=(400, 80),
            fade_in_duration=0.3,
            fade_out_duration=0.3  # Quick fade out
        )
        
        # State tracking
        self._running = False
        self._lock = threading.Lock()
        
        # Error tracking
        self._camera_error_count = 0
        self._screen_error_count = 0
        self._max_consecutive_errors = 30
        
        # Statistics
        self._total_frames_processed = 0
        self._frames_with_camera = 0
        self._frames_with_screen = 0
        self._frames_with_both = 0
        
        # Performance monitoring
        self.enable_performance_monitoring = enable_performance_monitoring
        self.fps_threshold = fps_threshold
        self.fps_check_duration = fps_check_duration
        self.memory_threshold_mb = memory_threshold_mb
        self.memory_check_duration = memory_check_duration
        self.enable_profiling = enable_profiling
        
        # FPS tracking with moving average
        self._fps_window_size = 30  # Track last 30 frames
        self._frame_times: deque = deque(maxlen=self._fps_window_size)
        self._current_fps = 0.0
        self._last_frame_time = None
        
        # Low FPS tracking
        self._low_fps_start_time: Optional[float] = None
        self._fps_warnings_logged = 0
        
        # Memory tracking
        self._memory_snapshots: deque = deque(maxlen=100)  # Track last 100 snapshots
        self._current_memory_mb = 0.0
        self._baseline_memory_mb = 0.0
        self._high_memory_start_time: Optional[float] = None
        self._memory_warnings_logged = 0
        
        # Processing time tracking
        self._processing_times: deque = deque(maxlen=100)
        self._avg_processing_time_ms = 0.0
        
        # Performance state
        self._performance_disabled = False
        self._disable_reason: Optional[str] = None
        
        # Start memory tracking if enabled
        if self.enable_performance_monitoring:
            tracemalloc.start()
            # Get baseline memory
            current, peak = tracemalloc.get_traced_memory()
            self._baseline_memory_mb = current / 1024 / 1024
            print(f"Performance monitoring enabled (baseline memory: {self._baseline_memory_mb:.2f} MB)")
        
        print("DualAttentionCoordinator initialized")
        print(f"  Screen monitoring: {'enabled' if self.screen_monitor else 'disabled'}")
        print(f"  Camera weight: {camera_weight}")
        print(f"  Screen weight: {screen_weight}")
        print(f"  Distraction threshold: {distraction_threshold}")
        print(f"  Alert threshold: {alert_threshold}s")
        print(f"  Performance monitoring: {'enabled' if enable_performance_monitoring else 'disabled'}")
        if enable_performance_monitoring:
            print(f"    FPS threshold: {fps_threshold} FPS")
            print(f"    Memory threshold: {memory_threshold_mb} MB")
            print(f"    Profiling mode: {'enabled' if enable_profiling else 'disabled'}")
    
    def start(self) -> bool:
        """
        Start the dual attention detection system.
        
        This starts all sub-components, particularly the ScreenMonitor
        background thread.
        
        Returns:
            True if started successfully, False otherwise
        """
        with self._lock:
            if self._running:
                print("DualAttentionCoordinator is already running")
                return True
            
            try:
                # Start screen monitor if enabled
                if self.screen_monitor is not None:
                    if not self.screen_monitor.start():
                        print("Warning: ScreenMonitor failed to start")
                        print("Continuing with camera-only mode")
                        self.screen_monitor = None
                
                self._running = True
                print("DualAttentionCoordinator started successfully")
                return True
            
            except Exception as e:
                print(f"Error starting DualAttentionCoordinator: {e}")
                self._running = False
                return False
    
    def stop(self):
        """
        Stop the dual attention detection system.
        
        This stops all sub-components and cleans up resources.
        """
        with self._lock:
            if not self._running:
                return
            
            print("Stopping DualAttentionCoordinator...")
            
            # Stop screen monitor
            if self.screen_monitor is not None:
                try:
                    self.screen_monitor.stop()
                except Exception as e:
                    print(f"Error stopping ScreenMonitor: {e}")
            
            # Stop memory tracking
            if self.enable_performance_monitoring:
                try:
                    tracemalloc.stop()
                    print("Memory tracking stopped")
                except Exception as e:
                    print(f"Error stopping memory tracking: {e}")
            
            self._running = False
            print("DualAttentionCoordinator stopped")
    
    def is_running(self) -> bool:
        """
        Check if coordinator is currently running.
        
        Returns:
            True if running, False otherwise
        """
        with self._lock:
            return self._running
    
    def process_frame(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray]
    ) -> DualAttentionResult:
        """
        Process a single frame through the dual attention detection pipeline.
        
        Processing flow:
        1. Calculate camera attention score
        2. Get latest screen monitoring result
        3. Fuse scores from both sources
        4. Update distraction tracker
        5. Manage alert state
        6. Return complete result
        
        Args:
            frame: Video frame (BGR format)
            landmarks: Facial landmarks (None if no face detected)
        
        Returns:
            DualAttentionResult with complete detection information
        """
        # Start performance tracking
        frame_start_time = time.time()
        
        # Check if performance disabled
        if self._performance_disabled:
            # Return minimal result
            return self._create_disabled_result(frame_start_time)
        
        timestamp = frame_start_time
        self._total_frames_processed += 1
        
        # Update FPS tracking
        if self.enable_performance_monitoring:
            self._update_fps_tracking(timestamp)
        
        # Step 1: Calculate camera attention score
        camera_score, face_presence = self._get_camera_score(landmarks, frame.shape[:2])
        
        # Step 2: Get screen monitoring result
        screen_result = self._get_screen_result()
        
        # Step 3: Fuse scores
        try:
            fused_score = self.score_fusion.fuse_scores(
                camera_score=camera_score,
                face_presence=face_presence,
                screen_result=screen_result
            )
            
            # Update statistics
            if face_presence:
                self._frames_with_camera += 1
            if screen_result.screen_presence:
                self._frames_with_screen += 1
            if face_presence and screen_result.screen_presence:
                self._frames_with_both += 1
        
        except Exception as e:
            print(f"Error in score fusion: {e}")
            # Create default fused score
            fused_score = FusedScore(
                combined_score=0.0,
                camera_score=camera_score,
                screen_score=0.0,
                face_presence=face_presence,
                screen_presence=False,
                fusion_mode="none",
                timestamp=timestamp
            )
        
        # Step 4: Update distraction tracker
        try:
            # DEBUG: In ra điểm số để kiểm tra
            print(f"[DEBUG] Face: {fused_score.face_presence}, "
                  f"Camera: {fused_score.camera_score:.1f}, "
                  f"Combined: {fused_score.combined_score:.1f}, "
                  f"Mode: {fused_score.fusion_mode}")
            
            distraction_state = self.distraction_tracker.update(fused_score.combined_score)
            
            # Debug logging for distraction tracking
            print(f"[DEBUG] Distracted: {distraction_state.is_distracted}, "
                  f"Duration: {distraction_state.distraction_duration:.1f}s, "
                  f"Total: {distraction_state.total_distraction_time:.1f}s, "
                  f"State: {distraction_state.current_state.value}")
        except Exception as e:
            print(f"Error in distraction tracker: {e}")
            # Create default distraction state
            from src.video_analysis.distraction_tracker import DistractionStateEnum
            distraction_state = DistractionState(
                is_distracted=False,
                distraction_duration=0.0,
                should_alert=False,
                total_distraction_time=0.0,
                last_alert_time=None,
                current_state=DistractionStateEnum.FOCUSED
            )
        
        # Step 5: Manage alert state
        should_show_alert = False
        alert_message = None
        
        try:
            # ===== LOGIC MỚI: CHỈ cảnh báo khi KHÔNG có khuôn mặt =====
            # Nếu có khuôn mặt (face_presence = True) → KHÔNG BAO GIỜ cảnh báo
            if fused_score.face_presence:
                # CÓ khuôn mặt → TẮT mọi cảnh báo
                should_show_alert = False
                if self.alert_manager.is_alert_active():
                    self.alert_manager.dismiss_alert()
                    print("✓ Phát hiện khuôn mặt - TẮT cảnh báo")
            else:
                # KHÔNG có khuôn mặt → Kiểm tra xem có nên cảnh báo không
                if distraction_state.should_alert:
                    # Trigger alert
                    self.alert_manager.trigger_alert(distraction_state.distraction_duration)
                    should_show_alert = True
                    alert_message = f"CANH BAO: KHONG THAY KHUON MAT! ({distraction_state.distraction_duration:.1f}s)"
                
                # Check if alert is still active (fading out)
                if self.alert_manager.is_alert_active():
                    should_show_alert = True
        
        except Exception as e:
            print(f"Error in alert manager: {e}")
        
        # Step 6: Performance monitoring and checks
        if self.enable_performance_monitoring:
            # Track processing time
            processing_time_ms = (time.time() - frame_start_time) * 1000
            self._processing_times.append(processing_time_ms)
            self._avg_processing_time_ms = sum(self._processing_times) / len(self._processing_times)
            
            # Track memory usage
            self._update_memory_tracking()
            
            # Check performance thresholds
            self._check_performance_thresholds()
            
            # Log profiling info if enabled
            if self.enable_profiling and self._total_frames_processed % 30 == 0:
                self._log_profiling_info()
        
        # Step 7: Return complete result
        return DualAttentionResult(
            fused_score=fused_score,
            distraction_state=distraction_state,
            should_show_alert=should_show_alert,
            alert_message=alert_message,
            timestamp=timestamp
        )
    
    def _get_camera_score(
        self,
        landmarks: Optional[np.ndarray],
        frame_shape: tuple
    ) -> tuple[float, bool]:
        """
        Get attention score from camera detector.
        
        Args:
            landmarks: Facial landmarks (None if no face)
            frame_shape: (height, width) of frame
        
        Returns:
            Tuple of (camera_score, face_presence)
        """
        if landmarks is None:
            # No face detected
            self._camera_error_count = 0  # Reset error count
            # Gọi với face_detected=False
            try:
                camera_score, _ = self.camera_detector.calculate_attention_score(
                    None, frame_shape, face_detected=False
                )
                # CHUYỂN ĐỔI: 0-10 → 0-100
                camera_score = camera_score * 10.0
            except:
                camera_score = 0.0
            return camera_score, False
        
        try:
            # Calculate attention score - CÓ khuôn mặt
            camera_score, _ = self.camera_detector.calculate_attention_score(
                landmarks, frame_shape, face_detected=True
            )
            self._camera_error_count = 0  # Reset error count on success
            
            # CHUYỂN ĐỔI: attention_detector trả về 0-10, nhưng hệ thống cần 0-100
            camera_score_100 = camera_score * 10.0
            return camera_score_100, True
        
        except Exception as e:
            self._camera_error_count += 1
            print(f"Error calculating camera score: {e}")
            
            # Check if too many consecutive errors
            if self._camera_error_count >= self._max_consecutive_errors:
                print("Warning: Too many camera errors, may need to restart")
            
            return 0.0, False
    
    def _get_screen_result(self) -> ScreenMonitorResult:
        """
        Get latest result from screen monitor.
        
        Returns:
            ScreenMonitorResult (default if screen monitor not available)
        """
        if self.screen_monitor is None or not self.screen_monitor.is_running():
            # Screen monitor not available, return default result
            return ScreenMonitorResult(
                screen_presence=False,
                face_detected=False,
                face_bbox=None,
                confidence=0.0,
                timestamp=time.time(),
                screen_score=0.0
            )
        
        try:
            result = self.screen_monitor.get_latest_result()
            self._screen_error_count = 0  # Reset error count on success
            return result
        
        except Exception as e:
            self._screen_error_count += 1
            print(f"Error getting screen result: {e}")
            
            # Check if too many consecutive errors
            if self._screen_error_count >= self._max_consecutive_errors:
                print("Warning: Too many screen errors, disabling screen monitor")
                try:
                    self.screen_monitor.stop()
                except:
                    pass
                self.screen_monitor = None
            
            # Return default result
            return ScreenMonitorResult(
                screen_presence=False,
                face_detected=False,
                face_bbox=None,
                confidence=0.0,
                timestamp=time.time(),
                screen_score=0.0
            )
    
    def _update_fps_tracking(self, current_time: float):
        """
        Update FPS tracking with moving average.
        
        Args:
            current_time: Current timestamp
        """
        if self._last_frame_time is not None:
            frame_time = current_time - self._last_frame_time
            self._frame_times.append(frame_time)
            
            # Calculate FPS from moving average
            if len(self._frame_times) > 0:
                avg_frame_time = sum(self._frame_times) / len(self._frame_times)
                self._current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        self._last_frame_time = current_time
    
    def _update_memory_tracking(self):
        """
        Update memory usage tracking.
        """
        try:
            current, peak = tracemalloc.get_traced_memory()
            current_mb = current / 1024 / 1024
            self._current_memory_mb = current_mb - self._baseline_memory_mb
            self._memory_snapshots.append(self._current_memory_mb)
        except Exception as e:
            if self.enable_profiling:
                print(f"Error tracking memory: {e}")
    
    def _check_performance_thresholds(self):
        """
        Check if performance thresholds are exceeded and disable if necessary.
        """
        current_time = time.time()
        
        # Check FPS threshold
        if self._current_fps > 0 and self._current_fps < self.fps_threshold:
            if self._low_fps_start_time is None:
                self._low_fps_start_time = current_time
                self._log_performance_warning(
                    f"FPS dropped below threshold: {self._current_fps:.1f} < {self.fps_threshold}"
                )
            elif current_time - self._low_fps_start_time >= self.fps_check_duration:
                # FPS has been low for too long
                self._disable_due_to_performance(
                    f"FPS below {self.fps_threshold} for {self.fps_check_duration}s"
                )
                return
        else:
            # FPS is good, reset timer
            if self._low_fps_start_time is not None:
                self._log_performance_warning(
                    f"FPS recovered: {self._current_fps:.1f} FPS"
                )
            self._low_fps_start_time = None
        
        # Check memory threshold
        if self._current_memory_mb > self.memory_threshold_mb:
            if self._high_memory_start_time is None:
                self._high_memory_start_time = current_time
                self._log_performance_warning(
                    f"Memory usage exceeded threshold: {self._current_memory_mb:.1f} MB > {self.memory_threshold_mb} MB"
                )
            elif current_time - self._high_memory_start_time >= self.memory_check_duration:
                # Memory has been high for too long
                self._disable_due_to_performance(
                    f"Memory above {self.memory_threshold_mb} MB for {self.memory_check_duration}s"
                )
                return
        else:
            # Memory is good, reset timer
            if self._high_memory_start_time is not None:
                self._log_performance_warning(
                    f"Memory usage recovered: {self._current_memory_mb:.1f} MB"
                )
            self._high_memory_start_time = None
    
    def _disable_due_to_performance(self, reason: str):
        """
        Disable dual attention due to performance issues.
        
        Args:
            reason: Reason for disabling
        """
        self._performance_disabled = True
        self._disable_reason = reason
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'='*70}")
        print(f"[{timestamp}] PERFORMANCE AUTO-DISABLE")
        print(f"Reason: {reason}")
        print(f"{'='*70}\n")
        
        # Stop screen monitor to free resources
        if self.screen_monitor is not None:
            try:
                self.screen_monitor.stop()
                print("ScreenMonitor stopped to free resources")
            except Exception as e:
                print(f"Error stopping ScreenMonitor: {e}")
    
    def _log_performance_warning(self, message: str):
        """
        Log a performance warning with timestamp.
        
        Args:
            message: Warning message
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] PERFORMANCE WARNING: {message}")
    
    def _log_profiling_info(self):
        """
        Log detailed profiling information.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] PROFILING INFO:")
        print(f"  FPS: {self._current_fps:.1f}")
        print(f"  Avg Processing Time: {self._avg_processing_time_ms:.2f} ms")
        print(f"  Memory Usage: {self._current_memory_mb:.2f} MB")
        print(f"  Frames Processed: {self._total_frames_processed}")
        
        if len(self._memory_snapshots) > 0:
            avg_memory = sum(self._memory_snapshots) / len(self._memory_snapshots)
            max_memory = max(self._memory_snapshots)
            print(f"  Avg Memory (last 100): {avg_memory:.2f} MB")
            print(f"  Peak Memory (last 100): {max_memory:.2f} MB")
    
    def _create_disabled_result(self, timestamp: float) -> DualAttentionResult:
        """
        Create a minimal result when performance disabled.
        
        Args:
            timestamp: Current timestamp
        
        Returns:
            Minimal DualAttentionResult
        """
        from src.video_analysis.distraction_tracker import DistractionStateEnum
        
        fused_score = FusedScore(
            combined_score=0.0,
            camera_score=0.0,
            screen_score=0.0,
            face_presence=False,
            screen_presence=False,
            fusion_mode="disabled",
            timestamp=timestamp
        )
        
        distraction_state = DistractionState(
            is_distracted=False,
            distraction_duration=0.0,
            should_alert=False,
            total_distraction_time=0.0,
            last_alert_time=None,
            current_state=DistractionStateEnum.FOCUSED
        )
        
        return DualAttentionResult(
            fused_score=fused_score,
            distraction_state=distraction_state,
            should_show_alert=False,
            alert_message=None,
            timestamp=timestamp
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated statistics from all components.
        
        Returns:
            Dictionary containing:
            - camera_score: Latest camera attention score
            - screen_score: Latest screen attention score
            - combined_score: Latest combined score
            - face_presence: Camera face detection status
            - screen_presence: Screen face detection status
            - fusion_mode: Current fusion mode
            - current_distraction: Current distraction duration
            - total_distraction: Total distraction time
            - alert_count: Number of alerts triggered
            - frames_processed: Total frames processed
            - frames_with_camera: Frames with camera detection
            - frames_with_screen: Frames with screen detection
            - frames_with_both: Frames with both detections
        """
        # Get fusion details
        fusion_details = self.score_fusion.get_fusion_details()
        
        # Get distraction statistics
        distraction_stats = self.distraction_tracker.get_statistics()
        
        # Get camera statistics
        camera_stats = self.camera_detector.get_statistics()
        
        stats = {
            # Current scores
            'camera_score': fusion_details.get('camera_score', 0.0),
            'screen_score': fusion_details.get('screen_score', 0.0),
            'combined_score': fusion_details.get('combined_score', 0.0),
            
            # Presence indicators
            'face_presence': fusion_details.get('face_presence', False),
            'screen_presence': fusion_details.get('screen_presence', False),
            
            # Fusion mode
            'fusion_mode': fusion_details.get('fusion_mode', 'none'),
            
            # Distraction tracking
            'current_distraction': distraction_stats.get('distraction_duration', 0.0),
            'total_distraction': distraction_stats.get('total_distraction_time', 0.0),
            'alert_count': distraction_stats.get('alert_count', 0),
            'distraction_state': distraction_stats.get('current_state', 'focused'),
            
            # Frame statistics
            'frames_processed': self._total_frames_processed,
            'frames_with_camera': self._frames_with_camera,
            'frames_with_screen': self._frames_with_screen,
            'frames_with_both': self._frames_with_both,
            
            # Camera statistics
            'average_attention': camera_stats.get('average_attention', 100.0),
            'blink_count': camera_stats.get('blink_count', 0),
            
            # System status
            'screen_monitor_active': self.screen_monitor is not None and self.screen_monitor.is_running(),
            'is_running': self._running
        }
        
        # Add performance metrics if monitoring enabled
        if self.enable_performance_monitoring:
            stats.update({
                # Performance metrics
                'fps': self._current_fps,
                'avg_processing_time_ms': self._avg_processing_time_ms,
                'memory_usage_mb': self._current_memory_mb,
                'baseline_memory_mb': self._baseline_memory_mb,
                
                # Performance status
                'performance_monitoring_enabled': True,
                'performance_disabled': self._performance_disabled,
                'disable_reason': self._disable_reason,
                
                # Thresholds
                'fps_threshold': self.fps_threshold,
                'memory_threshold_mb': self.memory_threshold_mb,
                
                # Warnings
                'low_fps_duration': (time.time() - self._low_fps_start_time) if self._low_fps_start_time else 0.0,
                'high_memory_duration': (time.time() - self._high_memory_start_time) if self._high_memory_start_time else 0.0,
            })
        else:
            stats['performance_monitoring_enabled'] = False
        
        return stats
    
    def reset_statistics(self):
        """
        Reset all statistics in all components.
        
        This resets:
        - Distraction tracker state and statistics
        - Alert manager state
        - Camera detector statistics
        - Frame counters
        """
        print("Resetting dual attention statistics...")
        
        # Reset distraction tracker
        try:
            self.distraction_tracker.reset()
        except Exception as e:
            print(f"Error resetting distraction tracker: {e}")
        
        # Reset alert manager
        try:
            self.alert_manager.reset()
        except Exception as e:
            print(f"Error resetting alert manager: {e}")
        
        # Reset camera detector
        try:
            self.camera_detector.reset()
        except Exception as e:
            print(f"Error resetting camera detector: {e}")
        
        # Reset frame counters
        self._total_frames_processed = 0
        self._frames_with_camera = 0
        self._frames_with_screen = 0
        self._frames_with_both = 0
        
        # Reset error counters
        self._camera_error_count = 0
        self._screen_error_count = 0
        
        # Reset performance metrics
        if self.enable_performance_monitoring:
            self._frame_times.clear()
            self._current_fps = 0.0
            self._last_frame_time = None
            self._low_fps_start_time = None
            self._fps_warnings_logged = 0
            
            self._memory_snapshots.clear()
            self._high_memory_start_time = None
            self._memory_warnings_logged = 0
            
            self._processing_times.clear()
            self._avg_processing_time_ms = 0.0
            
            # Reset baseline memory
            try:
                current, peak = tracemalloc.get_traced_memory()
                self._baseline_memory_mb = current / 1024 / 1024
                self._current_memory_mb = 0.0
            except:
                pass
            
            # Re-enable if was disabled
            if self._performance_disabled:
                self._performance_disabled = False
                self._disable_reason = None
                print("Performance monitoring re-enabled")
        
        print("Statistics reset complete")


if __name__ == '__main__':
    # Demo usage
    print("DualAttentionCoordinator Demo")
    print("=" * 70)
    
    try:
        import cv2
        from src.inference.face_detector import FaceDetector
        
        # Initialize components
        print("\nInitializing components...")
        face_detector = FaceDetector(device='auto', confidence_threshold=0.85)
        attention_detector = AttentionDetector()
        
        # Initialize coordinator
        print("\nInitializing coordinator...")
        coordinator = DualAttentionCoordinator(
            camera_detector=attention_detector,
            face_detector=face_detector,
            enable_screen_monitor=True
        )
        
        # Start coordinator
        print("\nStarting coordinator...")
        if coordinator.start():
            print("Coordinator started successfully!")
            
            # Open camera
            print("\nOpening camera...")
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: Could not open camera")
                coordinator.stop()
                exit(1)
            
            print("\nProcessing frames for 10 seconds...")
            print("Press 'q' to quit early")
            print("-" * 70)
            
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < 10.0:
                ret, frame = cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                
                # Detect face
                detections = face_detector.detect_faces(frame)
                landmarks = detections[0].landmarks if len(detections) > 0 else None
                
                # Process frame
                result = coordinator.process_frame(frame, landmarks)
                
                # Draw alert if needed
                if result.should_show_alert:
                    frame = coordinator.alert_manager.draw_alert(
                        frame,
                        result.distraction_state.distraction_duration
                    )
                
                # Display info
                frame_count += 1
                if frame_count % 30 == 0:  # Every 30 frames
                    print(f"\nFrame {frame_count}:")
                    print(f"  Combined score: {result.fused_score.combined_score:.1f}")
                    print(f"  Fusion mode: {result.fused_score.fusion_mode}")
                    print(f"  Distraction: {result.distraction_state.distraction_duration:.1f}s")
                    print(f"  Alert: {result.should_show_alert}")
                
                # Show frame
                cv2.imshow('Dual Attention Demo', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Show statistics
            print("\n" + "-" * 70)
            print("Final Statistics:")
            stats = coordinator.get_statistics()
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # Stop coordinator
            print("\nStopping coordinator...")
            coordinator.stop()
            print("Demo complete!")
        
        else:
            print("Failed to start coordinator")
    
    except Exception as e:
        print(f"\nError in demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
