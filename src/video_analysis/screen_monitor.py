"""
Screen Monitor Module

Monitors screen for face presence to enhance attention detection.
Captures screen periodically and detects faces in the captured frames.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import threading
import time
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import cv2

from src.utils.screen_capture import ScreenCapture, is_screen_capture_available
from src.inference.face_detector import FaceDetector


@dataclass
class ScreenMonitorResult:
    """
    Result from screen monitoring containing face detection information.
    
    Attributes:
        screen_presence: Whether a face is present on screen
        face_detected: Whether a face was detected in the latest capture
        face_bbox: Bounding box of detected face (x, y, width, height) or None
        confidence: Detection confidence score (0.0 to 1.0)
        timestamp: Time when the result was captured
        screen_score: Attention score based on screen presence (0-100)
    """
    screen_presence: bool
    face_detected: bool
    face_bbox: Optional[Tuple[int, int, int, int]]
    confidence: float
    timestamp: float
    screen_score: float  # 0-100
    
    def __post_init__(self):
        """Validate screen monitor result data."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if not (0.0 <= self.screen_score <= 100.0):
            raise ValueError(f"Screen score must be between 0.0 and 100.0, got {self.screen_score}")


class ScreenMonitor:
    """
    Monitor screen for face presence to enhance attention detection.
    
    Features:
    - Background thread for periodic screen capture (every 0.5 seconds)
    - Face detection in captured screen frames
    - Thread-safe result caching with lock
    - Error handling and retry logic for screen capture failures
    - Automatic calculation of screen attention score
    
    Requirements:
    - 2.1: Capture entire primary screen every 0.5 seconds
    - 2.2: Detect face presence in screen captures
    - 2.3: Set screen_presence to False after 1 second without detection
    - 2.4: Use confidence threshold of 0.85 for face detection
    - 2.5: Track only the largest face when multiple faces detected
    
    Example:
        >>> face_detector = FaceDetector()
        >>> monitor = ScreenMonitor(face_detector)
        >>> monitor.start()
        >>> time.sleep(1)
        >>> result = monitor.get_latest_result()
        >>> print(f"Face on screen: {result.screen_presence}")
        >>> monitor.stop()
    """
    
    def __init__(
        self,
        face_detector: FaceDetector,
        capture_interval: float = 0.5,
        confidence_threshold: float = 0.85,
        monitor_number: int = 1
    ):
        """
        Initialize ScreenMonitor.
        
        Args:
            face_detector: FaceDetector instance for detecting faces in screen captures
            capture_interval: Time between screen captures in seconds (default 0.5)
            confidence_threshold: Minimum confidence for face detection (default 0.85)
            monitor_number: Monitor to capture (1 = primary, 2 = secondary, etc.)
        
        Raises:
            RuntimeError: If screen capture is not available
            ValueError: If parameters are invalid
        """
        # Validate parameters
        if capture_interval <= 0:
            raise ValueError(f"capture_interval must be positive, got {capture_interval}")
        
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(
                f"confidence_threshold must be between 0.0 and 1.0, got {confidence_threshold}"
            )
        
        if not is_screen_capture_available():
            raise RuntimeError(
                "Screen capture is not available. "
                "Install mss with: pip install mss"
            )
        
        self.face_detector = face_detector
        self.capture_interval = capture_interval
        self.confidence_threshold = confidence_threshold
        self.monitor_number = monitor_number
        
        # Initialize screen capture
        self.screen_capture = ScreenCapture(monitor_number=monitor_number)
        
        # Thread management
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        
        # Result caching
        self._latest_result: Optional[ScreenMonitorResult] = None
        self._last_detection_time: float = 0.0
        
        # Error handling
        self._consecutive_failures = 0
        self._max_consecutive_failures = 10
        self._retry_delay = 0.1  # seconds
        
        print(f"ScreenMonitor initialized:")
        print(f"  Capture interval: {capture_interval}s")
        print(f"  Confidence threshold: {confidence_threshold}")
        print(f"  Monitor number: {monitor_number}")
    
    def start(self) -> bool:
        """
        Start screen monitoring in background thread.
        
        Returns:
            True if started successfully, False otherwise
        
        Example:
            >>> monitor = ScreenMonitor(face_detector)
            >>> if monitor.start():
            ...     print("Monitoring started")
        """
        if self._running:
            print("ScreenMonitor is already running")
            return True
        
        try:
            self._running = True
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            print("ScreenMonitor started successfully")
            return True
        
        except Exception as e:
            print(f"Error starting ScreenMonitor: {e}")
            self._running = False
            return False
    
    def stop(self):
        """
        Stop screen monitoring and cleanup resources.
        
        Example:
            >>> monitor.stop()
            >>> print("Monitoring stopped")
        """
        if not self._running:
            return
        
        print("Stopping ScreenMonitor...")
        self._running = False
        
        # Wait for thread to finish
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        print("ScreenMonitor stopped")
    
    def is_running(self) -> bool:
        """
        Check if screen monitoring is currently running.
        
        Returns:
            True if running, False otherwise
        """
        return self._running
    
    def get_latest_result(self) -> ScreenMonitorResult:
        """
        Get the latest screen monitoring result (thread-safe).
        
        Returns:
            Latest ScreenMonitorResult or a default result if no data available
        
        Example:
            >>> result = monitor.get_latest_result()
            >>> if result.screen_presence:
            ...     print(f"Face detected with confidence {result.confidence:.2f}")
        """
        with self._lock:
            if self._latest_result is None:
                # Return default result if no data yet
                return ScreenMonitorResult(
                    screen_presence=False,
                    face_detected=False,
                    face_bbox=None,
                    confidence=0.0,
                    timestamp=time.time(),
                    screen_score=0.0
                )
            
            # Check if result is stale (no detection for > 1 second)
            # Requirement 2.3: Set screen_presence to False after 1 second
            current_time = time.time()
            if (not self._latest_result.face_detected and 
                current_time - self._last_detection_time > 1.0):
                # Update screen_presence to False
                return ScreenMonitorResult(
                    screen_presence=False,
                    face_detected=False,
                    face_bbox=None,
                    confidence=0.0,
                    timestamp=self._latest_result.timestamp,
                    screen_score=0.0
                )
            
            return self._latest_result
    
    def _monitor_loop(self):
        """
        Main monitoring loop running in background thread.
        
        Continuously captures screen and detects faces at specified interval.
        """
        print("ScreenMonitor loop started")
        
        while self._running:
            try:
                # Capture screen
                frame = self._capture_with_retry()
                
                if frame is not None:
                    # Detect faces in screen capture
                    result = self._detect_face_in_screen(frame)
                    
                    # Update cached result (thread-safe)
                    with self._lock:
                        self._latest_result = result
                        if result.face_detected:
                            self._last_detection_time = result.timestamp
                    
                    # Reset failure count on success
                    self._consecutive_failures = 0
                
                else:
                    # Handle capture failure
                    self._consecutive_failures += 1
                    print(f"Screen capture failed ({self._consecutive_failures}/{self._max_consecutive_failures})")
                    
                    if self._consecutive_failures >= self._max_consecutive_failures:
                        print("Too many consecutive failures, stopping ScreenMonitor")
                        self._running = False
                        break
                
                # Wait for next capture interval
                time.sleep(self.capture_interval)
            
            except Exception as e:
                print(f"Error in ScreenMonitor loop: {e}")
                self._consecutive_failures += 1
                
                if self._consecutive_failures >= self._max_consecutive_failures:
                    print("Too many errors, stopping ScreenMonitor")
                    self._running = False
                    break
                
                time.sleep(self._retry_delay)
        
        print("ScreenMonitor loop ended")
    
    def _capture_with_retry(self, max_retries: int = 3) -> Optional[np.ndarray]:
        """
        Capture screen with retry logic.
        
        Args:
            max_retries: Maximum number of retry attempts
        
        Returns:
            Captured frame or None if all retries failed
        """
        for attempt in range(max_retries):
            try:
                frame = self.screen_capture.capture_current()
                if frame is not None:
                    return frame
            
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Screen capture attempt {attempt + 1} failed: {e}, retrying...")
                    time.sleep(self._retry_delay)
                else:
                    print(f"Screen capture failed after {max_retries} attempts: {e}")
        
        return None
    
    def _detect_face_in_screen(self, frame: np.ndarray) -> ScreenMonitorResult:
        """
        Detect face in captured screen frame.
        
        Args:
            frame: Captured screen frame (BGR format)
        
        Returns:
            ScreenMonitorResult with detection information
        """
        timestamp = time.time()
        
        try:
            # Detect faces using face detector
            detections = self.face_detector.detect_faces(frame)
            
            # Filter by confidence threshold (Requirement 2.4: 0.85)
            valid_detections = [
                d for d in detections 
                if d.confidence >= self.confidence_threshold
            ]
            
            if len(valid_detections) == 0:
                # No face detected
                return ScreenMonitorResult(
                    screen_presence=False,
                    face_detected=False,
                    face_bbox=None,
                    confidence=0.0,
                    timestamp=timestamp,
                    screen_score=0.0
                )
            
            # Requirement 2.5: Track only the largest face
            largest_face = max(valid_detections, key=lambda d: d.bbox[2] * d.bbox[3])
            
            # Calculate screen score based on face size and position
            screen_score = self._calculate_screen_score(largest_face, frame.shape)
            
            return ScreenMonitorResult(
                screen_presence=True,
                face_detected=True,
                face_bbox=largest_face.bbox,
                confidence=largest_face.confidence,
                timestamp=timestamp,
                screen_score=screen_score
            )
        
        except Exception as e:
            print(f"Error detecting face in screen: {e}")
            return ScreenMonitorResult(
                screen_presence=False,
                face_detected=False,
                face_bbox=None,
                confidence=0.0,
                timestamp=timestamp,
                screen_score=0.0
            )
    
    def _calculate_screen_score(
        self,
        detection,
        frame_shape: Tuple[int, int, int]
    ) -> float:
        """
        Calculate attention score based on face detection in screen.
        
        Score is based on:
        - Face size relative to screen (larger = higher score)
        - Face position (centered = higher score)
        - Detection confidence
        
        Args:
            detection: FaceDetection object
            frame_shape: Shape of the screen frame (height, width, channels)
        
        Returns:
            Screen attention score (0-100)
        """
        frame_height, frame_width = frame_shape[:2]
        x, y, w, h = detection.bbox
        
        # Calculate face size ratio (0-1)
        face_area = w * h
        screen_area = frame_width * frame_height
        size_ratio = min(face_area / screen_area, 0.1) / 0.1  # Cap at 10% of screen
        
        # Calculate center position score (0-1)
        face_center_x = x + w / 2
        face_center_y = y + h / 2
        screen_center_x = frame_width / 2
        screen_center_y = frame_height / 2
        
        # Distance from center (normalized)
        max_distance = np.sqrt(screen_center_x**2 + screen_center_y**2)
        distance = np.sqrt(
            (face_center_x - screen_center_x)**2 + 
            (face_center_y - screen_center_y)**2
        )
        center_score = 1.0 - min(distance / max_distance, 1.0)
        
        # Combine factors
        # 40% size, 30% position, 30% confidence
        score = (
            0.4 * size_ratio +
            0.3 * center_score +
            0.3 * detection.confidence
        ) * 100.0
        
        return min(score, 100.0)


if __name__ == '__main__':
    # Demo usage
    print("ScreenMonitor Demo")
    print("=" * 70)
    
    try:
        # Check if screen capture is available
        if not is_screen_capture_available():
            print("Error: Screen capture not available")
            print("Install mss with: pip install mss")
            exit(1)
        
        # Initialize face detector
        print("\nInitializing face detector...")
        face_detector = FaceDetector(device='auto', confidence_threshold=0.85)
        
        # Initialize screen monitor
        print("\nInitializing screen monitor...")
        monitor = ScreenMonitor(
            face_detector=face_detector,
            capture_interval=0.5,
            confidence_threshold=0.85,
            monitor_number=1
        )
        
        # Start monitoring
        print("\nStarting screen monitoring...")
        if monitor.start():
            print("Monitoring started successfully!")
            print("\nMonitoring for 10 seconds...")
            print("Move your face in front of the screen to test detection")
            print("-" * 70)
            
            # Monitor for 10 seconds
            for i in range(20):  # 20 iterations * 0.5s = 10s
                time.sleep(0.5)
                result = monitor.get_latest_result()
                
                print(f"\n[{i+1}/20] Screen Monitor Result:")
                print(f"  Face on screen: {result.screen_presence}")
                print(f"  Face detected: {result.face_detected}")
                print(f"  Confidence: {result.confidence:.2f}")
                print(f"  Screen score: {result.screen_score:.1f}/100")
                if result.face_bbox:
                    print(f"  Face bbox: {result.face_bbox}")
            
            # Stop monitoring
            print("\n" + "-" * 70)
            print("Stopping monitoring...")
            monitor.stop()
            print("Demo complete!")
        
        else:
            print("Failed to start monitoring")
    
    except Exception as e:
        print(f"\nError in demo: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
