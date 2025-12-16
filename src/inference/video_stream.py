"""
Video Stream Handler Module

Handles video input from camera or video files with threaded frame reading
to avoid blocking the main processing pipeline.

Requirements: 6.1, 6.2, 7.1, 7.2, 10.2
"""

import cv2
import threading
import queue
import time
from typing import Optional, Tuple, Union
import numpy as np
import logging

# Setup logger
logger = logging.getLogger(__name__)


class VideoStreamHandler:
    """
    Manages video stream input from camera or video file.
    
    Features:
    - Camera mode (device_id)
    - Video file mode (file path)
    - Threaded frame reading to avoid blocking
    - Frame buffer with max size
    - Retry logic for camera connection
    - FPS tracking
    """
    
    def __init__(
        self,
        source: Union[int, str],
        buffer_size: int = 5,
        retry_attempts: int = 3,
        retry_delay: float = 1.0
    ):
        """
        Initialize video stream handler.
        
        Args:
            source: Camera device ID (int) or video file path (str)
            buffer_size: Maximum number of frames to buffer
            retry_attempts: Number of retry attempts for camera connection
            retry_delay: Delay between retry attempts in seconds
        """
        self.source = source
        self.buffer_size = buffer_size
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_buffer: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.thread: Optional[threading.Thread] = None
        self.stopped = False
        
        # FPS tracking
        self.fps = 0.0
        self.frame_count = 0
        self.start_time = 0.0
        
        # Source type
        self.is_camera = isinstance(source, int)
        
        logger.info(f"VideoStreamHandler initialized with source: {source}")
    
    def start_stream(self) -> bool:
        """
        Start the video stream and begin threaded frame reading.
        
        Returns:
            bool: True if stream started successfully, False otherwise
        """
        # Try to open video source with retry logic
        for attempt in range(self.retry_attempts):
            try:
                self.cap = cv2.VideoCapture(self.source)
                
                if self.cap.isOpened():
                    # Get initial frame to verify stream
                    ret, frame = self.cap.read()
                    if ret:
                        logger.info(f"Video stream opened successfully on attempt {attempt + 1}")
                        
                        # Reset frame buffer
                        self.frame_buffer = queue.Queue(maxsize=self.buffer_size)
                        self.stopped = False
                        
                        # Start frame reading thread
                        self.thread = threading.Thread(target=self._read_frames, daemon=True)
                        self.thread.start()
                        
                        # Initialize FPS tracking
                        self.start_time = time.time()
                        self.frame_count = 0
                        
                        return True
                    else:
                        logger.warning(f"Could not read frame from source on attempt {attempt + 1}")
                        self.cap.release()
                else:
                    logger.warning(f"Could not open video source on attempt {attempt + 1}")
                
            except Exception as e:
                logger.error(f"Error opening video source on attempt {attempt + 1}: {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.retry_attempts - 1:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        
        logger.error(f"Failed to open video source after {self.retry_attempts} attempts")
        return False
    
    def _read_frames(self) -> None:
        """
        Internal method to continuously read frames in a separate thread.
        Runs until stopped or stream ends.
        """
        logger.info("Frame reading thread started")
        
        while not self.stopped:
            if not self.cap or not self.cap.isOpened():
                logger.warning("Video capture is not opened, stopping thread")
                break
            
            try:
                ret, frame = self.cap.read()
                
                if not ret:
                    if self.is_camera:
                        # Camera disconnected, try to reconnect
                        logger.warning("Camera frame read failed, attempting reconnect...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        # Video file ended
                        logger.info("Video file ended")
                        self.stopped = True
                        break
                
                # Try to add frame to buffer (non-blocking)
                try:
                    self.frame_buffer.put(frame, block=False)
                except queue.Full:
                    # Buffer is full, remove oldest frame and add new one
                    try:
                        self.frame_buffer.get_nowait()
                        self.frame_buffer.put(frame, block=False)
                    except queue.Empty:
                        pass
                
            except Exception as e:
                logger.error(f"Error reading frame: {e}")
                if not self.is_camera:
                    # For video files, stop on error
                    self.stopped = True
                    break
                else:
                    # For camera, wait and retry
                    time.sleep(self.retry_delay)
        
        logger.info("Frame reading thread stopped")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the buffer.
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
                - success: True if frame was read successfully
                - frame: The frame as numpy array, or None if no frame available
        """
        try:
            # Try to get frame from buffer with short timeout
            frame = self.frame_buffer.get(timeout=0.1)
            
            # Update FPS tracking
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                self.fps = self.frame_count / elapsed
            
            return True, frame
            
        except queue.Empty:
            # No frame available
            if self.stopped:
                return False, None
            else:
                # Stream is still running but no frame ready yet
                return True, None
    
    def stop_stream(self) -> None:
        """
        Stop the video stream and clean up resources.
        """
        logger.info("Stopping video stream...")
        
        # Signal thread to stop
        self.stopped = True
        
        # Wait for thread to finish (with timeout)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("Frame reading thread did not stop gracefully")
        
        # Release video capture
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
        
        logger.info("Video stream stopped and resources released")
    
    def is_opened(self) -> bool:
        """
        Check if video stream is currently opened and running.
        
        Returns:
            bool: True if stream is opened and running
        """
        return (
            self.cap is not None 
            and self.cap.isOpened() 
            and not self.stopped
            and (self.thread is not None and self.thread.is_alive())
        )
    
    def get_fps(self) -> float:
        """
        Get current frames per second.
        
        Returns:
            float: Current FPS
        """
        return self.fps
    
    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """
        Get the frame size (width, height).
        
        Returns:
            Optional[Tuple[int, int]]: (width, height) or None if not available
        """
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return None
    
    def get_total_frames(self) -> Optional[int]:
        """
        Get total number of frames (for video files only).
        
        Returns:
            Optional[int]: Total frames or None if not available (camera)
        """
        if not self.is_camera and self.cap and self.cap.isOpened():
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return None
    
    def get_source_fps(self) -> Optional[float]:
        """
        Get the source FPS (from video file or camera).
        
        Returns:
            Optional[float]: Source FPS or None if not available
        """
        if self.cap and self.cap.isOpened():
            return self.cap.get(cv2.CAP_PROP_FPS)
        return None
    
    def __enter__(self):
        """Context manager entry."""
        self.start_stream()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_stream()
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        self.stop_stream()
