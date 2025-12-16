# -*- coding: utf-8 -*-
"""
Video Handler Module

Contains video processing and display logic.
"""

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from .constants import VIDEO_DISPLAY


class VideoHandler:
    """Handles video capture, processing, and display."""
    
    def __init__(self, canvas: tk.Label):
        """
        Initialize video handler.
        
        Args:
            canvas: Tkinter label to display video
        """
        self.canvas = canvas
        self.current_frame = None
        
    def update_display(self, frame: np.ndarray):
        """
        Update video display with new frame.
        
        Args:
            frame: BGR frame from OpenCV
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Ensure canvas has valid size
        if canvas_width <= 1:
            canvas_width = VIDEO_DISPLAY['default_width']
        if canvas_height <= 1:
            canvas_height = VIDEO_DISPLAY['default_height']
        
        # Limit canvas size to maximum
        canvas_width = min(canvas_width, VIDEO_DISPLAY['max_width'])
        canvas_height = min(canvas_height, VIDEO_DISPLAY['max_height'])
        
        # Calculate aspect ratio preserving resize
        frame_height, frame_width = frame_rgb.shape[:2]
        frame_aspect = frame_width / frame_height
        canvas_aspect = canvas_width / canvas_height
        
        # Calculate new dimensions to fit canvas while preserving aspect ratio
        if canvas_aspect > frame_aspect:
            # Canvas is wider - fit to height
            new_height = canvas_height
            new_width = int(canvas_height * frame_aspect)
        else:
            # Canvas is taller - fit to width
            new_width = canvas_width
            new_height = int(canvas_width / frame_aspect)
        
        # Ensure dimensions are at least 1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # Use faster interpolation for smoother playback
        # INTER_LINEAR is faster than INTER_AREA and good enough for display
        frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert to PhotoImage
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update canvas
        self.canvas.config(image=photo)
        self.canvas.image = photo
        self.current_frame = frame
        
    def clear_display(self):
        """Clear video display - show black screen."""
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Ensure canvas has valid size
        if canvas_width <= 1:
            canvas_width = VIDEO_DISPLAY['default_width']
        if canvas_height <= 1:
            canvas_height = VIDEO_DISPLAY['default_height']
        
        # Create black image
        black_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Convert to PhotoImage
        image = Image.fromarray(black_frame)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update canvas
        self.canvas.config(image=photo)
        self.canvas.image = photo
        self.current_frame = None
        
    def resize_frame(self, frame: np.ndarray, factor: float) -> np.ndarray:
        """
        Resize frame by a factor.
        
        Args:
            frame: Input frame
            factor: Resize factor (e.g., 0.5 for half size)
            
        Returns:
            Resized frame
        """
        if factor >= 1.0:
            return frame
            
        h, w = frame.shape[:2]
        new_w = int(w * factor)
        new_h = int(h * factor)
        # Use INTER_LINEAR for faster resizing
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


class VideoSourceManager:
    """Manages different video sources (camera, file, screen)."""
    
    def __init__(self):
        """Initialize video source manager."""
        self.cap = None
        self.source_type = None
        self.screen_capture = None
        
    def open_camera(self, camera_id: int = 0) -> bool:
        """
        Open camera source.
        
        Args:
            camera_id: Camera device ID
            
        Returns:
            True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(camera_id)
        if self.cap.isOpened():
            self.source_type = "camera"
            return True
        return False
        
    def open_video_file(self, file_path: str) -> bool:
        """
        Open video file source.
        
        Args:
            file_path: Path to video file
            
        Returns:
            True if successful, False otherwise
        """
        self.cap = cv2.VideoCapture(file_path)
        if self.cap.isOpened():
            self.source_type = "file"
            return True
        return False
        
    def open_screen_capture(self, screen_capture_obj) -> bool:
        """
        Open screen capture source.
        
        Args:
            screen_capture_obj: ScreenCapture instance
            
        Returns:
            True if successful, False otherwise
        """
        self.screen_capture = screen_capture_obj
        self.cap = "screen"  # Special marker
        self.source_type = "screen"
        return True
        
    def read_frame(self):
        """
        Read frame from current source.
        
        Returns:
            Tuple of (success, frame) or (None, None) if no source
        """
        if self.cap is None:
            return None, None
            
        if self.source_type == "screen":
            frame = self.screen_capture.capture_current()
            if frame is not None:
                return True, frame
            return False, None
        else:
            return self.cap.read()
            
    def release(self):
        """Release video source."""
        if self.cap is not None and self.cap != "screen":
            self.cap.release()
        self.cap = None
        self.source_type = None
        self.screen_capture = None
        
    def is_opened(self) -> bool:
        """Check if source is opened."""
        if self.cap is None:
            return False
        if self.source_type == "screen":
            return True
        return self.cap.isOpened()
