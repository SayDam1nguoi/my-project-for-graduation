"""
Visualization Engine Module

Renders emotion recognition results on video frames with colored bounding boxes,
labels, confidence scores, and FPS counter.

Requirements: 8.1, 8.2, 8.3, 8.4, 8.5
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from .model_loader import FaceDetection, EmotionPrediction


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.6
    font_thickness: int = 2
    
    # Box settings
    box_thickness: int = 2
    
    # Text settings
    text_offset_x: int = 5
    text_offset_y: int = 25
    text_background_alpha: float = 0.6
    
    # FPS counter settings
    fps_position: Tuple[int, int] = (10, 30)
    fps_font_scale: float = 0.7
    fps_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    
    # Confidence threshold for display
    min_confidence: float = 0.0


class VisualizationEngine:
    """
    Renders emotion recognition results on video frames.
    
    Features:
    - Colored bounding boxes for each emotion
    - Emotion labels with confidence scores
    - FPS counter
    - Customizable colors and styles
    """
    
    # Emotion color mapping (BGR format for OpenCV)
    EMOTION_COLORS: Dict[str, Tuple[int, int, int]] = {
        'Happy': (0, 255, 0),      # Green
        'Sad': (255, 0, 0),         # Blue
        'Angry': (0, 0, 255),       # Red
        'Fear': (128, 0, 128),      # Purple
        'Surprise': (0, 255, 255),  # Yellow
        'Disgust': (0, 165, 255),   # Orange
        'Neutral': (128, 128, 128), # Gray
    }
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualization engine.
        
        Args:
            config: Visualization configuration. If None, uses default config.
        """
        self.config = config or VisualizationConfig()
        
    def draw_results(
        self,
        frame: np.ndarray,
        results: List[Tuple[FaceDetection, EmotionPrediction]],
        fps: Optional[float] = None,
        show_probabilities: bool = False
    ) -> np.ndarray:
        """
        Draw emotion recognition results on frame.
        
        Args:
            frame: Input frame (BGR format)
            results: List of (FaceDetection, EmotionPrediction) tuples
            fps: Current FPS to display (optional)
            show_probabilities: Whether to show all emotion probabilities
            
        Returns:
            np.ndarray: Frame with visualizations drawn
        """
        # Create a copy to avoid modifying original
        output_frame = frame.copy()
        
        # Draw each face detection and emotion prediction
        for face_detection, emotion_prediction in results:
            # Skip if confidence is too low
            if emotion_prediction.confidence < self.config.min_confidence:
                continue
            
            # Draw bounding box and label
            output_frame = self._draw_face_box(
                output_frame,
                face_detection,
                emotion_prediction
            )
            
            # Optionally draw probability distribution
            if show_probabilities:
                output_frame = self._draw_probabilities(
                    output_frame,
                    face_detection,
                    emotion_prediction
                )
        
        # Draw FPS counter
        if fps is not None:
            output_frame = self._draw_fps(output_frame, fps)
        
        return output_frame
    
    def _draw_face_box(
        self,
        frame: np.ndarray,
        face: FaceDetection,
        emotion: EmotionPrediction
    ) -> np.ndarray:
        """
        Draw bounding box and emotion label for a single face.
        
        Args:
            frame: Input frame
            face: Face detection result
            emotion: Emotion prediction result
            
        Returns:
            np.ndarray: Frame with box and label drawn
        """
        x, y, w, h = face.bbox
        
        # Get color for this emotion
        color = self.EMOTION_COLORS.get(emotion.emotion, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            self.config.box_thickness
        )
        
        # Prepare label text
        label = f"{emotion.emotion} {emotion.confidence:.0%}"
        
        # Calculate text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            self.config.font,
            self.config.font_scale,
            self.config.font_thickness
        )
        
        # Draw text background (semi-transparent)
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x, y - text_height - baseline - 10),
            (x + text_width + 10, y),
            color,
            -1  # Filled rectangle
        )
        
        # Blend overlay with original frame
        cv2.addWeighted(
            overlay,
            self.config.text_background_alpha,
            frame,
            1 - self.config.text_background_alpha,
            0,
            frame
        )
        
        # Draw text
        cv2.putText(
            frame,
            label,
            (x + self.config.text_offset_x, y - 10),
            self.config.font,
            self.config.font_scale,
            (255, 255, 255),  # White text
            self.config.font_thickness,
            cv2.LINE_AA
        )
        
        # Draw face detection confidence (smaller text below box)
        if face.confidence is not None:
            face_conf_text = f"Face: {face.confidence:.0%}"
            cv2.putText(
                frame,
                face_conf_text,
                (x, y + h + 20),
                self.config.font,
                self.config.font_scale * 0.7,
                color,
                1,
                cv2.LINE_AA
            )
        
        return frame
    
    def _draw_probabilities(
        self,
        frame: np.ndarray,
        face: FaceDetection,
        emotion: EmotionPrediction
    ) -> np.ndarray:
        """
        Draw probability distribution for all emotions.
        
        Args:
            frame: Input frame
            face: Face detection result
            emotion: Emotion prediction result
            
        Returns:
            np.ndarray: Frame with probabilities drawn
        """
        x, y, w, h = face.bbox
        
        # Position for probability text (to the right of face box)
        prob_x = x + w + 10
        prob_y = y + 20
        
        # Sort emotions by probability (descending)
        sorted_emotions = sorted(
            emotion.probabilities.items(),
            key=lambda item: item[1],
            reverse=True
        )
        
        # Draw each emotion probability
        for i, (emo_name, prob) in enumerate(sorted_emotions):
            color = self.EMOTION_COLORS.get(emo_name, (255, 255, 255))
            text = f"{emo_name}: {prob:.1%}"
            
            # Draw text
            cv2.putText(
                frame,
                text,
                (prob_x, prob_y + i * 25),
                self.config.font,
                self.config.font_scale * 0.6,
                color,
                1,
                cv2.LINE_AA
            )
            
            # Draw probability bar
            bar_width = int(prob * 100)
            cv2.rectangle(
                frame,
                (prob_x + 120, prob_y + i * 25 - 12),
                (prob_x + 120 + bar_width, prob_y + i * 25 - 2),
                color,
                -1
            )
        
        return frame
    
    def _draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draw FPS counter on frame.
        
        Args:
            frame: Input frame
            fps: Current FPS value
            
        Returns:
            np.ndarray: Frame with FPS counter drawn
        """
        fps_text = f"FPS: {fps:.1f}"
        
        # Draw text background
        (text_width, text_height), baseline = cv2.getTextSize(
            fps_text,
            self.config.font,
            self.config.fps_font_scale,
            2
        )
        
        x, y = self.config.fps_position
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - 5, y - text_height - 5),
            (x + text_width + 5, y + baseline + 5),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Draw FPS text
        cv2.putText(
            frame,
            fps_text,
            (x, y),
            self.config.font,
            self.config.fps_font_scale,
            self.config.fps_color,
            2,
            cv2.LINE_AA
        )
        
        return frame
    
    def create_summary_overlay(
        self,
        frame: np.ndarray,
        stats: Dict[str, any]
    ) -> np.ndarray:
        """
        Create a summary overlay with statistics.
        
        Args:
            frame: Input frame
            stats: Dictionary with statistics to display
                   e.g., {'total_faces': 3, 'emotions': {'Happy': 2, 'Sad': 1}}
            
        Returns:
            np.ndarray: Frame with summary overlay
        """
        output_frame = frame.copy()
        
        # Position for summary (top-right corner)
        frame_height, frame_width = frame.shape[:2]
        summary_x = frame_width - 250
        summary_y = 30
        
        # Draw semi-transparent background
        overlay = output_frame.copy()
        cv2.rectangle(
            overlay,
            (summary_x - 10, summary_y - 25),
            (frame_width - 10, summary_y + len(stats) * 30 + 10),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, output_frame, 0.3, 0, output_frame)
        
        # Draw statistics
        y_offset = summary_y
        for key, value in stats.items():
            text = f"{key}: {value}"
            cv2.putText(
                output_frame,
                text,
                (summary_x, y_offset),
                self.config.font,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            y_offset += 30
        
        return output_frame
    
    def draw_landmarks(
        self,
        frame: np.ndarray,
        face: FaceDetection,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """
        Draw facial landmarks if available.
        
        Args:
            frame: Input frame
            face: Face detection result with landmarks
            color: Color for landmarks (BGR)
            
        Returns:
            np.ndarray: Frame with landmarks drawn
        """
        if face.landmarks is None:
            return frame
        
        output_frame = frame.copy()
        
        # Draw each landmark point
        for landmark in face.landmarks:
            x, y = int(landmark[0]), int(landmark[1])
            cv2.circle(output_frame, (x, y), 2, color, -1)
        
        return output_frame
    
    @staticmethod
    def get_emotion_color(emotion: str) -> Tuple[int, int, int]:
        """
        Get the color for a specific emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Tuple[int, int, int]: BGR color tuple
        """
        return VisualizationEngine.EMOTION_COLORS.get(
            emotion,
            (255, 255, 255)  # Default white
        )
    
    @staticmethod
    def create_legend(
        width: int = 300,
        height: int = 250
    ) -> np.ndarray:
        """
        Create a legend image showing emotion colors.
        
        Args:
            width: Legend width
            height: Legend height
            
        Returns:
            np.ndarray: Legend image
        """
        legend = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(
            legend,
            "Emotion Colors",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        # Draw each emotion with its color
        y_offset = 60
        for emotion, color in VisualizationEngine.EMOTION_COLORS.items():
            # Color box
            cv2.rectangle(
                legend,
                (10, y_offset - 15),
                (40, y_offset + 5),
                color,
                -1
            )
            
            # Emotion name
            cv2.putText(
                legend,
                emotion,
                (50, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            y_offset += 30
        
        return legend
