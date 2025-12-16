"""
Emotion Scanner Core - Backend chức năng

Tách riêng logic xử lý khỏi giao diện.
"""

import cv2
import time
import threading
import numpy as np
from typing import Optional, Callable
from pathlib import Path

from ..inference.face_detector import FaceDetector
from ..inference.emotion_classifier import EmotionClassifier
from ..utils.screen_capture import ScreenCapture
from ..speech_analysis.system_audio_capture import SystemAudioCapture
from ..speech_analysis.audio_capture import AudioCapture
from ..speech_analysis.speech_to_text import create_stt_engine, STTConfig
from ..video_analysis.attention_detector import AttentionDetector
from ..utils.video_audio_player import VideoAudioPlayer, check_moviepy_installed


class EmotionScanner:
    """
    Backend xử lý emotion detection và speech-to-text.
    
    Tách riêng logic để có thể dùng cho nhiều frontend khác nhau.
    """
    
    def __init__(self, 
                 enable_emotion: bool = True,
                 enable_attention: bool = True,
                 enable_speech: bool = True):
        """
        Khởi tạo scanner.
        
        Args:
            enable_emotion: Bật emotion detection
            enable_attention: Bật attention detection
            enable_speech: Bật speech-to-text
        """
        self.enable_emotion = enable_emotion
        self.enable_attention = enable_attention
        self.enable_speech = enable_speech
        
        # Components
        self.face_detector = None
        self.emotion_classifier = None
        self.attention_detector = None
        self.stt_engine = None
        self.screen_capture = None
        
        # State
        self.running = False
        self.audio_thread = None
        self.audio_capture = None
        self.latest_transcription = ""
        self.transcription_lock = threading.Lock()
        
        # Callbacks
        self.on_transcription = None  # Callback(text: str)
        self.on_emotion = None  # Callback(emotion: str, confidence: float)
        self.on_attention = None  # Callback(score: float, should_alert: bool)
    
    def initialize(self):
        """Khởi tạo các components."""
        if self.enable_emotion:
            self.face_detector = FaceDetector(device='auto', max_faces=5)
            self.emotion_classifier = EmotionClassifier(device='auto')
        
        if self.enable_attention:
            self.attention_detector = AttentionDetector(
                gaze_threshold=0.25,
                pose_threshold=12.0,
                alert_duration=3.0
            )
        
        if self.enable_speech:
            config = STTConfig(model_name="medium", language="vi", device="cpu")
            self.stt_engine = create_stt_engine(config, robust=False)
        
        self.screen_capture = ScreenCapture()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Xử lý một frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Processed frame với annotations
        """
        if not self.enable_emotion:
            return frame
        
        # Detect faces
        detections = self.face_detector.detect_faces(frame)
        
        # Process each face
        for detection in detections:
            if detection.landmarks is not None:
                x, y, w, h = detection.bbox
                
                # Emotion
                face_img = frame[y:y+h, x:x+w]
                if face_img.size > 0:
                    emotion, confidence = self.emotion_classifier.predict(face_img)
                    
                    # Callback
                    if self.on_emotion:
                        self.on_emotion(emotion, confidence)
                    
                    # Draw
                    color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{emotion}: {confidence:.2f}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Attention
                if self.enable_attention:
                    # Khuôn mặt đã được phát hiện (có landmarks)
                    score, details = self.attention_detector.calculate_attention_score(
                        detection.landmarks, frame.shape[:2], face_detected=True
                    )
                    
                    should_alert = self.attention_detector.should_alert(score)
                    
                    # Callback
                    if self.on_attention:
                        self.on_attention(score, should_alert)
                    
                    # Draw - Ngưỡng mới: >= 8.0 (thang 0-10)
                    att_color = (0, 255, 0) if score >= 8.0 else (0, 0, 255)
                    cv2.putText(frame, f"Attention: {score:.1f}/10", (x, y+h+20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, att_color, 2)
                    
                    if should_alert:
                        cv2.putText(frame, "!!! KHONG THAY KHUON MAT !!!", 
                                   (frame.shape[1]//2-200, frame.shape[0]-30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
        return frame
    
    def start_audio_capture(self, use_system_audio: bool = False):
        """
        Bắt đầu thu âm.
        
        Args:
            use_system_audio: True = system audio, False = microphone
        """
        if not self.enable_speech:
            return
        
        if use_system_audio:
            self.audio_capture = SystemAudioCapture()
        else:
            self.audio_capture = AudioCapture()
        
        self.audio_capture.start_recording()
        self.running = True
        
        # Start processing thread
        self.audio_thread = threading.Thread(target=self._process_audio, daemon=True)
        self.audio_thread.start()
    
    def stop_audio_capture(self):
        """Dừng thu âm."""
        self.running = False
        if self.audio_capture:
            self.audio_capture.stop_recording()
    
    def get_latest_transcription(self) -> str:
        """Lấy transcription mới nhất."""
        with self.transcription_lock:
            return self.latest_transcription
    
    def _process_audio(self):
        """Xử lý audio trong thread riêng."""
        while self.running:
            try:
                chunk = self.audio_capture.get_audio_chunk(timeout=2.0)
                if chunk is not None and len(chunk) > 0:
                    result = self.stt_engine.transcribe_chunk(chunk)
                    if result.text and result.text.strip():
                        with self.transcription_lock:
                            self.latest_transcription = result.text
                        
                        # Callback
                        if self.on_transcription:
                            self.on_transcription(result.text)
            except Exception as e:
                print(f"Audio processing error: {e}")
                time.sleep(1)
    
    def capture_screen(self) -> Optional[np.ndarray]:
        """Capture màn hình."""
        return self.screen_capture.capture()
    
    def cleanup(self):
        """Dọn dẹp tài nguyên."""
        self.stop_audio_capture()
