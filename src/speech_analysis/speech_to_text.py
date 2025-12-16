"""
Speech-to-Text Engine Module

Chuyển đổi giọng nói thành văn bản với hỗ trợ tiếng Việt.
Sử dụng Vosk cho speech recognition.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Iterator, Callable
import numpy as np


@dataclass
class STTConfig:
    """Cấu hình Speech-to-Text."""
    vosk_model_path: str = "models/vosk-model-vn-0.4"
    language: str = "vi"
    sample_rate: int = 16000


@dataclass
class TranscriptionSegment:
    """Một đoạn văn bản được chuyển đổi."""
    text: str
    start_time: float
    end_time: float
    confidence: float


@dataclass
class TranscriptionResult:
    """Kết quả chuyển đổi giọng nói thành văn bản."""
    text: str
    confidence: float
    language: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    processing_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


class SpeechToTextEngine(ABC):
    """Base class cho Speech-to-Text engines."""
    
    def __init__(self, config: STTConfig):
        """
        Khởi tạo STT engine.
        
        Args:
            config: Cấu hình STT
        """
        self.config = config
        self.sample_rate = config.sample_rate
        self.language = config.language
        
    @abstractmethod
    def transcribe_chunk(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Chuyển đổi một chunk âm thanh thành văn bản.
        
        Args:
            audio: Audio data (numpy array, int16)
            
        Returns:
            TranscriptionResult với văn bản và metadata
        """
        pass
    
    @abstractmethod
    def transcribe_realtime(self, 
                           audio_stream: Iterator[np.ndarray],
                           callback: Callable[[TranscriptionResult], None]) -> None:
        """
        Xử lý stream âm thanh real-time.
        
        Args:
            audio_stream: Iterator của audio chunks
            callback: Function được gọi khi có kết quả mới
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Lấy danh sách ngôn ngữ được hỗ trợ.
        
        Returns:
            List các mã ngôn ngữ (vi, en, etc.)
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Kiểm tra engine có khả dụng không.
        
        Returns:
            True nếu engine có thể sử dụng
        """
        pass


class VoskSTTEngine(SpeechToTextEngine):
    """
    Vosk Speech-to-Text Engine.
    
    Sử dụng Vosk model để chuyển đổi giọng nói thành văn bản.
    Hoạt động offline và nhẹ hơn, phù hợp cho real-time processing.
    """
    
    def __init__(self, config: STTConfig, model_path: Optional[str] = None):
        """
        Khởi tạo Vosk STT engine.
        
        Args:
            config: Cấu hình STT
            model_path: Đường dẫn đến Vosk model (optional, uses config if None)
            
        Raises:
            ImportError: Nếu vosk không được cài đặt
            RuntimeError: Nếu không thể load model
        """
        super().__init__(config)
        
        try:
            from vosk import Model, KaldiRecognizer
            import json
            self.vosk_model = Model
            self.vosk_recognizer = KaldiRecognizer
            self.json = json
        except ImportError:
            raise ImportError(
                "Vosk không được cài đặt. "
                "Cài đặt với: pip install vosk"
            )
        
        # Use model_path from parameter or config
        if model_path is None:
            model_path = config.vosk_model_path
        
        # Load model
        try:
            import os
            if not os.path.exists(model_path):
                raise RuntimeError(
                    f"Vosk model không tồn tại tại {model_path}. "
                    "Download model từ https://alphacephei.com/vosk/models"
                )
            
            self.model = self.vosk_model(model_path)
            self.recognizer = self.vosk_recognizer(self.model, config.sample_rate)
            
        except Exception as e:
            raise RuntimeError(f"Không thể load Vosk model: {e}")
        
        # Buffer cho real-time processing
        self.audio_buffer = []
        self.last_transcription = ""
    
    def transcribe_chunk(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Chuyển đổi một chunk âm thanh thành văn bản.
        
        Args:
            audio: Audio data (numpy array, int16)
            
        Returns:
            TranscriptionResult với văn bản và metadata
        """
        import time
        start_time = time.time()
        
        # Convert to bytes
        if audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        
        audio_bytes = audio.tobytes()
        
        if len(audio_bytes) == 0:
            return TranscriptionResult(
                text="",
                confidence=0.0,
                language=self.language,
                segments=[],
                processing_time=0.0
            )
        
        try:
            # Reset recognizer
            self.recognizer = self.vosk_recognizer(self.model, self.sample_rate)
            
            # Process audio
            self.recognizer.AcceptWaveform(audio_bytes)
            result_json = self.recognizer.FinalResult()
            result = self.json.loads(result_json)
            
            text = result.get("text", "")
            
            # Vosk không cung cấp confidence score, ước tính dựa trên text
            confidence = 0.8 if text else 0.0
            
            # Vosk không tự động thêm dấu câu
            if text and not text.endswith(('.', '!', '?')):
                text += '.'
            
            processing_time = time.time() - start_time
            
            # Tạo segment đơn giản
            segments = []
            if text:
                segments.append(TranscriptionSegment(
                    text=text,
                    start_time=0.0,
                    end_time=len(audio) / self.sample_rate,
                    confidence=confidence
                ))
            
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language=self.language,
                segments=segments,
                processing_time=processing_time
            )
            
        except Exception as e:
            # Trả về kết quả lỗi
            return TranscriptionResult(
                text="[không rõ]",
                confidence=0.0,
                language=self.language,
                segments=[],
                processing_time=time.time() - start_time
            )
    
    def transcribe_realtime(self,
                           audio_stream: Iterator[np.ndarray],
                           callback: Callable[[TranscriptionResult], None]) -> None:
        """
        Xử lý stream âm thanh real-time.
        
        Args:
            audio_stream: Iterator của audio chunks
            callback: Function được gọi khi có kết quả mới
        """
        import threading
        
        def process_stream():
            for audio_chunk in audio_stream:
                # Thêm vào buffer
                self.audio_buffer.append(audio_chunk)
                
                # Khi buffer đủ lớn (khoảng 2-3 giây), xử lý
                total_samples = sum(len(chunk) for chunk in self.audio_buffer)
                duration = total_samples / self.sample_rate
                
                if duration >= 2.0:
                    # Concatenate buffer
                    audio = np.concatenate(self.audio_buffer)
                    
                    # Transcribe
                    result = self.transcribe_chunk(audio)
                    
                    # Gọi callback nếu có văn bản mới
                    if result.text and result.text != self.last_transcription:
                        callback(result)
                        self.last_transcription = result.text
                    
                    # Clear buffer
                    self.audio_buffer = []
        
        # Chạy trong thread riêng
        thread = threading.Thread(target=process_stream, daemon=True)
        thread.start()
    
    def get_supported_languages(self) -> List[str]:
        """
        Lấy danh sách ngôn ngữ được hỗ trợ.
        
        Returns:
            List các mã ngôn ngữ (phụ thuộc vào model đã load)
        """
        return ["vi", "en"]  # Tùy thuộc vào model đã cài
    
    def is_available(self) -> bool:
        """
        Kiểm tra engine có khả dụng không.
        
        Returns:
            True nếu Vosk có thể sử dụng
        """
        try:
            from vosk import Model
            # Check if model and recognizer are initialized
            return self.model is not None and self.recognizer is not None
        except ImportError:
            return False


# Convenience function
def create_stt_engine(config: Optional[STTConfig] = None) -> SpeechToTextEngine:
    """
    Tạo Vosk STT engine với cấu hình mặc định.
    
    Args:
        config: Cấu hình STT (None = sử dụng mặc định)
        
    Returns:
        VoskSTTEngine instance
        
    Raises:
        RuntimeError: Nếu không thể khởi tạo engine
    """
    if config is None:
        config = STTConfig()
    
    try:
        engine = VoskSTTEngine(config)
        if engine.is_available():
            print(f"Sử dụng Vosk STT engine với model: {config.vosk_model_path}")
            return engine
        else:
            raise RuntimeError("Vosk engine không khả dụng")
    except Exception as e:
        raise RuntimeError(f"Không thể khởi tạo Vosk engine: {e}")
