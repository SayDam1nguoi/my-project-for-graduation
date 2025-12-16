"""
Google Speech-to-Text Engine Module

Sử dụng Google Cloud Speech-to-Text API để chuyển đổi giọng nói thành văn bản.
"""

import time
import numpy as np
from typing import Optional, List, Iterator, Callable
from datetime import datetime

from .speech_to_text import (
    SpeechToTextEngine, 
    STTConfig, 
    TranscriptionResult, 
    TranscriptionSegment
)


class GoogleSTTEngine(SpeechToTextEngine):
    """
    Google Cloud Speech-to-Text Engine.
    
    Sử dụng Google Cloud Speech-to-Text API để chuyển đổi giọng nói.
    Yêu cầu:
    - Google Cloud account với Speech-to-Text API enabled
    - Service account credentials (JSON file)
    - pip install google-cloud-speech
    """
    
    def __init__(
        self, 
        config: STTConfig,
        credentials_path: Optional[str] = None,
        use_enhanced: bool = True
    ):
        """
        Khởi tạo Google STT engine.
        
        Args:
            config: Cấu hình STT
            credentials_path: Đường dẫn đến Google Cloud credentials JSON
            use_enhanced: Sử dụng enhanced model (tốt hơn nhưng đắt hơn)
            
        Raises:
            ImportError: Nếu google-cloud-speech không được cài đặt
            RuntimeError: Nếu không thể khởi tạo client
        """
        super().__init__(config)
        
        try:
            from google.cloud import speech
            import os
            
            self.speech = speech
            
            # Set credentials if provided
            if credentials_path:
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            
            # Initialize client
            self.client = speech.SpeechClient()
            
            # Configuration
            self.use_enhanced = use_enhanced
            self.encoding = speech.RecognitionConfig.AudioEncoding.LINEAR16
            
            # Language mapping
            self.language_map = {
                "vi": "vi-VN",
                "en": "en-US",
                "ja": "ja-JP",
                "ko": "ko-KR",
                "zh": "zh-CN"
            }
            
        except ImportError:
            raise ImportError(
                "Google Cloud Speech không được cài đặt. "
                "Cài đặt với: pip install google-cloud-speech"
            )
        except Exception as e:
            raise RuntimeError(f"Không thể khởi tạo Google STT client: {e}")
    
    def transcribe_chunk(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Chuyển đổi một chunk âm thanh thành văn bản.
        
        Args:
            audio: Audio data (numpy array, int16)
            
        Returns:
            TranscriptionResult với văn bản và metadata
        """
        start_time = time.time()
        
        try:
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
            
            # Prepare audio
            audio_content = self.speech.RecognitionAudio(content=audio_bytes)
            
            # Get language code
            language_code = self.language_map.get(self.language, "vi-VN")
            
            # Configure recognition
            config = self.speech.RecognitionConfig(
                encoding=self.encoding,
                sample_rate_hertz=self.sample_rate,
                language_code=language_code,
                enable_automatic_punctuation=True,  # Tự động thêm dấu câu
                use_enhanced=self.use_enhanced,  # Enhanced model
                model="default" if not self.use_enhanced else "phone_call",
                # Alternative language codes (nếu cần)
                # alternative_language_codes=["en-US"] if language_code == "vi-VN" else []
            )
            
            # Perform recognition
            response = self.client.recognize(config=config, audio=audio_content)
            
            # Process results
            if not response.results:
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language=self.language,
                    segments=[],
                    processing_time=time.time() - start_time
                )
            
            # Get best result
            result = response.results[0]
            alternative = result.alternatives[0]
            
            text = alternative.transcript
            confidence = alternative.confidence
            
            # Create segments from words (if available)
            segments = []
            if hasattr(alternative, 'words') and alternative.words:
                for word_info in alternative.words:
                    start_sec = word_info.start_time.total_seconds()
                    end_sec = word_info.end_time.total_seconds()
                    
                    segments.append(TranscriptionSegment(
                        text=word_info.word,
                        start_time=start_sec,
                        end_time=end_sec,
                        confidence=confidence
                    ))
            else:
                # Single segment for entire text
                segments.append(TranscriptionSegment(
                    text=text,
                    start_time=0.0,
                    end_time=len(audio) / self.sample_rate,
                    confidence=confidence
                ))
            
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                text=text,
                confidence=confidence,
                language=self.language,
                segments=segments,
                processing_time=processing_time
            )
            
        except Exception as e:
            # Return error result
            return TranscriptionResult(
                text=f"[lỗi: {str(e)}]",
                confidence=0.0,
                language=self.language,
                segments=[],
                processing_time=time.time() - start_time
            )
    
    def transcribe_realtime(
        self,
        audio_stream: Iterator[np.ndarray],
        callback: Callable[[TranscriptionResult], None]
    ) -> None:
        """
        Xử lý stream âm thanh real-time với streaming API.
        
        Args:
            audio_stream: Iterator của audio chunks
            callback: Function được gọi khi có kết quả mới
        """
        try:
            # Get language code
            language_code = self.language_map.get(self.language, "vi-VN")
            
            # Streaming config
            streaming_config = self.speech.StreamingRecognitionConfig(
                config=self.speech.RecognitionConfig(
                    encoding=self.encoding,
                    sample_rate_hertz=self.sample_rate,
                    language_code=language_code,
                    enable_automatic_punctuation=True,
                    use_enhanced=self.use_enhanced,
                ),
                interim_results=True  # Get interim results
            )
            
            # Generator for audio requests
            def request_generator():
                for audio_chunk in audio_stream:
                    # Convert to bytes
                    if audio_chunk.dtype != np.int16:
                        audio_chunk = audio_chunk.astype(np.int16)
                    
                    yield self.speech.StreamingRecognizeRequest(
                        audio_content=audio_chunk.tobytes()
                    )
            
            # Stream recognition
            responses = self.client.streaming_recognize(
                streaming_config,
                request_generator()
            )
            
            # Process responses
            for response in responses:
                if not response.results:
                    continue
                
                result = response.results[0]
                if not result.alternatives:
                    continue
                
                alternative = result.alternatives[0]
                
                # Only callback on final results
                if result.is_final:
                    transcription_result = TranscriptionResult(
                        text=alternative.transcript,
                        confidence=alternative.confidence,
                        language=self.language,
                        segments=[],
                        processing_time=0.0
                    )
                    callback(transcription_result)
                    
        except Exception as e:
            # Emit error via callback
            error_result = TranscriptionResult(
                text=f"[lỗi streaming: {str(e)}]",
                confidence=0.0,
                language=self.language,
                segments=[],
                processing_time=0.0
            )
            callback(error_result)
    
    def get_supported_languages(self) -> List[str]:
        """
        Lấy danh sách ngôn ngữ được hỗ trợ.
        
        Returns:
            List các mã ngôn ngữ
        """
        return list(self.language_map.keys())
    
    def is_available(self) -> bool:
        """
        Kiểm tra engine có khả dụng không.
        
        Returns:
            True nếu Google STT có thể sử dụng
        """
        try:
            # Try to create a client
            return self.client is not None
        except Exception:
            return False
