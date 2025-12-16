"""
OpenAI Whisper STT Engine Module

Implements Speech-to-Text using original OpenAI Whisper library.
Supports Vietnamese language with high accuracy.
"""

import time
import logging
from typing import Optional, List, Iterator, Callable
import numpy as np
from pathlib import Path

from .speech_to_text import (
    SpeechToTextEngine,
    STTConfig,
    TranscriptionResult,
    TranscriptionSegment
)
from .exceptions import ModelLoadError

# Setup logging
logger = logging.getLogger(__name__)


class OpenAIWhisperSTTEngine(SpeechToTextEngine):
    """
    OpenAI Whisper Speech-to-Text Engine.
    
    Uses original OpenAI Whisper models for high-accuracy transcription.
    Supports Vietnamese language with word-level timestamps.
    """
    
    def __init__(
        self,
        config: STTConfig,
        model_size: str = "base",
        device: str = "cpu",
        download_root: Optional[str] = None,
        in_memory: bool = False
    ):
        """
        Initialize OpenAI Whisper STT engine.
        
        Args:
            config: Base STT configuration
            model_size: Model size (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to use (cpu or cuda)
            download_root: Directory to save model files
            in_memory: Load model in memory (faster but uses more RAM)
            
        Raises:
            ImportError: If openai-whisper is not installed
            ModelLoadError: If model loading fails
        """
        super().__init__(config)
        
        # Import OpenAI Whisper
        try:
            import whisper
            self.whisper = whisper
        except ImportError:
            raise ImportError(
                "OpenAI Whisper not installed. "
                "Install with: pip install openai-whisper"
            )
        
        # Store configuration
        self.model_size = model_size
        self.device = device
        self.download_root = download_root
        self.in_memory = in_memory
        
        # Load model
        self.model = None
        self._load_model()
        
        logger.info(
            f"OpenAI Whisper STT Engine initialized: "
            f"model={model_size}, device={device}"
        )
    
    def _load_model(self) -> None:
        """
        Load OpenAI Whisper model.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            logger.info(f"Loading OpenAI Whisper model: {self.model_size}")
            
            # Load model
            self.model = self.whisper.load_model(
                name=self.model_size,
                device=self.device,
                download_root=self.download_root,
                in_memory=self.in_memory
            )
            
            logger.info(f"OpenAI Whisper model loaded successfully: {self.model_size}")
            
        except Exception as e:
            error_msg = f"Failed to load OpenAI Whisper model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
    
    def transcribe_chunk(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe an audio chunk using OpenAI Whisper.
        
        Args:
            audio: Audio data (numpy array, int16 or float32)
            
        Returns:
            TranscriptionResult with text, confidence, and segments
        """
        start_time = time.time()
        
        try:
            # Convert audio to float32 if needed
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            
            # Ensure audio is in correct range [-1, 1]
            audio = np.clip(audio, -1.0, 1.0)
            
            # Check for empty audio
            if len(audio) == 0:
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language=self.language,
                    segments=[],
                    processing_time=0.0
                )
            
            # Transcribe with OpenAI Whisper - OPTIMIZED FOR ACCURACY
            # Lấy config từ STTConfig nếu có
            beam_size = getattr(self.config, 'beam_size', 10)
            best_of = getattr(self.config, 'best_of', 10)
            temperature = getattr(self.config, 'temperature', [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            initial_prompt = getattr(self.config, 'initial_prompt', 
                "Xin chào, tôi đang nói tiếng Việt. Đây là một cuộc phỏng vấn.")
            
            result = self.model.transcribe(
                audio,
                language=self.language,
                task="transcribe",
                verbose=False,
                word_timestamps=True,
                
                # ACCURACY SETTINGS
                beam_size=beam_size,                    # Beam search cho accuracy
                best_of=best_of,                        # Nhiều candidates
                temperature=temperature,                # Multiple temperatures
                
                # CONTEXT SETTINGS
                condition_on_previous_text=True,        # BẬT: Dùng context để chính xác hơn
                initial_prompt=initial_prompt,          # Vietnamese prompt
                
                # FILTERING SETTINGS - CÂN BẰNG
                compression_ratio_threshold=2.4,        # Standard threshold
                logprob_threshold=-1.0,                 # Standard threshold
                no_speech_threshold=0.6,                # Standard threshold
                
                # OTHER SETTINGS
                suppress_tokens=[-1],                   # Suppress special tokens
                without_timestamps=False,
                fp16=False                              # FP32 cho accuracy
            )
            
            # Extract segments
            segments = []
            full_text_parts = []
            total_confidence = 0.0
            segment_count = 0
            
            for segment in result.get('segments', []):
                # Extract segment information
                segment_text = segment['text'].strip()
                if not segment_text:
                    continue
                
                # Calculate confidence from avg_logprob first
                # OpenAI Whisper provides avg_logprob, convert to confidence [0, 1]
                avg_logprob = segment.get('avg_logprob', -1.0)
                confidence = self._logprob_to_confidence(avg_logprob)
                
                # Filter out very early segments (often hallucination at start) - CÂN BẰNG
                segment_start = segment.get('start', 0.0)
                if segment_start < 2.0 and confidence < 0.7:  # Đoạn đầu 2s cần confidence >= 0.7
                    logger.info(f"🚫 Filtered early low confidence ({confidence:.2f} at {segment_start:.1f}s): {segment_text}")
                    continue
                
                # Filter out low confidence segments (likely hallucination) - CÂN BẰNG
                if confidence < 0.5:  # 0.5 để không bỏ sót quá nhiều
                    logger.debug(f"Filtered low confidence ({confidence:.2f}): {segment_text}")
                    continue
                
                # Filter out common hallucinations/spam (Vietnamese)
                if self._is_hallucination(segment_text):
                    logger.info(f"🚫 Filtered hallucination: {segment_text}")
                    continue
                
                # Filter out segments with high compression ratio (repetitive = hallucination) - TĂNG CƯỜNG
                compression_ratio = segment.get('compression_ratio', 1.0)
                if compression_ratio > 2.0:  # GIẢM: 2.0 để lọc chặt hơn (từ 2.4)
                    logger.info(f"🚫 Filtered high compression ({compression_ratio:.2f}): {segment_text}")
                    continue
                
                # Filter out segments with no_speech_prob too high - TĂNG CƯỜNG
                no_speech_prob = segment.get('no_speech_prob', 0.0)
                if no_speech_prob > 0.6:  # GIẢM: 0.6 để lọc chặt hơn (từ 0.7)
                    logger.info(f"🚫 Filtered high no_speech_prob ({no_speech_prob:.2f}): {segment_text}")
                    continue
                
                # Create transcription segment
                trans_segment = TranscriptionSegment(
                    text=segment_text,
                    start_time=segment['start'],
                    end_time=segment['end'],
                    confidence=confidence
                )
                segments.append(trans_segment)
                
                # Accumulate for full text
                full_text_parts.append(segment_text)
                total_confidence += confidence
                segment_count += 1
            
            # Calculate overall confidence
            avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.0
            
            # Join text parts
            full_text = " ".join(full_text_parts)
            
            # Post-process: Fix common Vietnamese transcription errors
            full_text = self._fix_common_vietnamese_errors(full_text)
            
            # FINAL FILTER: Check toàn bộ transcript có hallucination không
            if self._is_hallucination(full_text):
                logger.warning(f"🚫 FULL TEXT hallucination detected, returning empty: {full_text[:100]}...")
                return TranscriptionResult(
                    text="",
                    confidence=0.0,
                    language=self.language,
                    segments=[],
                    processing_time=time.time() - start_time
                )
            
            # Text is already punctuated by Whisper
            
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                text=full_text,
                confidence=avg_confidence,
                language=result.get('language', self.language),
                segments=segments,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            processing_time = time.time() - start_time
            
            # Return error result
            return TranscriptionResult(
                text="[không rõ]",
                confidence=0.0,
                language=self.language,
                segments=[],
                processing_time=processing_time
            )
    
    def _logprob_to_confidence(self, avg_logprob: float) -> float:
        """
        Convert average log probability to confidence score.
        
        OpenAI Whisper provides avg_logprob in range [-inf, 0].
        We convert this to confidence in range [0, 1].
        
        Args:
            avg_logprob: Average log probability from Whisper
            
        Returns:
            Confidence score in range [0, 1]
        """
        import math
        
        # Clamp to reasonable range
        avg_logprob = max(avg_logprob, -5.0)  # Avoid extreme values
        
        # Convert to confidence using exponential
        confidence = math.exp(avg_logprob)
        
        # Ensure in [0, 1] range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _is_hallucination(self, text: str) -> bool:
        """
        Check if text is likely a hallucination/spam.
        
        Common Whisper hallucinations include:
        - YouTube subscription prompts
        - Channel promotion text
        - Generic filler phrases
        
        Args:
            text: Text to check
        
        Returns:
            True if text is likely hallucination
        """
        text_lower = text.lower().strip()
        
        # Exact phrase matches (full hallucination sentences) - TĂNG CƯỜNG
        exact_hallucinations = [
            # Subscribe variations
            'hãy subscribe cho kênh ghiền mì gõ để không bỏ lỡ những video hấp dẫn',
            'hãy subscribe cho kênh ghiền mì gõ',
            'subscribe cho kênh ghiền mì gõ',
            'ghiền mì gõ để không bỏ lỡ những video hấp dẫn',
            'để không bỏ lỡ những video hấp dẫn',
            'không bỏ lỡ những video hấp dẫn',
            'bỏ lỡ những video hấp dẫn',
            'đăng ký kênh để không bỏ lỡ video mới',
            'đăng ký kênh để không bỏ lỡ',
            'nhấn subscribe và bật chuông thông báo',
            'bấm subscribe và bật chuông',
            'subscribe và bật chuông',
            
            # Thanks/goodbye
            'cảm ơn các bạn đã xem video',
            'cảm ơn đã xem video',
            'cảm ơn đã theo dõi',
            'hẹn gặp lại trong video sau',
            'hẹn gặp lại các bạn',
            
            # Channel names
            'ghiền mì gõ',
            'kênh ghiền mì gõ',
            
            # Generic spam
            'đừng quên đăng ký',
            'đừng quên subscribe',
            'nhớ đăng ký nhé',
            'nhớ subscribe nhé',
        ]
        
        # Check exact matches first (most specific)
        for exact in exact_hallucinations:
            if exact in text_lower:
                logger.info(f"🚫 EXACT MATCH hallucination: {text}")
                return True
        
        # Check if text contains too many hallucination phrases (partial match) - TĂNG CƯỜNG
        partial_hallucinations = [
            'subscribe cho kênh',
            'subscribe kênh',
            'đăng ký kênh',
            'đăng kí kênh',
            'bỏ lỡ những video',
            'bỏ lỡ video',
            'video hấp dẫn',
            'ghiền mì gõ',
            'ghiền mì',
            'bật chuông thông báo',
            'bật chuông',
            'nhấn chuông',
            'bấm chuông',
            'theo dõi kênh',
            'like và share',
            'like share',
            'chia sẻ video',
        ]
        
        partial_match_count = sum(1 for phrase in partial_hallucinations if phrase in text_lower)
        if partial_match_count >= 1:  # GIẢM: Chỉ cần 1 cụm từ = hallucination (filter cực mạnh)
            logger.info(f"🚫 PARTIAL MATCH hallucination ({partial_match_count} matches): {text}")
            return True
        
        # Common Vietnamese hallucination patterns (keywords)
        hallucination_keywords = [
            # YouTube subscription prompts
            'subscribe',
            'đăng ký kênh',
            'đăng kí kênh',
            'theo dõi kênh',
            'bấm subscribe',
            'nhấn subscribe',
            'channel',
            'like và share',
            'like share',
            'chia sẻ video',
            'bấm chuông',
            'nhấn chuông',
            'bật thông báo',
            'notification',
            
            # Channel names (common patterns)
            'ghiền mì gõ',
            'ghiền mì',
            
            # Generic filler
            'cảm ơn đã xem',
            'cảm ơn các bạn đã xem',
            'hẹn gặp lại',
            'see you',
            'bye bye',
            
            # Music/sound effects
            '[âm nhạc]',
            '[music]',
            '[tiếng vỗ tay]',
            '[applause]',
        ]
        
        # Check if text contains multiple hallucination indicators
        # (more than 2 keywords = likely spam)
        keyword_count = sum(1 for keyword in hallucination_keywords if keyword in text_lower)
        if keyword_count >= 2:
            return True
        
        # Check single strong indicators - TĂNG CƯỜNG (filter cực mạnh)
        strong_indicators = [
            # Subscribe variations
            'subscribe',
            'subcribe',  # Typo common
            'đăng ký kênh',
            'đăng kí kênh',
            'theo dõi kênh',
            
            # Channel names
            'ghiền mì gõ',
            'ghiền mì',
            'mì gõ',
            
            # Notification
            'bấm chuông',
            'nhấn chuông',
            'bật chuông',
            'chuông thông báo',
            'thông báo',
            
            # Like/share
            'like và share',
            'like share',
            'chia sẻ video',
            
            # Generic
            'nhấn subscribe',
            'bấm subscribe',
            'đừng quên',
            'nhớ đăng ký',
        ]
        
        for indicator in strong_indicators:
            if indicator in text_lower:
                logger.info(f"🚫 STRONG INDICATOR hallucination: '{indicator}' in {text}")
                return True
        
        # Check if text is too short (likely noise)
        if len(text_lower) < 3:
            return True
        
        # Check if text is all punctuation or special characters
        if all(not c.isalnum() for c in text_lower):
            return True
        
        return False
    
    def _fix_common_vietnamese_errors(self, text: str) -> str:
        """
        Fix common Vietnamese transcription errors from Whisper.
        
        Args:
            text: Original transcribed text
        
        Returns:
            Corrected text
        """
        if not text:
            return text
        
        # Common misrecognitions (case-insensitive)
        corrections = {
            # Greetings - "alo" variations
            r'\bHà Lội\b': 'alo',
            r'\bHà Nội\b': 'alo',
            r'\bHà Lội\b': 'alo',
            r'\bHà lội\b': 'alo',
            r'\bA lô\b': 'alo',
            r'\bA-lô\b': 'alo',
            r'\bAlô\b': 'alo',
            r'\bHalo\b': 'alo',
            
            # "anh" at start of sentence (likely "alo")
            r'^anh\b': 'alo',
            r'^Anh\b': 'alo',
            
            # Common Vietnamese words
            r'\bvâng ạ\b': 'vâng ạ',
            r'\bvâng a\b': 'vâng ạ',  # Fix tone
            r'\bdạ\b': 'dạ',
            r'\bda\b': 'dạ',  # Fix tone
            r'\bvâng\b': 'vâng',
            r'\bvang\b': 'vâng',  # Fix tone
            r'\bưa\b': 'ừa',  # Fix tone
            r'\bua\b': 'ừa',
            
            # Common phrases
            r'\bxin chào\b': 'xin chào',
            r'\bcảm ơn\b': 'cảm ơn',
            r'\bcam on\b': 'cảm ơn',  # Fix tone
            r'\bkhông\b': 'không',
            r'\bkhong\b': 'không',  # Fix tone
            r'\bcó\b': 'có',
            r'\bco\b': 'có',  # Fix tone
            
            # Numbers
            r'\bmột\b': 'một',
            r'\bmot\b': 'một',
            r'\bhai\b': 'hai',
            r'\bba\b': 'ba',
            r'\bbốn\b': 'bốn',
            r'\bbon\b': 'bốn',
            r'\bnăm\b': 'năm',
            r'\bnam\b': 'năm',
            
            # Fix spacing issues
            r'\s+': ' ',  # Multiple spaces to single space
        }
        
        import re
        corrected = text
        
        for pattern, replacement in corrections.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        # Trim whitespace
        corrected = corrected.strip()
        
        return corrected
    
    def transcribe_realtime(
        self,
        audio_stream: Iterator[np.ndarray],
        callback: Callable[[TranscriptionResult], None]
    ) -> None:
        """
        Process audio stream in real-time.
        
        Args:
            audio_stream: Iterator of audio chunks
            callback: Function called with each transcription result
        """
        import threading
        
        def process_stream():
            audio_buffer = []
            
            for audio_chunk in audio_stream:
                # Add to buffer
                audio_buffer.append(audio_chunk)
                
                # Calculate buffer duration
                total_samples = sum(len(chunk) for chunk in audio_buffer)
                duration = total_samples / self.sample_rate
                
                # Process when buffer reaches 2-3 seconds
                if duration >= 2.0:
                    # Concatenate buffer
                    audio = np.concatenate(audio_buffer)
                    
                    # Transcribe
                    result = self.transcribe_chunk(audio)
                    
                    # Call callback if we have text
                    if result.text and result.text != "[không rõ]":
                        callback(result)
                    
                    # Keep last 0.5 seconds for overlap
                    overlap_samples = int(0.5 * self.sample_rate)
                    if len(audio) > overlap_samples:
                        audio_buffer = [audio[-overlap_samples:]]
                    else:
                        audio_buffer = []
        
        # Run in separate thread
        thread = threading.Thread(target=process_stream, daemon=True)
        thread.start()
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        OpenAI Whisper supports 99 languages.
        
        Returns:
            List of language codes
        """
        # Common languages
        return ["vi", "en", "zh", "ja", "ko", "th", "fr", "de", "es", "it", "pt", "ru"]
    
    def is_available(self) -> bool:
        """
        Check if engine is available and ready.
        
        Returns:
            True if model is loaded and ready
        """
        return self.model is not None
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "engine": "openai-whisper",
            "model_size": self.model_size,
            "device": self.device,
            "language": self.language,
            "is_available": self.is_available(),
            "in_memory": self.in_memory
        }


def create_openai_whisper_engine(
    config: Optional[STTConfig] = None,
    model_size: str = "base",
    device: str = "cpu"
) -> OpenAIWhisperSTTEngine:
    """
    Create an OpenAI Whisper STT engine with default configuration.
    
    Args:
        config: STT configuration (uses default if None)
        model_size: Whisper model size
        device: Device to use
        
    Returns:
        OpenAIWhisperSTTEngine instance
        
    Raises:
        ModelLoadError: If engine creation fails
    """
    if config is None:
        config = STTConfig()
    
    try:
        engine = OpenAIWhisperSTTEngine(
            config=config,
            model_size=model_size,
            device=device
        )
        
        if engine.is_available():
            logger.info(
                f"OpenAI Whisper STT engine created: "
                f"model={model_size}, device={device}"
            )
            return engine
        else:
            raise ModelLoadError("OpenAI Whisper engine not available")
            
    except Exception as e:
        raise ModelLoadError(f"Failed to create OpenAI Whisper engine: {str(e)}")

