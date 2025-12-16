"""
OpenAI Whisper STT Engine Implementation.

Implements Speech-to-Text using original OpenAI Whisper library.
Provides highest accuracy for Vietnamese transcription.
"""

import logging
import time
import math
from typing import Iterator, Optional, List
import numpy as np

from ..interfaces.engine import ISTTEngine, EngineInfo
from ..models.config import EngineConfig
from ..models.transcription import TranscriptionResult, TranscriptionSegment
from ..exceptions import EngineException, ModelLoadError
from .base_engine import BaseSTTEngine


logger = logging.getLogger(__name__)


class OpenAIWhisperEngine(BaseSTTEngine):
    """
    OpenAI Whisper STT Engine.
    
    Uses original OpenAI Whisper models for highest accuracy transcription.
    Supports Vietnamese language with word-level timestamps and automatic punctuation.
    
    Features:
    - Highest accuracy (95-98%)
    - Automatic punctuation
    - Word-level timestamps
    - Multiple temperature sampling
    - Beam search optimization
    
    Trade-offs:
    - Slower processing (2-3x real-time on CPU)
    - Higher memory usage
    - Requires openai-whisper package
    """
    
    def __init__(self):
        """Initialize OpenAI Whisper engine."""
        super().__init__("openai-whisper")
        self.whisper = None
        self.model = None
    
    def _initialize_impl(self, config: EngineConfig) -> bool:
        """
        Initialize OpenAI Whisper engine with configuration.
        
        Args:
            config: Engine configuration.
            
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            # Import OpenAI Whisper
            import whisper
            self.whisper = whisper
            
            logger.info(f"Loading OpenAI Whisper model: {config.model_size}")
            
            # Determine device
            device = "cpu" if config.device.value == "cpu" else "cuda"
            
            # Load model
            self.model = whisper.load_model(
                name=config.model_size,
                device=device,
                download_root=config.download_root,
                in_memory=True  # Load in memory for faster access
            )
            
            logger.info(
                f"OpenAI Whisper model loaded: {config.model_size} on {device}"
            )
            return True
            
        except ImportError:
            logger.error(
                "OpenAI Whisper not installed. "
                "Install with: pip install openai-whisper"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load OpenAI Whisper model: {e}")
            return False
    
    def _transcribe_chunk_impl(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> TranscriptionResult:
        """
        Transcribe audio chunk using OpenAI Whisper.
        
        Args:
            audio: Audio data to transcribe.
            sample_rate: Sample rate of audio.
            
        Returns:
            Transcription result.
        """
        start_time = time.time()
        
        # Prepare audio
        audio = self._prepare_audio(audio)
        
        # Check for empty audio
        if len(audio) == 0:
            return self._create_empty_result(0.0)
        
        # Get configuration
        config = self.get_config()
        
        # Prepare transcription parameters
        transcribe_params = {
            "language": config.language,
            "task": "transcribe",
            "verbose": False,
            "word_timestamps": True,
            "beam_size": config.beam_size,
            "best_of": config.best_of,
            "temperature": self._get_temperature_list(config.temperature),
            "condition_on_previous_text": True,
            "initial_prompt": self._get_initial_prompt(config.language),
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "suppress_tokens": [-1],
            "without_timestamps": False,
            "fp16": False,  # Use FP32 for accuracy
        }
        
        # Transcribe
        result = self.model.transcribe(audio, **transcribe_params)
        
        # Process segments
        segments = self._process_segments(result.get("segments", []))
        
        # Calculate overall confidence
        avg_confidence = self._calculate_average_confidence(segments)
        
        # Join text
        full_text = " ".join(seg.text for seg in segments)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=result.get("language", config.language),
            confidence=avg_confidence,
            processing_time=processing_time,
            engine_name=self._name,
            metadata={
                "model_size": config.model_size,
                "beam_size": config.beam_size,
                "device": config.device.value,
            }
        )
    
    def _transcribe_stream_impl(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int,
    ) -> Iterator[TranscriptionResult]:
        """
        Transcribe audio stream.
        
        Note: OpenAI Whisper is not optimized for streaming.
        This implementation buffers audio and processes in chunks.
        
        Args:
            audio_stream: Iterator of audio chunks.
            sample_rate: Sample rate of audio.
            
        Yields:
            Transcription results.
        """
        audio_buffer = []
        config = self.get_config()
        
        for audio_chunk in audio_stream:
            audio_buffer.append(audio_chunk)
            
            # Calculate buffer duration
            total_samples = sum(len(chunk) for chunk in audio_buffer)
            duration = total_samples / sample_rate
            
            # Process when buffer reaches threshold (3 seconds)
            if duration >= 3.0:
                # Concatenate buffer
                audio = np.concatenate(audio_buffer)
                
                # Transcribe
                result = self._transcribe_chunk_impl(audio, sample_rate)
                
                # Yield if we have text
                if result.text:
                    yield result
                
                # Keep overlap for context
                overlap_samples = int(0.5 * sample_rate)
                if len(audio) > overlap_samples:
                    audio_buffer = [audio[-overlap_samples:]]
                else:
                    audio_buffer = []
    
    def _get_engine_info_impl(self) -> EngineInfo:
        """
        Get OpenAI Whisper engine information.
        
        Returns:
            Engine information.
        """
        config = self.get_config()
        model_size = config.model_size if config else "unknown"
        
        return EngineInfo(
            name="openai-whisper",
            version="latest",
            supported_languages=["vi", "en", "zh", "ja", "ko", "th", "fr", "de", "es", "it", "pt", "ru"],
            supports_streaming=True,
            supports_timestamps=True,
            supports_confidence=True,
            max_audio_length=None,  # No hard limit
        )
    
    def _cleanup_impl(self) -> None:
        """Cleanup OpenAI Whisper resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.whisper is not None:
            self.whisper = None
    
    # Helper methods
    
    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Prepare audio for transcription.
        
        Args:
            audio: Raw audio data.
            
        Returns:
            Prepared audio in float32 format, range [-1, 1].
        """
        # Convert to float32 if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure in correct range
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _get_temperature_list(self, temperature: float) -> List[float]:
        """
        Get temperature list for multiple sampling.
        
        Args:
            temperature: Base temperature.
            
        Returns:
            List of temperatures for sampling.
        """
        if temperature == 0.0:
            return [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        else:
            return [temperature]
    
    def _get_initial_prompt(self, language: str) -> str:
        """
        Get initial prompt for language.
        
        Args:
            language: Language code.
            
        Returns:
            Initial prompt text.
        """
        prompts = {
            "vi": "Xin chào, tôi đang nói tiếng Việt. Đây là một cuộc phỏng vấn.",
            "en": "Hello, I am speaking English. This is an interview.",
        }
        return prompts.get(language, "")
    
    def _process_segments(
        self, 
        raw_segments: List[dict]
    ) -> List[TranscriptionSegment]:
        """
        Process raw segments from Whisper.
        
        Args:
            raw_segments: Raw segments from Whisper.
            
        Returns:
            List of processed transcription segments.
        """
        segments = []
        
        for segment in raw_segments:
            text = segment.get("text", "").strip()
            if not text:
                continue
            
            # Calculate confidence from log probability
            avg_logprob = segment.get("avg_logprob", -1.0)
            confidence = self._logprob_to_confidence(avg_logprob)
            
            # Create segment
            try:
                trans_segment = TranscriptionSegment(
                    text=text,
                    start_time=segment.get("start", 0.0),
                    end_time=segment.get("end", 0.0),
                    confidence=confidence,
                    language=self.get_config().language,
                    metadata={
                        "avg_logprob": avg_logprob,
                        "compression_ratio": segment.get("compression_ratio", 1.0),
                        "no_speech_prob": segment.get("no_speech_prob", 0.0),
                    }
                )
                segments.append(trans_segment)
            except ValueError as e:
                logger.warning(f"Invalid segment data: {e}")
                continue
        
        return segments
    
    def _logprob_to_confidence(self, avg_logprob: float) -> float:
        """
        Convert average log probability to confidence score.
        
        Args:
            avg_logprob: Average log probability from Whisper.
            
        Returns:
            Confidence score in range [0, 1].
        """
        # Clamp to reasonable range
        avg_logprob = max(avg_logprob, -5.0)
        
        # Convert to confidence using exponential
        confidence = math.exp(avg_logprob)
        
        # Ensure in [0, 1] range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _calculate_average_confidence(
        self, 
        segments: List[TranscriptionSegment]
    ) -> float:
        """
        Calculate average confidence across segments.
        
        Args:
            segments: List of transcription segments.
            
        Returns:
            Average confidence score.
        """
        if not segments:
            return 0.0
        
        total_confidence = sum(seg.confidence for seg in segments)
        return total_confidence / len(segments)
    
    def _create_empty_result(self, processing_time: float) -> TranscriptionResult:
        """
        Create empty transcription result.
        
        Args:
            processing_time: Time taken for processing.
            
        Returns:
            Empty transcription result.
        """
        config = self.get_config()
        
        return TranscriptionResult(
            text="",
            segments=[],
            language=config.language if config else "vi",
            confidence=0.0,
            processing_time=processing_time,
            engine_name=self._name,
            metadata={}
        )
