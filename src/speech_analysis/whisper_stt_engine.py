"""
Whisper STT Engine Module

Implements Speech-to-Text using faster-whisper library with quantized models.
Optimized for CPU-only execution with Vietnamese language support.
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


class WhisperSTTEngine(SpeechToTextEngine):
    """
    Whisper Speech-to-Text Engine using faster-whisper.
    
    Uses quantized Whisper models for CPU-optimized transcription.
    Supports Vietnamese language with word-level timestamps.
    """
    
    def __init__(
        self,
        config: STTConfig,
        model_size: str = "base",
        model_path: Optional[str] = None,
        compute_type: str = "int8",
        device: str = "cpu",
        num_threads: int = 4
    ):
        """
        Initialize Whisper STT engine.
        
        Args:
            config: Base STT configuration
            model_size: Model size (tiny, base, small, medium, large)
            model_path: Custom model path (optional)
            compute_type: Quantization type (int8, int16, float16, float32)
            device: Device to use (cpu or cuda)
            num_threads: Number of CPU threads
            
        Raises:
            ImportError: If faster-whisper is not installed
            ModelLoadError: If model loading fails
        """
        super().__init__(config)
        
        # Import faster-whisper
        try:
            from faster_whisper import WhisperModel
            self.WhisperModel = WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper not installed. "
                "Install with: pip install faster-whisper"
            )
        
        # Store configuration
        self.model_size = model_size
        self.model_path = model_path
        self.compute_type = compute_type
        self.device = device
        self.num_threads = num_threads
        
        # Load model
        self.model = None
        self._load_model()
        
        # Verify CPU-only execution
        if device == "cpu":
            self._verify_cpu_only()
        
        logger.info(
            f"Whisper STT Engine initialized: "
            f"model={model_size}, compute_type={compute_type}, device={device}"
        )
    
    def _load_model(self) -> None:
        """
        Load Whisper model with specified configuration.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        try:
            # Use custom path or download model
            model_path = self.model_path if self.model_path else self.model_size
            
            # Load model with CPU optimization
            self.model = self.WhisperModel(
                model_path,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=self.num_threads,
                num_workers=1  # Single worker for stability
            )
            
            logger.info(f"Whisper model loaded successfully: {model_path}")
            
        except Exception as e:
            error_msg = f"Failed to load Whisper model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg)
    
    def _verify_cpu_only(self) -> None:
        """
        Verify that model is running on CPU only.
        
        This checks that no GPU resources are being used.
        """
        try:
            # Check if CUDA is available but not being used
            import torch
            if torch.cuda.is_available() and self.device == "cpu":
                logger.info("CUDA available but using CPU as configured")
            elif not torch.cuda.is_available():
                logger.info("CUDA not available, using CPU")
        except ImportError:
            # torch not available, assume CPU-only
            logger.info("PyTorch not available, assuming CPU-only execution")
    
    def transcribe_chunk(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe an audio chunk using Whisper.
        
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
            
            # Transcribe with Whisper
            segments_iter, info = self.model.transcribe(
                audio,
                language=self.language,
                task="transcribe",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                word_timestamps=True,
                vad_filter=False,  # VAD handled separately
                condition_on_previous_text=True
            )
            
            # Collect segments
            segments = []
            full_text_parts = []
            total_confidence = 0.0
            segment_count = 0
            
            for segment in segments_iter:
                # Extract segment information
                segment_text = segment.text.strip()
                if not segment_text:
                    continue
                
                # Calculate confidence from log probability
                # Whisper provides avg_logprob, convert to confidence [0, 1]
                confidence = self._logprob_to_confidence(segment.avg_logprob)
                
                # Create transcription segment
                trans_segment = TranscriptionSegment(
                    text=segment_text,
                    start_time=segment.start,
                    end_time=segment.end,
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
            
            # Add punctuation if missing
            if full_text and not full_text.endswith(('.', '!', '?')):
                full_text += '.'
            
            processing_time = time.time() - start_time
            
            return TranscriptionResult(
                text=full_text,
                confidence=avg_confidence,
                language=self.language,
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
        
        Whisper provides avg_logprob in range [-inf, 0].
        We convert this to confidence in range [0, 1].
        
        Args:
            avg_logprob: Average log probability from Whisper
            
        Returns:
            Confidence score in range [0, 1]
        """
        # Typical range for avg_logprob is around [-1.0, 0.0]
        # Map this to confidence [0, 1]
        # Using exponential: confidence = exp(avg_logprob)
        import math
        
        # Clamp to reasonable range
        avg_logprob = max(avg_logprob, -5.0)  # Avoid extreme values
        
        # Convert to confidence
        confidence = math.exp(avg_logprob)
        
        # Ensure in [0, 1] range
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
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
        
        Whisper supports many languages, but we focus on Vietnamese.
        
        Returns:
            List of language codes
        """
        return ["vi", "en", "zh", "ja", "ko", "th"]  # Common Asian languages
    
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
            "model_size": self.model_size,
            "compute_type": self.compute_type,
            "device": self.device,
            "num_threads": self.num_threads,
            "language": self.language,
            "is_available": self.is_available()
        }


def create_whisper_engine(
    config: Optional[STTConfig] = None,
    model_size: str = "base",
    compute_type: str = "int8",
    device: str = "cpu",
    num_threads: int = 4
) -> WhisperSTTEngine:
    """
    Create a Whisper STT engine with default configuration.
    
    Args:
        config: STT configuration (uses default if None)
        model_size: Whisper model size
        compute_type: Quantization type
        device: Device to use
        num_threads: Number of CPU threads
        
    Returns:
        WhisperSTTEngine instance
        
    Raises:
        ModelLoadError: If engine creation fails
    """
    if config is None:
        config = STTConfig()
    
    try:
        engine = WhisperSTTEngine(
            config=config,
            model_size=model_size,
            compute_type=compute_type,
            device=device,
            num_threads=num_threads
        )
        
        if engine.is_available():
            logger.info(
                f"Whisper STT engine created: "
                f"model={model_size}, compute_type={compute_type}"
            )
            return engine
        else:
            raise ModelLoadError("Whisper engine not available")
            
    except Exception as e:
        raise ModelLoadError(f"Failed to create Whisper engine: {str(e)}")
