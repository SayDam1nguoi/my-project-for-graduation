"""
Vosk STT Engine Implementation.

Implements Speech-to-Text using Vosk library.
Provides fast, lightweight fallback option for Vietnamese transcription.
"""

import logging
import time
import json
from typing import Iterator, Optional, List
import numpy as np
from pathlib import Path

from ..interfaces.engine import ISTTEngine, EngineInfo
from ..models.config import EngineConfig
from ..models.transcription import TranscriptionResult, TranscriptionSegment
from ..exceptions import EngineException, ModelLoadError
from .base_engine import BaseSTTEngine


logger = logging.getLogger(__name__)


class VoskEngine(BaseSTTEngine):
    """
    Vosk STT Engine.
    
    Uses Vosk library for fast, lightweight transcription.
    Suitable as a fallback option when Whisper models are unavailable.
    
    Features:
    - Fast processing (< 1x real-time on CPU)
    - Low memory usage
    - Offline operation
    - Lightweight models
    
    Trade-offs:
    - Lower accuracy (85-90%)
    - No automatic punctuation
    - Limited language support
    - No confidence scores from model
    """
    
    def __init__(self):
        """Initialize Vosk engine."""
        super().__init__("vosk")
        self.Model = None
        self.KaldiRecognizer = None
        self.model = None
        self.recognizer = None
    
    def _initialize_impl(self, config: EngineConfig) -> bool:
        """
        Initialize Vosk engine with configuration.
        
        Args:
            config: Engine configuration.
            
        Returns:
            True if initialization successful, False otherwise.
        """
        try:
            # Import Vosk
            from vosk import Model, KaldiRecognizer
            self.Model = Model
            self.KaldiRecognizer = KaldiRecognizer
            
            # Determine model path
            model_path = config.model_path
            if not model_path:
                # Try default Vietnamese model path
                model_path = "models/vosk-model-vn-0.4"
            
            # Check if model exists
            if not Path(model_path).exists():
                logger.error(
                    f"Vosk model not found at {model_path}. "
                    "Download from https://alphacephei.com/vosk/models"
                )
                return False
            
            logger.info(f"Loading Vosk model from: {model_path}")
            
            # Load model
            self.model = Model(model_path)
            
            # Create recognizer (will be recreated for each transcription)
            # Using 16000 Hz as default sample rate
            self.recognizer = KaldiRecognizer(self.model, 16000)
            
            logger.info(f"Vosk model loaded successfully from {model_path}")
            return True
            
        except ImportError:
            logger.error(
                "Vosk not installed. "
                "Install with: pip install vosk"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            return False
    
    def _transcribe_chunk_impl(
        self, 
        audio: np.ndarray, 
        sample_rate: int
    ) -> TranscriptionResult:
        """
        Transcribe audio chunk using Vosk.
        
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
        
        # Convert to bytes (Vosk expects int16)
        audio_bytes = audio.tobytes()
        
        # Create new recognizer for this transcription
        recognizer = self.KaldiRecognizer(self.model, sample_rate)
        
        # Process audio
        recognizer.AcceptWaveform(audio_bytes)
        result_json = recognizer.FinalResult()
        result = json.loads(result_json)
        
        # Extract text
        text = result.get("text", "").strip()
        
        # Vosk doesn't provide confidence scores, estimate based on text presence
        confidence = 0.8 if text else 0.0
        
        # Add punctuation if missing (Vosk doesn't add punctuation)
        if text and not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Create segment
        segments = []
        if text:
            duration = len(audio) / sample_rate
            segments.append(
                TranscriptionSegment(
                    text=text,
                    start_time=0.0,
                    end_time=duration,
                    confidence=confidence,
                    language=self.get_config().language,
                    metadata={}
                )
            )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        return TranscriptionResult(
            text=text,
            segments=segments,
            language=self.get_config().language,
            confidence=confidence,
            processing_time=processing_time,
            engine_name=self._name,
            metadata={
                "model_path": self.get_config().model_path or "default",
            }
        )
    
    def _transcribe_stream_impl(
        self,
        audio_stream: Iterator[np.ndarray],
        sample_rate: int,
    ) -> Iterator[TranscriptionResult]:
        """
        Transcribe audio stream.
        
        Vosk is well-suited for streaming transcription.
        
        Args:
            audio_stream: Iterator of audio chunks.
            sample_rate: Sample rate of audio.
            
        Yields:
            Transcription results.
        """
        # Create recognizer for streaming
        recognizer = self.KaldiRecognizer(self.model, sample_rate)
        
        for audio_chunk in audio_stream:
            # Prepare audio
            audio = self._prepare_audio(audio_chunk)
            
            if len(audio) == 0:
                continue
            
            # Convert to bytes
            audio_bytes = audio.tobytes()
            
            # Process chunk
            if recognizer.AcceptWaveform(audio_bytes):
                # Final result available
                result_json = recognizer.Result()
                result = json.loads(result_json)
                
                text = result.get("text", "").strip()
                if text:
                    # Add punctuation
                    if not text.endswith(('.', '!', '?')):
                        text += '.'
                    
                    confidence = 0.8
                    duration = len(audio) / sample_rate
                    
                    segments = [
                        TranscriptionSegment(
                            text=text,
                            start_time=0.0,
                            end_time=duration,
                            confidence=confidence,
                            language=self.get_config().language,
                            metadata={}
                        )
                    ]
                    
                    yield TranscriptionResult(
                        text=text,
                        segments=segments,
                        language=self.get_config().language,
                        confidence=confidence,
                        processing_time=0.0,  # Streaming, no total time
                        engine_name=self._name,
                        metadata={}
                    )
        
        # Get final partial result
        result_json = recognizer.FinalResult()
        result = json.loads(result_json)
        
        text = result.get("text", "").strip()
        if text:
            if not text.endswith(('.', '!', '?')):
                text += '.'
            
            confidence = 0.8
            
            segments = [
                TranscriptionSegment(
                    text=text,
                    start_time=0.0,
                    end_time=0.0,
                    confidence=confidence,
                    language=self.get_config().language,
                    metadata={}
                )
            ]
            
            yield TranscriptionResult(
                text=text,
                segments=segments,
                language=self.get_config().language,
                confidence=confidence,
                processing_time=0.0,
                engine_name=self._name,
                metadata={}
            )
    
    def _get_engine_info_impl(self) -> EngineInfo:
        """
        Get Vosk engine information.
        
        Returns:
            Engine information.
        """
        return EngineInfo(
            name="vosk",
            version="latest",
            supported_languages=["vi", "en", "zh", "ru", "fr", "de", "es"],
            supports_streaming=True,
            supports_timestamps=False,  # Limited timestamp support
            supports_confidence=False,  # No native confidence scores
            max_audio_length=None,  # No hard limit
        )
    
    def _cleanup_impl(self) -> None:
        """Cleanup Vosk resources."""
        if self.recognizer is not None:
            del self.recognizer
            self.recognizer = None
        
        if self.model is not None:
            del self.model
            self.model = None
        
        self.Model = None
        self.KaldiRecognizer = None
    
    # Helper methods
    
    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Prepare audio for transcription.
        
        Vosk expects int16 audio data.
        
        Args:
            audio: Raw audio data.
            
        Returns:
            Prepared audio in int16 format.
        """
        # Convert to int16 if needed
        if audio.dtype == np.float32 or audio.dtype == np.float64:
            # Assume float audio is in range [-1, 1]
            audio = np.clip(audio, -1.0, 1.0)
            audio = (audio * 32768.0).astype(np.int16)
        elif audio.dtype != np.int16:
            audio = audio.astype(np.int16)
        
        return audio
    
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
