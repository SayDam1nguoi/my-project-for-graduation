# -*- coding: utf-8 -*-
"""
Audio Transcriber Module

Provides functionality to transcribe audio files to text using Whisper STT.
"""

import logging
from pathlib import Path
from typing import Optional, Callable
import wave
import numpy as np

logger = logging.getLogger(__name__)


class AudioTranscriber:
    """Transcribes audio files to text using Whisper STT."""
    
    def __init__(self):
        """Initialize audio transcriber."""
        self.whisper_engine = None
        self.last_confidence = None  # Store last transcription confidence
        self._initialize_whisper()
    
    def _initialize_whisper(self):
        """Initialize Whisper STT engine."""
        try:
            from src.speech_analysis.whisper_stt_engine import WhisperSTTEngine
            from src.speech_analysis.config import STTConfig
            
            # Create config for Vietnamese
            config = STTConfig()
            config.language = "vi"
            config.sample_rate = 16000
            
            # Initialize Whisper with base model (good balance of speed/accuracy)
            self.whisper_engine = WhisperSTTEngine(
                config=config,
                model_size="base",
                compute_type="int8",
                device="cpu",
                num_threads=4
            )
            
            logger.info("Whisper STT engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            self.whisper_engine = None
    
    def transcribe_file(
        self,
        audio_file_path: str,
        output_text_path: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> Optional[str]:
        """
        Transcribe audio file to text.
        
        Args:
            audio_file_path: Path to WAV audio file
            output_text_path: Optional path to save transcription (default: same name as audio with .txt)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Transcribed text, or None if transcription failed
        """
        if not self.whisper_engine:
            logger.error("Whisper engine not initialized")
            if progress_callback:
                progress_callback("❌ Lỗi: Whisper engine chưa được khởi tạo")
            return None
        
        try:
            audio_path = Path(audio_file_path)
            
            if not audio_path.exists():
                logger.error(f"Audio file not found: {audio_file_path}")
                if progress_callback:
                    progress_callback(f"❌ Không tìm thấy file: {audio_path.name}")
                return None
            
            if progress_callback:
                progress_callback(f"🎤 Đang đọc file audio: {audio_path.name}...")
            
            # Read WAV file
            audio_data, sample_rate = self._read_wav_file(audio_file_path)
            
            if audio_data is None:
                if progress_callback:
                    progress_callback("❌ Không thể đọc file audio")
                return None
            
            if progress_callback:
                progress_callback(f"🔄 Đang chuyển đổi sang text (có thể mất vài phút)...")
            
            # Transcribe using Whisper
            result = self.whisper_engine.transcribe_chunk(audio_data)
            
            if not result or not result.text:
                logger.warning("Transcription returned empty result")
                self.last_confidence = 0.0
                if progress_callback:
                    progress_callback("⚠️ Không phát hiện được giọng nói trong file audio")
                return ""
            
            transcribed_text = result.text.strip()
            
            # Store confidence for clarity analysis
            self.last_confidence = result.confidence
            logger.info(f"Transcription confidence: {result.confidence:.2%}")
            
            # Filter hallucinations
            try:
                from src.speech_analysis.hallucination_filter import filter_hallucination
                filtered_text, was_filtered, reason = filter_hallucination(transcribed_text)
                
                if was_filtered:
                    logger.warning(f"Filtered hallucination: '{transcribed_text[:100]}...'")
                    logger.warning(f"Reason: {reason}")
                    if progress_callback:
                        progress_callback("⚠️ Phát hiện văn bản ảo (hallucination) - đã lọc bỏ")
                    transcribed_text = filtered_text  # Will be empty string
                else:
                    logger.info("Text passed hallucination filter")
            except Exception as e:
                logger.warning(f"Could not apply hallucination filter: {e}")
            
            # Determine output path
            if output_text_path is None:
                output_text_path = audio_path.with_suffix('.txt')
            
            # Save to file
            with open(output_text_path, 'w', encoding='utf-8') as f:
                f.write(transcribed_text)
            
            logger.info(f"Transcription saved to: {output_text_path}")
            
            if progress_callback:
                progress_callback(
                    f"✅ Hoàn thành!\n\n"
                    f"📝 Nội dung: {transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}\n\n"
                    f"💾 Đã lưu vào: {output_text_path.name}"
                )
            
            return transcribed_text
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            if progress_callback:
                progress_callback(f"❌ Lỗi khi chuyển đổi: {str(e)}")
            return None
    
    def _read_wav_file(self, file_path: str) -> tuple[Optional[np.ndarray], int]:
        """
        Read WAV file and return audio data.
        
        Args:
            file_path: Path to WAV file
            
        Returns:
            Tuple of (audio_data as numpy array, sample_rate), or (None, 0) on error
        """
        try:
            with wave.open(str(file_path), 'rb') as wav_file:
                # Get audio parameters
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                # Read audio data
                audio_bytes = wav_file.readframes(n_frames)
                
                # Convert to numpy array
                if sample_width == 1:  # 8-bit
                    dtype = np.uint8
                elif sample_width == 2:  # 16-bit
                    dtype = np.int16
                elif sample_width == 3:  # 24-bit
                    dtype = np.int32
                elif sample_width == 4:  # 32-bit
                    dtype = np.int32
                else:
                    logger.error(f"Unsupported sample width: {sample_width}")
                    return None, 0
                
                audio_data = np.frombuffer(audio_bytes, dtype=dtype)
                
                # Convert to mono if stereo
                if n_channels == 2:
                    audio_data = audio_data.reshape(-1, 2).mean(axis=1)
                
                # Convert to float32 and normalize to [-1, 1]
                if dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                elif dtype == np.uint8:
                    audio_data = (audio_data.astype(np.float32) - 128) / 128.0
                
                logger.info(
                    f"Loaded audio: {n_frames} frames, {sample_rate} Hz, "
                    f"{n_channels} channels, {sample_width} bytes/sample"
                )
                
                return audio_data, sample_rate
                
        except Exception as e:
            logger.error(f"Failed to read WAV file: {e}", exc_info=True)
            return None, 0
    
    def is_available(self) -> bool:
        """Check if transcriber is available."""
        return self.whisper_engine is not None


# Singleton instance
_transcriber_instance: Optional[AudioTranscriber] = None


def get_transcriber() -> AudioTranscriber:
    """Get singleton transcriber instance."""
    global _transcriber_instance
    if _transcriber_instance is None:
        _transcriber_instance = AudioTranscriber()
    return _transcriber_instance
