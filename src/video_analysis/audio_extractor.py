"""
Audio Extractor module for video transcription.

This module extracts and processes audio from video files.
"""

import numpy as np
from typing import Optional, Tuple
import logging
from pathlib import Path

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    VideoFileClip = None

try:
    import librosa
except ImportError:
    librosa = None

from ..speech_analysis.audio_cleaner import AudioCleaner


logger = logging.getLogger(__name__)


class AudioExtractionError(Exception):
    """Raised when audio extraction fails."""
    pass


class AudioExtractor:
    """Extracts and processes audio from video files."""
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        enable_enhancement: bool = True
    ):
        """
        Initialize audio extractor.
        
        Args:
            target_sample_rate: Target sample rate for output audio (default: 16000)
            target_channels: Target number of channels (1=mono, 2=stereo)
            enable_enhancement: Whether to apply audio enhancement
        """
        # Lazy check - only check when actually using
        self._moviepy_checked = False
        self._librosa_checked = False
        
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.enable_enhancement = enable_enhancement
        
        # Initialize audio cleaner for enhancement
        if enable_enhancement:
            try:
                self.audio_cleaner = AudioCleaner(sample_rate=target_sample_rate)
            except:
                self.audio_cleaner = None
                logger.warning("AudioCleaner not available, enhancement disabled")
        else:
            self.audio_cleaner = None
        
        logger.info(
            f"AudioExtractor initialized: sample_rate={target_sample_rate}, "
            f"channels={target_channels}, enhancement={enable_enhancement}"
        )
    
    def extract_audio(
        self,
        video_path: str,
        audio_track_index: int = 0
    ) -> np.ndarray:
        """
        Extract audio from video - Simplified version.
        
        Args:
            video_path: Path to video file
            audio_track_index: Index of audio track to extract (default: 0)
            
        Returns:
            Audio data as numpy array (16kHz mono float32)
            
        Raises:
            AudioExtractionError: If extraction fails
        """
        # Check dependencies when actually using
        if VideoFileClip is None:
            raise AudioExtractionError(
                "moviepy is required for audio extraction. "
                "Install it with: pip install moviepy"
            )
        
        if librosa is None:
            raise AudioExtractionError(
                "librosa is required for audio resampling. "
                "Install it with: pip install librosa"
            )
        
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise AudioExtractionError(f"Video file not found: {video_path}")
        
        logger.info(f"Extracting audio from: {video_path}")
        
        try:
            # Load video with moviepy
            video = VideoFileClip(str(video_path))
            
            # Check if video has audio
            if video.audio is None:
                video.close()
                raise AudioExtractionError(f"Video has no audio track: {video_path}")
            
            # Get audio properties
            fps = video.audio.fps
            duration = video.audio.duration
            
            logger.debug(f"Video audio properties: fps={fps}, duration={duration:.2f}s")
            
            # Extract audio as array - FIX for moviepy 1.0.3 bug
            # Bug: to_soundarray() has issue with numpy stacking
            # Solution: Use write_audiofile then load with librosa
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Write audio to temp file
                video.audio.write_audiofile(
                    tmp_path,
                    fps=16000,  # Target sample rate
                    nbytes=2,
                    codec='pcm_s16le',
                    logger=None,
                    verbose=False
                )
                
                # Close video
                video.close()
                
                # Load audio with librosa
                audio_array, sr = librosa.load(tmp_path, sr=self.target_sample_rate, mono=True)
                
                logger.info(
                    f"Audio extracted successfully: shape={audio_array.shape}, "
                    f"duration={len(audio_array) / self.target_sample_rate:.2f}s"
                )
                
                return audio_array
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            
        except AudioExtractionError:
            raise
        except Exception as e:
            raise AudioExtractionError(f"Failed to extract audio: {str(e)}") from e
    
    def extract_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        audio_track_index: int = 0
    ) -> np.ndarray:
        """
        Extract audio segment from video - Simplified version.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            audio_track_index: Index of audio track to extract
            
        Returns:
            Audio segment as numpy array (16kHz mono float32)
            
        Raises:
            AudioExtractionError: If extraction fails
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise AudioExtractionError(f"Video file not found: {video_path}")
        
        if start_time < 0:
            raise AudioExtractionError(f"Start time must be >= 0, got {start_time}")
        
        if end_time <= start_time:
            raise AudioExtractionError(
                f"End time must be > start time, got start={start_time}, end={end_time}"
            )
        
        logger.info(
            f"Extracting audio segment from {video_path}: "
            f"{start_time:.2f}s to {end_time:.2f}s"
        )
        
        try:
            # Load video with moviepy
            video = VideoFileClip(str(video_path))
            
            # Check if video has audio
            if video.audio is None:
                video.close()
                raise AudioExtractionError(f"Video has no audio track: {video_path}")
            
            # Extract audio segment
            audio_clip = video.audio.subclip(start_time, end_time)
            
            # Extract audio - FIX for moviepy 1.0.3 bug
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                # Write audio to temp file
                audio_clip.write_audiofile(
                    tmp_path,
                    fps=16000,
                    nbytes=2,
                    codec='pcm_s16le',
                    logger=None,
                    verbose=False
                )
                
                # Close video
                video.close()
                
                # Load audio with librosa
                audio_array, sr = librosa.load(tmp_path, sr=self.target_sample_rate, mono=True)
                
                logger.info(
                    f"Audio segment extracted: shape={audio_array.shape}, "
                    f"duration={len(audio_array) / self.target_sample_rate:.2f}s"
                )
                
                return audio_array
                
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
            
        except AudioExtractionError:
            raise
        except Exception as e:
            raise AudioExtractionError(
                f"Failed to extract audio segment: {str(e)}"
            ) from e
    
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance audio quality using AudioCleaner.
        
        Args:
            audio: Input audio data (float32, normalized)
            
        Returns:
            Enhanced audio data
        """
        if self.audio_cleaner is None:
            logger.warning("Audio enhancement disabled, returning original audio")
            return audio
        
        try:
            enhanced = self.audio_cleaner.clean_audio(audio)
            return enhanced
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}. Returning original audio.")
            return audio
    
    def resample_audio(
        self,
        audio: np.ndarray,
        original_sample_rate: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate using librosa.
        
        Args:
            audio: Input audio data
            original_sample_rate: Original sample rate in Hz
            
        Returns:
            Resampled audio data at target sample rate
        """
        if original_sample_rate == self.target_sample_rate:
            return audio
        
        try:
            # Use librosa for high-quality resampling
            resampled = librosa.resample(
                audio,
                orig_sr=original_sample_rate,
                target_sr=self.target_sample_rate,
                res_type='kaiser_best'  # High-quality resampling
            )
            
            logger.debug(
                f"Resampled audio from {original_sample_rate} Hz to "
                f"{self.target_sample_rate} Hz: "
                f"{len(audio)} -> {len(resampled)} samples"
            )
            
            return resampled
            
        except Exception as e:
            logger.error(f"Resampling failed: {e}")
            raise AudioExtractionError(f"Failed to resample audio: {str(e)}") from e
    
    def get_audio_info(self, video_path: str) -> dict:
        """
        Get audio information from video without extracting.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with audio information (sample_rate, channels, duration)
            
        Raises:
            AudioExtractionError: If unable to get info
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise AudioExtractionError(f"Video file not found: {video_path}")
        
        try:
            video = VideoFileClip(str(video_path))
            
            if video.audio is None:
                video.close()
                raise AudioExtractionError(f"Video has no audio track: {video_path}")
            
            info = {
                'sample_rate': video.audio.fps,
                'duration': video.audio.duration,
                'n_channels': video.audio.nchannels,
            }
            
            video.close()
            
            return info
            
        except AudioExtractionError:
            raise
        except Exception as e:
            raise AudioExtractionError(
                f"Failed to get audio info: {str(e)}"
            ) from e
