"""
Video Validator module for video transcription.

This module validates video files before processing, checking format,
audio tracks, and quality metrics.
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import logging

try:
    # Try moviepy 2.x import first
    from moviepy import VideoFileClip
except ImportError:
    try:
        # Fallback to moviepy 1.x import
        from moviepy.editor import VideoFileClip
    except ImportError:
        VideoFileClip = None

try:
    import ffmpeg
except ImportError:
    ffmpeg = None


# Configure logging
logger = logging.getLogger(__name__)


class VideoValidationError(Exception):
    """Raised when video validation fails."""
    pass


@dataclass
class AudioTrackInfo:
    """Information about an audio track in a video."""
    index: int
    codec: str
    sample_rate: int
    channels: int
    duration: float


@dataclass
class QualityMetrics:
    """Audio quality metrics."""
    sample_rate: int
    bit_rate: Optional[int]
    channels: int
    duration: float
    has_audio: bool


@dataclass
class ValidationResult:
    """Result of video validation."""
    is_valid: bool
    video_path: str
    format: str
    duration: float
    has_audio: bool
    audio_tracks: List[AudioTrackInfo]
    quality_metrics: Optional[QualityMetrics]
    error_message: Optional[str] = None


class VideoValidator:
    """Validates video files for transcription processing."""
    
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mkv', '.mov']
    MIN_SAMPLE_RATE = 8000  # Minimum acceptable sample rate
    MIN_DURATION = 0.1  # Minimum video duration in seconds
    
    def __init__(self):
        """Initialize video validator."""
        if VideoFileClip is None:
            logger.warning("moviepy not available, some validation features may be limited")
        if ffmpeg is None:
            logger.warning("ffmpeg-python not available, some validation features may be limited")
    
    def validate_video(self, video_path: str) -> ValidationResult:
        """
        Validate video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            ValidationResult with detailed information
            
        Raises:
            VideoValidationError: If video is corrupted or invalid
        """
        path = Path(video_path)
        
        # Check if file exists
        if not path.exists():
            return ValidationResult(
                is_valid=False,
                video_path=str(video_path),
                format="",
                duration=0.0,
                has_audio=False,
                audio_tracks=[],
                quality_metrics=None,
                error_message=f"Video file not found: {video_path}"
            )
        
        # Check format
        if not self.check_format(video_path):
            return ValidationResult(
                is_valid=False,
                video_path=str(video_path),
                format=path.suffix,
                duration=0.0,
                has_audio=False,
                audio_tracks=[],
                quality_metrics=None,
                error_message=f"Unsupported video format: {path.suffix}. Supported formats: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        try:
            # Try to detect audio tracks using ffmpeg first (more reliable)
            audio_tracks = self.detect_audio_tracks(video_path)
            has_audio = len(audio_tracks) > 0
            
            # Get video duration and quality metrics
            quality_metrics = None
            duration = 0.0
            
            if has_audio:
                quality_metrics = self.assess_quality(video_path)
                duration = quality_metrics.duration
            else:
                # Try to get duration even without audio
                try:
                    if VideoFileClip is not None:
                        with VideoFileClip(str(video_path)) as clip:
                            duration = clip.duration
                except Exception as e:
                    logger.warning(f"Could not get video duration: {e}")
            
            # Validate minimum duration
            if duration < self.MIN_DURATION:
                return ValidationResult(
                    is_valid=False,
                    video_path=str(video_path),
                    format=path.suffix,
                    duration=duration,
                    has_audio=has_audio,
                    audio_tracks=audio_tracks,
                    quality_metrics=quality_metrics,
                    error_message=f"Video duration too short: {duration}s (minimum: {self.MIN_DURATION}s)"
                )
            
            # Check if audio quality is acceptable
            if has_audio and quality_metrics:
                if quality_metrics.sample_rate < self.MIN_SAMPLE_RATE:
                    return ValidationResult(
                        is_valid=False,
                        video_path=str(video_path),
                        format=path.suffix,
                        duration=duration,
                        has_audio=has_audio,
                        audio_tracks=audio_tracks,
                        quality_metrics=quality_metrics,
                        error_message=f"Audio sample rate too low: {quality_metrics.sample_rate}Hz (minimum: {self.MIN_SAMPLE_RATE}Hz)"
                    )
            
            # All checks passed
            return ValidationResult(
                is_valid=True,
                video_path=str(video_path),
                format=path.suffix,
                duration=duration,
                has_audio=has_audio,
                audio_tracks=audio_tracks,
                quality_metrics=quality_metrics,
                error_message=None
            )
            
        except Exception as e:
            # Handle corrupted or unreadable videos
            error_msg = f"Video file appears to be corrupted or unreadable: {str(e)}"
            logger.error(error_msg)
            
            return ValidationResult(
                is_valid=False,
                video_path=str(video_path),
                format=path.suffix,
                duration=0.0,
                has_audio=False,
                audio_tracks=[],
                quality_metrics=None,
                error_message=error_msg
            )
    
    def check_format(self, video_path: str) -> bool:
        """
        Check if video format is supported.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if format is supported
        """
        path = Path(video_path)
        return path.suffix.lower() in self.SUPPORTED_FORMATS
    
    def detect_audio_tracks(self, video_path: str) -> List[AudioTrackInfo]:
        """
        Detect audio tracks in video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of audio track information
            
        Raises:
            VideoValidationError: If video cannot be probed
        """
        audio_tracks = []
        
        # Try ffmpeg first (more reliable and detailed)
        if ffmpeg is not None:
            try:
                probe = ffmpeg.probe(str(video_path))
                
                for stream in probe.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        index = stream.get('index', 0)
                        codec = stream.get('codec_name', 'unknown')
                        sample_rate = int(stream.get('sample_rate', 0))
                        channels = stream.get('channels', 0)
                        
                        # Get duration from stream or format
                        duration = 0.0
                        if 'duration' in stream:
                            duration = float(stream['duration'])
                        elif 'duration' in probe.get('format', {}):
                            duration = float(probe['format']['duration'])
                        
                        audio_tracks.append(AudioTrackInfo(
                            index=index,
                            codec=codec,
                            sample_rate=sample_rate,
                            channels=channels,
                            duration=duration
                        ))
                
                return audio_tracks
                
            except ffmpeg.Error as e:
                error_msg = f"FFmpeg error while probing video: {e.stderr.decode() if e.stderr else str(e)}"
                logger.error(error_msg)
                raise VideoValidationError(error_msg)
            except Exception as e:
                logger.warning(f"FFmpeg probe failed, trying moviepy: {e}")
        
        # Fallback to moviepy
        if VideoFileClip is not None:
            try:
                with VideoFileClip(str(video_path)) as clip:
                    if clip.audio is not None:
                        # MoviePy doesn't provide detailed track info, so we create a basic entry
                        audio_tracks.append(AudioTrackInfo(
                            index=0,
                            codec='unknown',
                            sample_rate=clip.audio.fps if hasattr(clip.audio, 'fps') else 0,
                            channels=clip.audio.nchannels if hasattr(clip.audio, 'nchannels') else 0,
                            duration=clip.duration
                        ))
                
                return audio_tracks
                
            except Exception as e:
                error_msg = f"MoviePy error while detecting audio tracks: {str(e)}"
                logger.error(error_msg)
                raise VideoValidationError(error_msg)
        
        # If both methods failed
        if ffmpeg is None and VideoFileClip is None:
            raise VideoValidationError("No video processing library available (ffmpeg-python or moviepy required)")
        
        return audio_tracks
    
    def assess_quality(self, video_path: str) -> QualityMetrics:
        """
        Assess audio quality in video.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Quality metrics
            
        Raises:
            VideoValidationError: If quality cannot be assessed
        """
        # Try ffmpeg first for detailed quality info
        if ffmpeg is not None:
            try:
                probe = ffmpeg.probe(str(video_path))
                
                # Find first audio stream
                audio_stream = None
                for stream in probe.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        audio_stream = stream
                        break
                
                if audio_stream is None:
                    return QualityMetrics(
                        sample_rate=0,
                        bit_rate=None,
                        channels=0,
                        duration=float(probe.get('format', {}).get('duration', 0.0)),
                        has_audio=False
                    )
                
                sample_rate = int(audio_stream.get('sample_rate', 0))
                bit_rate = int(audio_stream.get('bit_rate', 0)) if 'bit_rate' in audio_stream else None
                channels = audio_stream.get('channels', 0)
                
                # Get duration
                duration = 0.0
                if 'duration' in audio_stream:
                    duration = float(audio_stream['duration'])
                elif 'duration' in probe.get('format', {}):
                    duration = float(probe['format']['duration'])
                
                return QualityMetrics(
                    sample_rate=sample_rate,
                    bit_rate=bit_rate,
                    channels=channels,
                    duration=duration,
                    has_audio=True
                )
                
            except Exception as e:
                logger.warning(f"FFmpeg quality assessment failed, trying moviepy: {e}")
        
        # Fallback to moviepy
        if VideoFileClip is not None:
            try:
                with VideoFileClip(str(video_path)) as clip:
                    if clip.audio is None:
                        return QualityMetrics(
                            sample_rate=0,
                            bit_rate=None,
                            channels=0,
                            duration=clip.duration,
                            has_audio=False
                        )
                    
                    return QualityMetrics(
                        sample_rate=clip.audio.fps if hasattr(clip.audio, 'fps') else 0,
                        bit_rate=None,  # MoviePy doesn't provide bit rate
                        channels=clip.audio.nchannels if hasattr(clip.audio, 'nchannels') else 0,
                        duration=clip.duration,
                        has_audio=True
                    )
                    
            except Exception as e:
                error_msg = f"MoviePy error while assessing quality: {str(e)}"
                logger.error(error_msg)
                raise VideoValidationError(error_msg)
        
        # If both methods failed
        raise VideoValidationError("No video processing library available (ffmpeg-python or moviepy required)")
