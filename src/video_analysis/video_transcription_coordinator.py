"""
Video Transcription Coordinator module.

This module coordinates all video transcription components to provide
a complete video-to-text transcription pipeline with subtitle generation.

Implements Requirements 7.1, 7.2, 9.1, 9.3, 9.4, 9.5
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List, Callable, Dict, Any
from pathlib import Path

import numpy as np

from .video_validator import VideoValidator, ValidationResult, VideoValidationError
from .audio_extractor import AudioExtractor, AudioExtractionError
from .language_detector import LanguageDetector, LanguageDetectionError
from .batch_processor import BatchProcessor, ProcessingProgress, TranscriptionError
from .timestamp_aligner import TimestampAligner, TranscriptionSegment, WordTimestamp
from .subtitle_generator import SubtitleGenerator
from .speaker_diarization import SpeakerDiarization, SpeakerSegment
from .transcription_cache import TranscriptionCache, CacheInfo
from ..speech_analysis.whisper_stt_engine import WhisperSTTEngine
from ..speech_analysis.config import VideoTranscriptionConfig
from ..speech_analysis.speech_to_text import TranscriptionResult


logger = logging.getLogger(__name__)


class SubtitleGenerationError(Exception):
    """Raised when subtitle generation fails."""
    pass


@dataclass
class VideoTranscriptionResult:
    """Complete result from video transcription."""
    video_path: str
    duration: float
    language: str
    segments: List[TranscriptionSegment]
    full_text: str
    processing_time: float
    confidence_avg: float
    speakers: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for caching."""
        return {
            'video_path': self.video_path,
            'duration': self.duration,
            'language': self.language,
            'segments': [
                {
                    'text': seg.text,
                    'start': seg.start,
                    'end': seg.end,
                    'speaker': seg.speaker,
                    'language': seg.language,
                    'words': [
                        {
                            'word': w.word,
                            'start': w.start,
                            'end': w.end,
                            'confidence': w.confidence,
                            'speaker': w.speaker
                        }
                        for w in seg.words
                    ]
                }
                for seg in self.segments
            ],
            'full_text': self.full_text,
            'processing_time': self.processing_time,
            'confidence_avg': self.confidence_avg,
            'speakers': self.speakers,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoTranscriptionResult':
        """Create result from dictionary (for cache retrieval)."""
        # Reconstruct segments
        segments = []
        for seg_data in data.get('segments', []):
            words = [
                WordTimestamp(
                    word=w['word'],
                    start=w['start'],
                    end=w['end'],
                    confidence=w['confidence'],
                    speaker=w.get('speaker')
                )
                for w in seg_data.get('words', [])
            ]
            
            segment = TranscriptionSegment(
                text=seg_data['text'],
                start=seg_data['start'],
                end=seg_data['end'],
                words=words,
                speaker=seg_data.get('speaker'),
                language=seg_data.get('language', 'vi')
            )
            segments.append(segment)
        
        return cls(
            video_path=data['video_path'],
            duration=data['duration'],
            language=data['language'],
            segments=segments,
            full_text=data['full_text'],
            processing_time=data['processing_time'],
            confidence_avg=data['confidence_avg'],
            speakers=data.get('speakers'),
            metadata=data.get('metadata', {})
        )


@dataclass
class BatchTranscriptionSummary:
    """Summary of batch transcription results."""
    total_files: int
    successful: int
    failed: int
    total_duration: float
    total_processing_time: float
    results: List[VideoTranscriptionResult]
    errors: Dict[str, str] = field(default_factory=dict)


class VideoTranscriptionCoordinator:
    """
    Coordinates all video transcription components.
    
    Provides a complete pipeline from video file to transcription and subtitles.
    Implements caching, batch processing, and comprehensive error handling.
    """
    
    def __init__(
        self,
        whisper_engine: WhisperSTTEngine,
        config: Optional[VideoTranscriptionConfig] = None
    ):
        """
        Initialize video transcription coordinator.
        
        Args:
            whisper_engine: Whisper STT engine instance
            config: Video transcription configuration (uses default if None)
        """
        self.whisper_engine = whisper_engine
        self.config = config or VideoTranscriptionConfig()
        
        # Validate configuration
        self.config.validate()
        
        # Initialize components
        self.validator = VideoValidator()
        self.audio_extractor = AudioExtractor(
            target_sample_rate=16000,
            target_channels=1,
            enable_enhancement=False  # Tắt enhancement để tránh lỗi
        )
        self.language_detector = LanguageDetector(
            whisper_model=whisper_engine.model,
            default_language=self.config.default_language
        )
        self.batch_processor = BatchProcessor(
            chunk_duration=self.config.chunk_duration,
            overlap_duration=self.config.overlap_duration,
            max_parallel=self.config.max_parallel_chunks,
            sample_rate=16000
        )
        self.timestamp_aligner = TimestampAligner()
        self.subtitle_generator = SubtitleGenerator(
            max_chars_per_line=self.config.subtitle_max_chars_per_line,
            max_lines=self.config.subtitle_max_lines,
            min_duration=self.config.subtitle_min_duration,
            max_duration=self.config.subtitle_max_duration
        )
        
        # Optional components
        self.speaker_diarization = None
        if self.config.enable_diarization:
            self.speaker_diarization = SpeakerDiarization(
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers
            )
        
        # Cache
        self.cache = None
        if self.config.enable_cache:
            self.cache = TranscriptionCache(
                cache_dir=self.config.cache_dir,
                max_size_mb=self.config.cache_max_size_mb
            )
        
        logger.info(
            f"VideoTranscriptionCoordinator initialized: "
            f"model={self.config.whisper_model_size}, "
            f"cache={'enabled' if self.config.enable_cache else 'disabled'}, "
            f"diarization={'enabled' if self.config.enable_diarization else 'disabled'}"
        )
    
    def transcribe_video(
        self,
        video_path: str,
        language: Optional[str] = None,
        enable_diarization: Optional[bool] = None,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> VideoTranscriptionResult:
        """
        Transcribe video file.
        
        Requirement 7.1: Cache storage with video file hash
        Requirement 7.2: Verify cache validity before reuse
        
        Args:
            video_path: Path to video file
            language: Language code (None = auto-detect)
            enable_diarization: Enable speaker diarization (None = use config)
            progress_callback: Callback for progress updates
            
        Returns:
            VideoTranscriptionResult with complete transcription
            
        Raises:
            VideoValidationError: If video validation fails
            AudioExtractionError: If audio extraction fails
            TranscriptionError: If transcription fails
        """
        start_time = time.time()
        
        logger.info(f"Starting video transcription: {video_path}")
        
        # Check cache first (Requirement 7.1, 7.2)
        if self.cache:
            cached_result = self.cache.get(video_path)
            if cached_result:
                logger.info(f"Cache hit for {video_path}")
                # Convert cached dict to VideoTranscriptionResult
                result = VideoTranscriptionResult.from_dict(
                    cached_result.get('transcription', cached_result)
                )
                return result
            else:
                logger.info(f"Cache miss for {video_path}")
        
        try:
            # Step 1: Validate video
            logger.info("Step 1: Validating video")
            validation_result = self.validator.validate_video(video_path)
            
            if not validation_result.is_valid:
                raise VideoValidationError(
                    f"Video validation failed: {validation_result.error_message}"
                )
            
            if not validation_result.has_audio:
                raise VideoValidationError(
                    f"Video has no audio track: {video_path}"
                )
            
            video_duration = validation_result.duration
            logger.info(f"Video validated: duration={video_duration:.2f}s")
            
            # Step 2: Extract audio
            logger.info("Step 2: Extracting audio")
            audio = self.audio_extractor.extract_audio(video_path)
            logger.info(f"Audio extracted: {len(audio)} samples")
            
            # Step 3: Detect language (if not specified)
            if language is None and self.config.auto_detect_language:
                logger.info("Step 3: Detecting language")
                try:
                    lang_result = self.language_detector.detect_language(
                        audio,
                        sample_duration=self.config.language_detection_duration
                    )
                    language = lang_result.language
                    logger.info(
                        f"Language detected: {language} "
                        f"(confidence: {lang_result.confidence:.3f})"
                    )
                except LanguageDetectionError as e:
                    logger.warning(f"Language detection failed: {e}, using default")
                    language = self.config.default_language
            else:
                language = language or self.config.default_language
                logger.info(f"Using language: {language}")
            
            # Update whisper engine language
            self.whisper_engine.language = language
            
            # Step 4: Transcribe audio with batch processing
            logger.info("Step 4: Transcribing audio")
            
            def transcribe_fn(audio_chunk: np.ndarray) -> TranscriptionResult:
                """Transcribe function for batch processor."""
                return self.whisper_engine.transcribe_chunk(audio_chunk)
            
            transcription_result = self.batch_processor.process_audio(
                audio=audio,
                video_path=video_path,
                transcribe_fn=transcribe_fn,
                progress_callback=progress_callback
            )
            
            logger.info(
                f"Transcription completed: {len(transcription_result.segments)} segments"
            )
            
            # Step 5: Convert to TranscriptionSegment format with word timestamps
            logger.info("Step 5: Processing segments")
            segments = self._convert_to_transcription_segments(
                transcription_result.segments
            )
            
            # Step 6: Speaker diarization (optional)
            speaker_segments = None
            speakers = None
            
            if enable_diarization is None:
                enable_diarization = self.config.enable_diarization
            
            if enable_diarization and self.speaker_diarization:
                logger.info("Step 6: Performing speaker diarization")
                try:
                    speaker_segments = self.speaker_diarization.diarize(audio)
                    
                    # Assign speakers to segments
                    segments = self.speaker_diarization.assign_speakers_to_transcription(
                        segments,
                        speaker_segments
                    )
                    
                    # Get unique speakers
                    speakers = list(set(seg.speaker for seg in segments if seg.speaker))
                    logger.info(f"Speaker diarization completed: {len(speakers)} speakers")
                    
                except Exception as e:
                    logger.warning(f"Speaker diarization failed: {e}")
            
            # Step 7: Align timestamps
            logger.info("Step 7: Aligning timestamps")
            # Extract word timestamps from segments
            all_words = []
            for seg in segments:
                all_words.extend(seg.words)
            
            # Correct drift and validate
            if all_words:
                aligned_words = self.timestamp_aligner.correct_drift(
                    all_words,
                    video_duration
                )
                
                # Validate timestamps
                is_valid = self.timestamp_aligner.validate_timestamps(
                    aligned_words,
                    video_duration
                )
                
                if not is_valid:
                    logger.warning("Timestamp validation failed, using original timestamps")
                else:
                    # Update segments with aligned words
                    segments = self._update_segments_with_words(segments, aligned_words)
            
            # Step 8: Create final result
            full_text = " ".join(seg.text for seg in segments if seg.text.strip())
            
            # Filter hallucinations from segments
            try:
                from src.speech_analysis.hallucination_filter import get_hallucination_filter
                halluc_filter = get_hallucination_filter()
                
                # Filter segments
                filtered_segments, num_filtered = halluc_filter.filter_segments(segments)
                
                if num_filtered > 0:
                    logger.warning(f"Filtered {num_filtered} hallucinated segments from video transcription")
                    segments = filtered_segments
                    # Recreate full text from filtered segments
                    full_text = " ".join(seg.text for seg in segments if seg.text.strip())
                
                # Also check full text
                filtered_full_text, was_filtered, reason = halluc_filter.filter_text(full_text)
                if was_filtered:
                    logger.warning(f"Full transcription text is hallucination: {reason}")
                    full_text = filtered_full_text  # Will be empty
                    segments = []  # Clear all segments
            except Exception as e:
                logger.warning(f"Could not apply hallucination filter: {e}")
            
            # Calculate average confidence
            if segments:
                total_confidence = sum(
                    sum(w.confidence for w in seg.words) / len(seg.words)
                    if seg.words else 0.0
                    for seg in segments
                )
                confidence_avg = total_confidence / len(segments)
            else:
                confidence_avg = 0.0
            
            processing_time = time.time() - start_time
            
            result = VideoTranscriptionResult(
                video_path=video_path,
                duration=video_duration,
                language=language,
                segments=segments,
                full_text=full_text,
                processing_time=processing_time,
                confidence_avg=confidence_avg,
                speakers=speakers,
                metadata={
                    'audio_samples': len(audio),
                    'num_segments': len(segments),
                    'diarization_enabled': enable_diarization,
                    'cache_enabled': self.config.enable_cache
                }
            )
            
            # Cache result (Requirement 7.1)
            if self.cache:
                logger.info("Caching transcription result")
                self.cache.put(video_path, result.to_dict())
            
            logger.info(
                f"Video transcription completed: "
                f"duration={video_duration:.2f}s, "
                f"processing_time={processing_time:.2f}s, "
                f"confidence={confidence_avg:.3f}"
            )
            
            return result
            
        except (VideoValidationError, AudioExtractionError, TranscriptionError):
            # Re-raise known errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            error_msg = f"Video transcription failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise TranscriptionError(error_msg) from e
    
    def transcribe_batch(
        self,
        video_paths: List[str],
        language: Optional[str] = None,
        enable_diarization: Optional[bool] = None,
        progress_callback: Optional[Callable[[ProcessingProgress], None]] = None
    ) -> BatchTranscriptionSummary:
        """
        Transcribe multiple video files in batch.
        
        Requirement 9.1: Queue all files for processing
        Requirement 9.3: Skip failed file and continue with remaining files
        Requirement 9.4: Report overall progress across all files
        Requirement 9.5: Generate summary report with success and failure counts
        
        Args:
            video_paths: List of video file paths
            language: Language code (None = auto-detect for each)
            enable_diarization: Enable speaker diarization
            progress_callback: Callback for progress updates
            
        Returns:
            BatchTranscriptionSummary with results and statistics
        """
        logger.info(f"Starting batch transcription: {len(video_paths)} files")
        
        start_time = time.time()
        results = []
        errors = {}
        total_duration = 0.0
        
        # Requirement 9.1: Queue all files
        for file_index, video_path in enumerate(video_paths):
            try:
                logger.info(
                    f"Processing file {file_index + 1}/{len(video_paths)}: {video_path}"
                )
                
                # Create progress callback for this file
                def file_progress_callback(progress: ProcessingProgress):
                    # Requirement 9.4: Report overall progress
                    overall_progress = ProcessingProgress(
                        current_file=video_path,
                        total_files=len(video_paths),
                        current_file_index=file_index,
                        current_chunk=progress.current_chunk,
                        total_chunks=progress.total_chunks,
                        percentage=(file_index * 100 + progress.percentage) / len(video_paths),
                        elapsed_time=time.time() - start_time,
                        estimated_remaining=progress.estimated_remaining,
                        status=progress.status
                    )
                    
                    if progress_callback:
                        progress_callback(overall_progress)
                
                # Transcribe video
                result = self.transcribe_video(
                    video_path=video_path,
                    language=language,
                    enable_diarization=enable_diarization,
                    progress_callback=file_progress_callback
                )
                
                results.append(result)
                total_duration += result.duration
                
                logger.info(f"Successfully transcribed: {video_path}")
                
            except Exception as e:
                # Requirement 9.3: Skip failed file and continue
                error_msg = str(e)
                errors[video_path] = error_msg
                logger.error(f"Failed to transcribe {video_path}: {error_msg}")
                # Continue with next file
                continue
        
        total_processing_time = time.time() - start_time
        
        # Requirement 9.5: Generate summary report
        summary = BatchTranscriptionSummary(
            total_files=len(video_paths),
            successful=len(results),
            failed=len(errors),
            total_duration=total_duration,
            total_processing_time=total_processing_time,
            results=results,
            errors=errors
        )
        
        logger.info(
            f"Batch transcription completed: "
            f"successful={summary.successful}/{summary.total_files}, "
            f"failed={summary.failed}, "
            f"total_time={total_processing_time:.2f}s"
        )
        
        return summary
    
    def export_subtitles(
        self,
        result: VideoTranscriptionResult,
        output_path: str,
        format: str = "srt",
        include_speaker: bool = False
    ) -> None:
        """
        Export subtitles from transcription result.
        
        Args:
            result: Video transcription result
            output_path: Path to output subtitle file
            format: Subtitle format ("srt" or "vtt")
            include_speaker: Include speaker labels in subtitles
            
        Raises:
            SubtitleGenerationError: If subtitle generation fails
        """
        logger.info(f"Exporting subtitles: format={format}, output={output_path}")
        
        try:
            if format.lower() == "srt":
                self.subtitle_generator.generate_srt(
                    segments=result.segments,
                    output_path=output_path,
                    include_speaker=include_speaker
                )
            elif format.lower() == "vtt":
                self.subtitle_generator.generate_vtt(
                    segments=result.segments,
                    output_path=output_path,
                    include_speaker=include_speaker
                )
            else:
                raise SubtitleGenerationError(
                    f"Unsupported subtitle format: {format}. Use 'srt' or 'vtt'"
                )
            
            logger.info(f"Subtitles exported successfully: {output_path}")
            
        except Exception as e:
            error_msg = f"Failed to export subtitles: {str(e)}"
            logger.error(error_msg)
            raise SubtitleGenerationError(error_msg) from e
    
    def get_cache_info(self) -> Optional[CacheInfo]:
        """
        Get cache information.
        
        Returns:
            CacheInfo object or None if cache is disabled
        """
        if self.cache:
            return self.cache.get_cache_info()
        return None
    
    def clear_cache(self) -> None:
        """Clear transcription cache."""
        if self.cache:
            logger.info("Clearing transcription cache")
            self.cache.clear()
    
    def _convert_to_transcription_segments(
        self,
        stt_segments: List
    ) -> List[TranscriptionSegment]:
        """
        Convert STT segments to TranscriptionSegment format.
        
        Args:
            stt_segments: Segments from STT engine
            
        Returns:
            List of TranscriptionSegment with word timestamps
        """
        result_segments = []
        
        for seg in stt_segments:
            # Create word timestamps (placeholder - Whisper provides these)
            words = []
            
            # For now, create simple word timestamps by splitting text
            # In real implementation, Whisper provides word-level timestamps
            text_words = seg.text.split()
            if text_words:
                duration = seg.end_time - seg.start_time
                word_duration = duration / len(text_words)
                
                for i, word in enumerate(text_words):
                    word_start = seg.start_time + (i * word_duration)
                    word_end = word_start + word_duration
                    
                    words.append(WordTimestamp(
                        word=word,
                        start=word_start,
                        end=word_end,
                        confidence=seg.confidence
                    ))
            
            result_segments.append(TranscriptionSegment(
                text=seg.text,
                start=seg.start_time,
                end=seg.end_time,
                words=words,
                language=self.whisper_engine.language
            ))
        
        return result_segments
    
    def _update_segments_with_words(
        self,
        segments: List[TranscriptionSegment],
        aligned_words: List[WordTimestamp]
    ) -> List[TranscriptionSegment]:
        """
        Update segments with aligned word timestamps.
        
        Args:
            segments: Original segments
            aligned_words: Aligned word timestamps
            
        Returns:
            Updated segments
        """
        # Create a mapping of words to segments
        updated_segments = []
        word_index = 0
        
        for seg in segments:
            # Find words that belong to this segment
            seg_words = []
            
            while word_index < len(aligned_words):
                word = aligned_words[word_index]
                
                # Check if word belongs to this segment (by time overlap)
                if word.start >= seg.start and word.end <= seg.end + 0.5:
                    seg_words.append(word)
                    word_index += 1
                elif word.start > seg.end:
                    # Word is beyond this segment
                    break
                else:
                    # Word might overlap, include it
                    seg_words.append(word)
                    word_index += 1
            
            # Update segment with aligned words
            updated_seg = TranscriptionSegment(
                text=seg.text,
                start=seg.start,
                end=seg.end,
                words=seg_words if seg_words else seg.words,
                speaker=seg.speaker,
                language=seg.language
            )
            updated_segments.append(updated_seg)
        
        return updated_segments
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up VideoTranscriptionCoordinator")
        
        if self.batch_processor:
            self.batch_processor.cleanup()
        
        if self.cache:
            self.cache.close()
        
        logger.info("Cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False
