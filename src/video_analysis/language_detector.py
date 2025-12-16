"""
Language Detector module for video transcription.

This module detects the language in audio using Whisper's built-in
language detection capabilities.
"""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)


class LanguageDetectionError(Exception):
    """Raised when language detection fails."""
    pass


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    language: str  # ISO 639-1 language code (e.g., 'vi', 'en')
    confidence: float  # Confidence score [0.0, 1.0]
    all_probabilities: Dict[str, float] = field(default_factory=dict)  # All language probabilities
    sample_duration: float = 0.0  # Duration of audio sample analyzed
    
    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be in [0.0, 1.0], got {self.confidence}")


@dataclass
class LanguageSegment:
    """Segment with detected language."""
    language: str  # ISO 639-1 language code
    start_time: float  # Start time in seconds
    end_time: float  # End time in seconds
    confidence: float  # Confidence score [0.0, 1.0]
    text: Optional[str] = None  # Transcribed text (if available)


class LanguageDetector:
    """
    Detects language in audio using Whisper model.
    
    Uses Whisper's built-in language detection to identify the primary
    language in audio samples.
    """
    
    def __init__(
        self,
        whisper_model,
        default_language: str = "vi",
        confidence_threshold: float = 0.5
    ):
        """
        Initialize language detector.
        
        Args:
            whisper_model: Whisper model instance (from faster-whisper)
            default_language: Default language if detection fails
            confidence_threshold: Minimum confidence for detection
        """
        self.whisper_model = whisper_model
        self.default_language = default_language
        self.confidence_threshold = confidence_threshold
        
        logger.info(
            f"LanguageDetector initialized: "
            f"default={default_language}, threshold={confidence_threshold}"
        )
    
    def detect_language(
        self,
        audio: np.ndarray,
        sample_duration: float = 30.0
    ) -> LanguageDetectionResult:
        """
        Detect language in audio.
        
        Analyzes the first `sample_duration` seconds of audio to detect
        the primary language.
        
        Args:
            audio: Audio data (numpy array, float32, 16kHz mono)
            sample_duration: Duration of sample to analyze (default: 30.0 seconds)
            
        Returns:
            LanguageDetectionResult with detected language and confidence
            
        Raises:
            LanguageDetectionError: If detection fails
        """
        if len(audio) == 0:
            raise LanguageDetectionError("Audio is empty")
        
        # Convert audio to float32 if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        # Ensure audio is in correct range [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)
        
        # Calculate sample rate (assume 16kHz)
        sample_rate = 16000
        
        # Extract sample for analysis
        max_samples = int(sample_duration * sample_rate)
        audio_sample = audio[:max_samples]
        
        actual_duration = len(audio_sample) / sample_rate
        
        logger.info(
            f"Detecting language from {actual_duration:.2f}s audio sample"
        )
        
        try:
            # Use Whisper's detect_language method
            # This returns language probabilities
            language_info = self.whisper_model.detect_language(audio_sample)
            
            # Extract language and probability
            # faster-whisper returns (language_code, language_probability, all_probabilities)
            if isinstance(language_info, tuple):
                if len(language_info) >= 2:
                    detected_language = language_info[0]
                    confidence = language_info[1]
                    all_probs = language_info[2] if len(language_info) > 2 else {}
                else:
                    raise LanguageDetectionError(
                        f"Unexpected language_info format: {language_info}"
                    )
            else:
                raise LanguageDetectionError(
                    f"Unexpected language_info type: {type(language_info)}"
                )
            
            # Validate confidence
            if confidence < 0.0 or confidence > 1.0:
                logger.warning(
                    f"Invalid confidence {confidence}, clamping to [0, 1]"
                )
                confidence = max(0.0, min(1.0, confidence))
            
            # Check if confidence is below threshold
            if confidence < self.confidence_threshold:
                logger.warning(
                    f"Low confidence {confidence:.3f} for language '{detected_language}', "
                    f"using default language '{self.default_language}'"
                )
                detected_language = self.default_language
            
            logger.info(
                f"Language detected: {detected_language} "
                f"(confidence: {confidence:.3f})"
            )
            
            return LanguageDetectionResult(
                language=detected_language,
                confidence=confidence,
                all_probabilities=all_probs,
                sample_duration=actual_duration
            )
            
        except LanguageDetectionError:
            raise
        except Exception as e:
            error_msg = f"Language detection failed: {str(e)}"
            logger.error(error_msg)
            raise LanguageDetectionError(error_msg) from e
    
    def detect_multiple_languages(
        self,
        audio: np.ndarray,
        segment_duration: float = 30.0
    ) -> List[LanguageSegment]:
        """
        Detect multiple languages in audio by analyzing segments.
        
        Splits audio into segments and detects language in each segment
        to identify mixed-language content.
        
        Args:
            audio: Audio data (numpy array, float32, 16kHz mono)
            segment_duration: Duration of each segment to analyze (default: 30.0s)
            
        Returns:
            List of LanguageSegment with detected languages
            
        Raises:
            LanguageDetectionError: If detection fails
        """
        if len(audio) == 0:
            raise LanguageDetectionError("Audio is empty")
        
        # Convert audio to float32 if needed
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        # Ensure audio is in correct range [-1, 1]
        audio = np.clip(audio, -1.0, 1.0)
        
        # Calculate sample rate (assume 16kHz)
        sample_rate = 16000
        
        # Calculate total duration
        total_duration = len(audio) / sample_rate
        
        logger.info(
            f"Detecting multiple languages in {total_duration:.2f}s audio "
            f"(segment_duration={segment_duration}s)"
        )
        
        # Split audio into segments
        segment_samples = int(segment_duration * sample_rate)
        segments = []
        
        current_time = 0.0
        while current_time < total_duration:
            start_sample = int(current_time * sample_rate)
            end_sample = min(start_sample + segment_samples, len(audio))
            
            segment_audio = audio[start_sample:end_sample]
            segment_end_time = end_sample / sample_rate
            
            # Detect language in segment
            try:
                result = self.detect_language(
                    segment_audio,
                    sample_duration=segment_duration
                )
                
                segment = LanguageSegment(
                    language=result.language,
                    start_time=current_time,
                    end_time=segment_end_time,
                    confidence=result.confidence
                )
                segments.append(segment)
                
                logger.debug(
                    f"Segment [{current_time:.1f}s - {segment_end_time:.1f}s]: "
                    f"{result.language} (confidence: {result.confidence:.3f})"
                )
                
            except LanguageDetectionError as e:
                logger.warning(
                    f"Failed to detect language in segment "
                    f"[{current_time:.1f}s - {segment_end_time:.1f}s]: {e}"
                )
                # Use default language for failed segments
                segment = LanguageSegment(
                    language=self.default_language,
                    start_time=current_time,
                    end_time=segment_end_time,
                    confidence=0.0
                )
                segments.append(segment)
            
            # Move to next segment
            current_time = segment_end_time
        
        # Merge consecutive segments with same language
        merged_segments = self._merge_consecutive_segments(segments)
        
        logger.info(
            f"Detected {len(merged_segments)} language segment(s) "
            f"in {total_duration:.2f}s audio"
        )
        
        return merged_segments
    
    def _merge_consecutive_segments(
        self,
        segments: List[LanguageSegment]
    ) -> List[LanguageSegment]:
        """
        Merge consecutive segments with the same language.
        
        Args:
            segments: List of language segments
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
        
        merged = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            # Check if same language and consecutive
            if (next_segment.language == current_segment.language and
                abs(next_segment.start_time - current_segment.end_time) < 0.1):
                # Merge segments
                current_segment = LanguageSegment(
                    language=current_segment.language,
                    start_time=current_segment.start_time,
                    end_time=next_segment.end_time,
                    confidence=max(current_segment.confidence, next_segment.confidence)
                )
            else:
                # Different language or not consecutive, save current and start new
                merged.append(current_segment)
                current_segment = next_segment
        
        # Add last segment
        merged.append(current_segment)
        
        return merged
    
    def get_primary_language(
        self,
        segments: List[LanguageSegment]
    ) -> str:
        """
        Get the primary (most common) language from segments.
        
        Args:
            segments: List of language segments
            
        Returns:
            Primary language code
        """
        if not segments:
            return self.default_language
        
        # Calculate duration for each language
        language_durations: Dict[str, float] = {}
        
        for segment in segments:
            duration = segment.end_time - segment.start_time
            if segment.language in language_durations:
                language_durations[segment.language] += duration
            else:
                language_durations[segment.language] = duration
        
        # Find language with longest duration
        primary_language = max(
            language_durations.items(),
            key=lambda x: x[1]
        )[0]
        
        logger.info(
            f"Primary language: {primary_language} "
            f"({language_durations[primary_language]:.1f}s)"
        )
        
        return primary_language
    
    def is_mixed_language(
        self,
        segments: List[LanguageSegment],
        threshold: float = 0.1
    ) -> bool:
        """
        Check if audio contains mixed languages.
        
        Args:
            segments: List of language segments
            threshold: Minimum proportion for secondary language (default: 0.1 = 10%)
            
        Returns:
            True if audio contains significant mixed languages
        """
        if len(segments) <= 1:
            return False
        
        # Calculate total duration
        total_duration = sum(s.end_time - s.start_time for s in segments)
        
        if total_duration == 0:
            return False
        
        # Calculate duration for each language
        language_durations: Dict[str, float] = {}
        
        for segment in segments:
            duration = segment.end_time - segment.start_time
            if segment.language in language_durations:
                language_durations[segment.language] += duration
            else:
                language_durations[segment.language] = duration
        
        # Check if any secondary language exceeds threshold
        if len(language_durations) <= 1:
            return False
        
        # Sort by duration
        sorted_languages = sorted(
            language_durations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Check if second language exceeds threshold
        if len(sorted_languages) >= 2:
            secondary_proportion = sorted_languages[1][1] / total_duration
            is_mixed = secondary_proportion >= threshold
            
            if is_mixed:
                logger.info(
                    f"Mixed language detected: "
                    f"primary={sorted_languages[0][0]} ({sorted_languages[0][1]:.1f}s), "
                    f"secondary={sorted_languages[1][0]} ({sorted_languages[1][1]:.1f}s)"
                )
            
            return is_mixed
        
        return False
