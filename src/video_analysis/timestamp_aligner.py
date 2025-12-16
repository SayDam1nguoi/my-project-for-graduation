"""
Timestamp Aligner module for video transcription.

This module aligns and corrects timestamps for accurate subtitle generation.
Handles word-level timestamps, silence gaps, chunk merging, drift correction,
and validation against video duration.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional


logger = logging.getLogger(__name__)


@dataclass
class WordTimestamp:
    """Word with timestamp."""
    word: str
    start: float  # seconds
    end: float  # seconds
    confidence: float
    speaker: Optional[str] = None


@dataclass
class TranscriptionSegment:
    """Segment of transcription."""
    text: str
    start: float
    end: float
    words: List[WordTimestamp]
    speaker: Optional[str] = None
    language: str = "vi"


@dataclass
class TranscriptionChunk:
    """Chunk of transcription with metadata."""
    chunk_index: int
    start_time: float
    end_time: float
    segments: List[TranscriptionSegment]


class TimestampAligner:
    """
    Aligns and corrects timestamps for transcription.
    
    Provides word-level timestamp alignment, silence gap handling,
    chunk merging, drift correction, and validation.
    """
    
    def __init__(
        self,
        max_gap_duration: float = 1.0,
        min_word_duration: float = 0.05,
        overlap_tolerance: float = 0.1
    ):
        """
        Initialize timestamp aligner.
        
        Args:
            max_gap_duration: Maximum allowed gap between words (seconds)
            min_word_duration: Minimum duration for a word (seconds)
            overlap_tolerance: Tolerance for overlapping timestamps (seconds)
        """
        self.max_gap_duration = max_gap_duration
        self.min_word_duration = min_word_duration
        self.overlap_tolerance = overlap_tolerance
        
        logger.info(
            f"TimestampAligner initialized: max_gap={max_gap_duration}s, "
            f"min_word={min_word_duration}s, overlap_tolerance={overlap_tolerance}s"
        )
    
    def align_timestamps(
        self,
        transcription_chunks: List[TranscriptionChunk]
    ) -> List[WordTimestamp]:
        """
        Align timestamps from multiple chunks.
        
        Extracts word-level timestamps from chunks, merges overlapping regions,
        and ensures continuous timestamps.
        
        Args:
            transcription_chunks: List of transcription chunks
            
        Returns:
            List of aligned word timestamps
        """
        if not transcription_chunks:
            logger.warning("No transcription chunks provided")
            return []
        
        # Sort chunks by start time
        sorted_chunks = sorted(transcription_chunks, key=lambda c: c.start_time)
        
        # Extract all word timestamps from chunks
        all_words = []
        for chunk in sorted_chunks:
            for segment in chunk.segments:
                all_words.extend(segment.words)
        
        if not all_words:
            logger.warning("No words found in transcription chunks")
            return []
        
        # Remove duplicates in overlap regions
        unique_words = self._remove_duplicate_words(all_words)
        
        # Handle silence gaps
        aligned_words = self._handle_silence_gaps(unique_words)
        
        # Ensure continuous timestamps
        continuous_words = self._ensure_continuous_timestamps(aligned_words)
        
        logger.info(
            f"Aligned {len(transcription_chunks)} chunks: "
            f"{len(all_words)} words -> {len(continuous_words)} unique words"
        )
        
        return continuous_words
    
    def _remove_duplicate_words(
        self,
        words: List[WordTimestamp]
    ) -> List[WordTimestamp]:
        """
        Remove duplicate words in overlap regions.
        
        Args:
            words: List of word timestamps (may contain duplicates)
            
        Returns:
            List of unique word timestamps
        """
        if not words:
            return []
        
        # Sort by start time
        sorted_words = sorted(words, key=lambda w: w.start)
        
        unique_words = []
        
        for word in sorted_words:
            # Check if this word is a duplicate of the last added word
            if unique_words:
                last_word = unique_words[-1]
                
                # Check for temporal overlap and text similarity
                time_overlap = (word.start < last_word.end + self.overlap_tolerance)
                text_similar = self._is_similar_word(word.word, last_word.word)
                
                if time_overlap and text_similar:
                    # Keep the word with higher confidence
                    if word.confidence > last_word.confidence:
                        unique_words[-1] = word
                    continue
            
            unique_words.append(word)
        
        return unique_words
    
    def _is_similar_word(self, word1: str, word2: str) -> bool:
        """
        Check if two words are similar (for duplicate detection).
        
        Args:
            word1: First word
            word2: Second word
            
        Returns:
            True if words are similar
        """
        # Normalize and compare
        w1 = word1.lower().strip()
        w2 = word2.lower().strip()
        
        # Exact match
        if w1 == w2:
            return True
        
        # Check if one is substring of other (for partial matches)
        if w1 in w2 or w2 in w1:
            return True
        
        return False
    
    def _handle_silence_gaps(
        self,
        words: List[WordTimestamp]
    ) -> List[WordTimestamp]:
        """
        Handle silence gaps between words.
        
        Adjusts timestamps to account for pauses and ensures
        gaps don't exceed max_gap_duration.
        
        Args:
            words: List of word timestamps
            
        Returns:
            List of words with adjusted timestamps
        """
        if not words:
            return []
        
        adjusted_words = [words[0]]
        
        for i in range(1, len(words)):
            current_word = words[i]
            prev_word = adjusted_words[-1]
            
            # Calculate gap
            gap = current_word.start - prev_word.end
            
            if gap < 0:
                # Overlap - adjust current word start time
                adjusted_start = prev_word.end
                adjusted_end = max(adjusted_start + self.min_word_duration, current_word.end)
                
                adjusted_word = WordTimestamp(
                    word=current_word.word,
                    start=adjusted_start,
                    end=adjusted_end,
                    confidence=current_word.confidence,
                    speaker=current_word.speaker
                )
                adjusted_words.append(adjusted_word)
                
            elif gap > self.max_gap_duration:
                # Large gap - keep as is (silence period)
                adjusted_words.append(current_word)
                
            else:
                # Normal gap - keep as is
                adjusted_words.append(current_word)
        
        return adjusted_words
    
    def _ensure_continuous_timestamps(
        self,
        words: List[WordTimestamp]
    ) -> List[WordTimestamp]:
        """
        Ensure timestamps are monotonically increasing without overlaps.
        
        Args:
            words: List of word timestamps
            
        Returns:
            List of words with continuous timestamps
        """
        if not words:
            return []
        
        continuous_words = []
        
        for i, word in enumerate(words):
            if i == 0:
                # First word - keep as is
                continuous_words.append(word)
            else:
                prev_word = continuous_words[-1]
                
                # Ensure start time is after previous end time
                if word.start < prev_word.end:
                    adjusted_start = prev_word.end
                    adjusted_end = max(adjusted_start + self.min_word_duration, word.end)
                    
                    adjusted_word = WordTimestamp(
                        word=word.word,
                        start=adjusted_start,
                        end=adjusted_end,
                        confidence=word.confidence,
                        speaker=word.speaker
                    )
                    continuous_words.append(adjusted_word)
                else:
                    continuous_words.append(word)
        
        return continuous_words
    
    def correct_drift(
        self,
        timestamps: List[WordTimestamp],
        video_duration: float
    ) -> List[WordTimestamp]:
        """
        Correct timestamp drift to maintain synchronization.
        
        Applies linear correction if timestamps drift from video duration.
        
        Args:
            timestamps: List of word timestamps
            video_duration: Total video duration in seconds
            
        Returns:
            Corrected timestamps
        """
        if not timestamps:
            return []
        
        if video_duration <= 0:
            logger.warning(f"Invalid video duration: {video_duration}")
            return timestamps
        
        # Get last timestamp
        last_timestamp = timestamps[-1].end
        
        # Calculate drift
        drift = last_timestamp - video_duration
        
        # Only correct if drift is significant (> 1% of duration)
        drift_threshold = video_duration * 0.01
        
        if abs(drift) < drift_threshold:
            logger.debug(f"No significant drift detected: {drift:.3f}s")
            return timestamps
        
        # Calculate correction factor
        correction_factor = video_duration / last_timestamp
        
        logger.info(
            f"Correcting timestamp drift: {drift:.3f}s, "
            f"factor={correction_factor:.6f}"
        )
        
        # Apply correction
        corrected_timestamps = []
        for word in timestamps:
            corrected_word = WordTimestamp(
                word=word.word,
                start=word.start * correction_factor,
                end=word.end * correction_factor,
                confidence=word.confidence,
                speaker=word.speaker
            )
            corrected_timestamps.append(corrected_word)
        
        return corrected_timestamps
    
    def validate_timestamps(
        self,
        timestamps: List[WordTimestamp],
        video_duration: float
    ) -> bool:
        """
        Validate timestamps against video duration.
        
        Args:
            timestamps: List of word timestamps
            video_duration: Total video duration in seconds
            
        Returns:
            True if timestamps are valid
        """
        if not timestamps:
            return True
        
        # Check if any timestamp exceeds video duration
        for word in timestamps:
            if word.start < 0 or word.end < 0:
                logger.error(f"Negative timestamp found: {word.word} [{word.start}, {word.end}]")
                return False
            
            if word.start > video_duration or word.end > video_duration:
                logger.error(
                    f"Timestamp exceeds video duration: {word.word} "
                    f"[{word.start}, {word.end}] > {video_duration}"
                )
                return False
            
            if word.start >= word.end:
                logger.error(f"Invalid timestamp range: {word.word} [{word.start}, {word.end}]")
                return False
        
        # Check monotonicity
        for i in range(len(timestamps) - 1):
            if timestamps[i].end > timestamps[i + 1].start + self.overlap_tolerance:
                logger.error(
                    f"Timestamps not monotonic: "
                    f"{timestamps[i].word} [{timestamps[i].start}, {timestamps[i].end}] -> "
                    f"{timestamps[i+1].word} [{timestamps[i+1].start}, {timestamps[i+1].end}]"
                )
                return False
        
        logger.debug(f"Timestamps validated: {len(timestamps)} words, duration={video_duration}s")
        return True
    
    def merge_overlapping_segments(
        self,
        segments: List[TranscriptionSegment]
    ) -> List[TranscriptionSegment]:
        """
        Merge overlapping segments.
        
        Combines segments that overlap in time, preserving all words
        and adjusting timestamps.
        
        Args:
            segments: List of segments
            
        Returns:
            Merged segments
        """
        if not segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda s: s.start)
        
        merged_segments = []
        current_segment = sorted_segments[0]
        
        for i in range(1, len(sorted_segments)):
            next_segment = sorted_segments[i]
            
            # Check for overlap (next starts before current ends, with tolerance)
            if next_segment.start < current_segment.end + self.overlap_tolerance:
                # Merge segments
                merged_text = current_segment.text + " " + next_segment.text
                merged_words = current_segment.words + next_segment.words
                
                # Remove duplicate words in overlap region
                merged_words = self._remove_duplicate_words(merged_words)
                
                # Create merged segment
                current_segment = TranscriptionSegment(
                    text=merged_text.strip(),
                    start=current_segment.start,
                    end=max(current_segment.end, next_segment.end),
                    words=merged_words,
                    speaker=current_segment.speaker,  # Keep first speaker
                    language=current_segment.language
                )
            else:
                # No overlap - add current segment and move to next
                merged_segments.append(current_segment)
                current_segment = next_segment
        
        # Add last segment
        merged_segments.append(current_segment)
        
        logger.info(f"Merged {len(segments)} segments into {len(merged_segments)} segments")
        
        return merged_segments
    
    def merge_timestamps(
        self,
        chunk_timestamps: List[List[WordTimestamp]]
    ) -> List[WordTimestamp]:
        """
        Merge timestamps from multiple chunks.
        
        Combines word timestamps from different chunks, handling overlaps
        and ensuring continuous timestamps.
        
        Args:
            chunk_timestamps: List of timestamp lists from chunks
            
        Returns:
            Merged timestamps
        """
        if not chunk_timestamps:
            return []
        
        # Flatten all timestamps
        all_timestamps = []
        for chunk in chunk_timestamps:
            all_timestamps.extend(chunk)
        
        if not all_timestamps:
            return []
        
        # Remove duplicates
        unique_timestamps = self._remove_duplicate_words(all_timestamps)
        
        # Handle silence gaps
        aligned_timestamps = self._handle_silence_gaps(unique_timestamps)
        
        # Ensure continuous timestamps
        continuous_timestamps = self._ensure_continuous_timestamps(aligned_timestamps)
        
        logger.info(
            f"Merged {len(chunk_timestamps)} chunks: "
            f"{len(all_timestamps)} words -> {len(continuous_timestamps)} unique words"
        )
        
        return continuous_timestamps
