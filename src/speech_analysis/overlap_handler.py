"""
Overlap Handler Module for Enhanced Vietnamese STT

This module provides audio segment buffering and overlap handling capabilities:
- Audio segment buffering with timestamps
- Chronological ordering of segments
- FIFO buffer management
- Segment merging functionality
"""

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """Represents a segment of audio with metadata."""
    audio: np.ndarray
    start_time: float
    end_time: float
    is_speech: bool = True
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        """Get duration of the segment in seconds."""
        return self.end_time - self.start_time


class OverlapHandler:
    """
    Overlap handler for managing audio buffer and processing overlapping segments.
    
    Provides buffering, queuing, and chronological ordering of audio segments
    to handle overlapping audio and ensure no data loss.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        overlap_duration: float = 0.5,
        max_buffer_size: int = 10
    ):
        """
        Initialize overlap handler.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            overlap_duration: Duration of overlap between segments in seconds (default: 0.5)
            max_buffer_size: Maximum number of segments to buffer (default: 10)
        """
        self.sample_rate = sample_rate
        self.overlap_duration = overlap_duration
        self.max_buffer_size = max_buffer_size
        
        # Use deque for efficient FIFO operations
        self._buffer: deque[AudioSegment] = deque(maxlen=max_buffer_size)
        
        logger.info(
            f"OverlapHandler initialized: sample_rate={sample_rate}, "
            f"overlap_duration={overlap_duration}, max_buffer_size={max_buffer_size}"
        )
    
    def add_segment(
        self, 
        audio: np.ndarray, 
        timestamp: float,
        is_speech: bool = True,
        confidence: float = 1.0
    ) -> None:
        """
        Add audio segment to buffer with timestamp.
        
        Args:
            audio: Audio data as numpy array
            timestamp: Start timestamp of the segment in seconds
            is_speech: Whether the segment contains speech (default: True)
            confidence: Confidence score for the segment (default: 1.0)
        """
        # Calculate end time based on audio length
        duration = len(audio) / self.sample_rate
        end_time = timestamp + duration
        
        segment = AudioSegment(
            audio=audio,
            start_time=timestamp,
            end_time=end_time,
            is_speech=is_speech,
            confidence=confidence
        )
        
        # Add to buffer (deque will automatically remove oldest if at capacity)
        if len(self._buffer) >= self.max_buffer_size:
            logger.warning(
                f"Buffer at capacity ({self.max_buffer_size}). "
                f"Oldest segment will be removed (FIFO)."
            )
        
        self._buffer.append(segment)
        
        logger.debug(
            f"Added segment: start={timestamp:.2f}s, end={end_time:.2f}s, "
            f"duration={duration:.2f}s, buffer_size={len(self._buffer)}"
        )
    
    def get_next_segment(self) -> Optional[AudioSegment]:
        """
        Get next segment for processing in chronological order.
        
        Returns:
            Next AudioSegment in chronological order, or None if buffer is empty
        """
        if not self._buffer:
            return None
        
        # Find segment with earliest start time
        earliest_idx = 0
        earliest_time = self._buffer[0].start_time
        
        for i, segment in enumerate(self._buffer):
            if segment.start_time < earliest_time:
                earliest_time = segment.start_time
                earliest_idx = i
        
        # Remove and return the earliest segment
        # Convert deque to list for indexed removal
        segments_list = list(self._buffer)
        segment = segments_list.pop(earliest_idx)
        self._buffer = deque(segments_list, maxlen=self.max_buffer_size)
        
        logger.debug(
            f"Retrieved segment: start={segment.start_time:.2f}s, "
            f"end={segment.end_time:.2f}s, remaining={len(self._buffer)}"
        )
        
        return segment
    
    def get_all_segments_ordered(self) -> List[AudioSegment]:
        """
        Get all segments in chronological order without removing them.
        
        Returns:
            List of AudioSegments sorted by start time
        """
        return sorted(self._buffer, key=lambda s: s.start_time)
    
    def merge_overlapping_segments(
        self, 
        segments: List[AudioSegment]
    ) -> AudioSegment:
        """
        Merge overlapping segments into continuous audio.
        
        Args:
            segments: List of AudioSegments to merge (should be sorted by start_time)
            
        Returns:
            Single merged AudioSegment with combined audio
            
        Raises:
            ValueError: If segments list is empty
        """
        if not segments:
            raise ValueError("Cannot merge empty segments list")
        
        # Sort segments by start time to ensure proper ordering
        sorted_segments = sorted(segments, key=lambda s: s.start_time)
        
        # Calculate total duration and find earliest/latest times
        earliest_start = sorted_segments[0].start_time
        latest_end = max(s.end_time for s in sorted_segments)
        total_duration = latest_end - earliest_start
        
        # Calculate total samples needed
        total_samples = int(total_duration * self.sample_rate)
        
        # Create output array
        merged_audio = np.zeros(total_samples, dtype=np.float32)
        overlap_count = np.zeros(total_samples, dtype=np.int32)
        
        # Merge segments with overlap handling
        for segment in sorted_segments:
            # Calculate position in merged array
            start_sample = int((segment.start_time - earliest_start) * self.sample_rate)
            end_sample = start_sample + len(segment.audio)
            
            # Ensure we don't exceed array bounds
            end_sample = min(end_sample, total_samples)
            audio_length = end_sample - start_sample
            
            # Add audio to merged array (accumulate overlapping regions)
            merged_audio[start_sample:end_sample] += segment.audio[:audio_length]
            overlap_count[start_sample:end_sample] += 1
        
        # Average overlapping regions
        # Avoid division by zero
        overlap_count[overlap_count == 0] = 1
        merged_audio = merged_audio / overlap_count
        
        # Calculate average confidence
        avg_confidence = np.mean([s.confidence for s in sorted_segments])
        
        # Create merged segment
        merged_segment = AudioSegment(
            audio=merged_audio,
            start_time=earliest_start,
            end_time=latest_end,
            is_speech=any(s.is_speech for s in sorted_segments),
            confidence=avg_confidence
        )
        
        logger.debug(
            f"Merged {len(segments)} segments: "
            f"start={earliest_start:.2f}s, end={latest_end:.2f}s, "
            f"duration={total_duration:.2f}s"
        )
        
        return merged_segment
    
    def is_buffer_full(self) -> bool:
        """
        Check if buffer has reached capacity.
        
        Returns:
            True if buffer is at maximum capacity, False otherwise
        """
        return len(self._buffer) >= self.max_buffer_size
    
    def buffer_size(self) -> int:
        """
        Get current number of segments in buffer.
        
        Returns:
            Number of segments currently in buffer
        """
        return len(self._buffer)
    
    def clear_buffer(self) -> None:
        """Clear all segments from buffer."""
        self._buffer.clear()
        logger.debug("Buffer cleared")
    
    def get_overlapping_segments(
        self, 
        start_time: float, 
        end_time: float
    ) -> List[AudioSegment]:
        """
        Get all segments that overlap with the given time range.
        
        Args:
            start_time: Start of time range in seconds
            end_time: End of time range in seconds
            
        Returns:
            List of AudioSegments that overlap with the time range
        """
        overlapping = []
        
        for segment in self._buffer:
            # Check if segment overlaps with time range
            if segment.start_time < end_time and segment.end_time > start_time:
                overlapping.append(segment)
        
        # Sort by start time
        overlapping.sort(key=lambda s: s.start_time)
        
        return overlapping
    
    def has_overlap(self, segment1: AudioSegment, segment2: AudioSegment) -> bool:
        """
        Check if two segments overlap in time.
        
        Args:
            segment1: First audio segment
            segment2: Second audio segment
            
        Returns:
            True if segments overlap, False otherwise
        """
        return (segment1.start_time < segment2.end_time and 
                segment1.end_time > segment2.start_time)
