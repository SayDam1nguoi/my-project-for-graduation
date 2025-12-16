"""
Speaker Diarization module for video transcription.

This module identifies and labels different speakers in audio.
Implements Requirements 6.1, 6.2, 6.3, 6.4, 6.5
"""

import numpy as np
import tempfile
import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """Speaker segment in audio."""
    start: float
    end: float
    speaker: str
    confidence: float = 1.0


class SpeakerDiarization:
    """
    Identifies and labels speakers in audio.
    
    This class provides speaker diarization functionality using pyannote.audio.
    It can detect the number of speakers, assign consistent labels, mark transitions,
    and handle overlapping speech.
    """
    
    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization",
        min_speakers: int = 1,
        max_speakers: int = 10
    ):
        """
        Initialize speaker diarization.
        
        Args:
            model_name: Name of diarization model
            min_speakers: Minimum number of speakers to detect
            max_speakers: Maximum number of speakers to detect
        """
        self.model_name = model_name
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.model = None
        self._speaker_label_map: Dict[str, str] = {}
        self._next_speaker_id = 0
        
    def _load_model(self):
        """Load the pyannote.audio model lazily."""
        if self.model is not None:
            return
            
        try:
            from pyannote.audio import Pipeline
            self.model = Pipeline.from_pretrained(self.model_name)
            logger.info(f"Loaded speaker diarization model: {self.model_name}")
        except ImportError:
            logger.error("pyannote.audio not installed. Install with: pip install pyannote.audio")
            raise ImportError(
                "pyannote.audio is required for speaker diarization. "
                "Install with: pip install pyannote.audio torch"
            )
        except Exception as e:
            logger.error(f"Failed to load diarization model: {e}")
            raise
    
    def _get_consistent_speaker_label(self, original_label: str) -> str:
        """
        Get consistent speaker label.
        
        Requirement 6.2: Assign consistent labels throughout video.
        
        Args:
            original_label: Original label from diarization model
            
        Returns:
            Consistent speaker label (e.g., "Speaker_0", "Speaker_1")
        """
        if original_label not in self._speaker_label_map:
            self._speaker_label_map[original_label] = f"Speaker_{self._next_speaker_id}"
            self._next_speaker_id += 1
        return self._speaker_label_map[original_label]
    
    def _detect_num_speakers(self, speaker_segments: List[SpeakerSegment]) -> int:
        """
        Detect number of unique speakers.
        
        Requirement 6.1: Detect number of unique speakers.
        
        Args:
            speaker_segments: List of speaker segments
            
        Returns:
            Number of unique speakers
        """
        unique_speakers = set(seg.speaker for seg in speaker_segments)
        return len(unique_speakers)
    
    def _detect_transitions(
        self,
        speaker_segments: List[SpeakerSegment]
    ) -> List[Tuple[float, str, str]]:
        """
        Detect speaker transitions.
        
        Requirement 6.3: Mark transition points with timestamps.
        
        Args:
            speaker_segments: List of speaker segments
            
        Returns:
            List of (timestamp, from_speaker, to_speaker) tuples
        """
        transitions = []
        for i in range(len(speaker_segments) - 1):
            current = speaker_segments[i]
            next_seg = speaker_segments[i + 1]
            
            if current.speaker != next_seg.speaker:
                # Transition occurs at the end of current segment
                transitions.append((current.end, current.speaker, next_seg.speaker))
        
        return transitions
    
    def _handle_overlapping_speech(
        self,
        speaker_segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        """
        Handle overlapping speech segments.
        
        Requirement 6.4: Handle simultaneous speakers appropriately.
        
        When speakers overlap, we keep both segments and mark them with
        combined labels (e.g., "Speaker_0+Speaker_1").
        
        Args:
            speaker_segments: List of speaker segments
            
        Returns:
            Processed segments with overlap handling
        """
        if not speaker_segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(speaker_segments, key=lambda x: x.start)
        
        # Detect overlaps and create combined segments
        result = []
        i = 0
        
        while i < len(sorted_segments):
            current = sorted_segments[i]
            overlapping = [current]
            
            # Find all segments that overlap with current
            j = i + 1
            while j < len(sorted_segments):
                next_seg = sorted_segments[j]
                # Check if next segment overlaps with any in overlapping list
                if any(self._segments_overlap(next_seg, seg) for seg in overlapping):
                    overlapping.append(next_seg)
                    j += 1
                else:
                    break
            
            if len(overlapping) == 1:
                # No overlap, add as is
                result.append(current)
            else:
                # Multiple overlapping speakers
                # Create segments for the overlap period
                result.extend(self._merge_overlapping_segments(overlapping))
            
            i = j if j > i + 1 else i + 1
        
        return sorted(result, key=lambda x: x.start)
    
    def _segments_overlap(self, seg1: SpeakerSegment, seg2: SpeakerSegment) -> bool:
        """Check if two segments overlap."""
        return not (seg1.end <= seg2.start or seg2.end <= seg1.start)
    
    def _merge_overlapping_segments(
        self,
        segments: List[SpeakerSegment]
    ) -> List[SpeakerSegment]:
        """
        Merge overlapping segments into combined speaker labels.
        
        Args:
            segments: List of overlapping segments
            
        Returns:
            List of merged segments
        """
        # Find all unique time points
        time_points = set()
        for seg in segments:
            time_points.add(seg.start)
            time_points.add(seg.end)
        
        time_points = sorted(time_points)
        result = []
        
        # For each time interval, determine which speakers are active
        for i in range(len(time_points) - 1):
            start = time_points[i]
            end = time_points[i + 1]
            
            # Find active speakers in this interval
            active_speakers = []
            for seg in segments:
                if seg.start <= start and seg.end >= end:
                    active_speakers.append(seg.speaker)
            
            if active_speakers:
                # Create combined label for overlapping speakers
                if len(active_speakers) > 1:
                    speaker_label = "+".join(sorted(active_speakers))
                else:
                    speaker_label = active_speakers[0]
                
                result.append(SpeakerSegment(
                    start=start,
                    end=end,
                    speaker=speaker_label,
                    confidence=1.0
                ))
        
        return result
    
    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> List[SpeakerSegment]:
        """
        Perform speaker diarization.
        
        Requirements 6.1, 6.2, 6.3, 6.4: Detect speakers, assign labels,
        mark transitions, handle overlapping speech.
        
        Args:
            audio: Audio data (numpy array, float32, mono)
            sample_rate: Sample rate (default 16000)
            
        Returns:
            List of speaker segments with consistent labels
        """
        self._load_model()
        
        # Reset speaker mapping for new audio
        self._speaker_label_map = {}
        self._next_speaker_id = 0
        
        # Save audio to temporary file (pyannote requires file input)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Write audio to file
            import soundfile as sf
            sf.write(tmp_path, audio, sample_rate)
            
            # Run diarization
            diarization = self.model(tmp_path, num_speakers=None)
            
            # Convert to our format with consistent labels
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                consistent_label = self._get_consistent_speaker_label(speaker)
                segments.append(SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=consistent_label,
                    confidence=1.0
                ))
            
            # Handle overlapping speech
            segments = self._handle_overlapping_speech(segments)
            
            # Log speaker information
            num_speakers = self._detect_num_speakers(segments)
            transitions = self._detect_transitions(segments)
            
            logger.info(f"Detected {num_speakers} unique speakers")
            logger.info(f"Found {len(transitions)} speaker transitions")
            
            return segments
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def assign_speakers_to_transcription(
        self,
        segments: List,
        speaker_segments: List[SpeakerSegment]
    ) -> List:
        """
        Assign speaker labels to transcription segments.
        
        Requirement 6.5: Include speaker labels in transcription output.
        
        For each transcription segment, find the speaker segment with the
        most overlap and assign that speaker label.
        
        Args:
            segments: Transcription segments (from timestamp_aligner.TranscriptionSegment)
            speaker_segments: Speaker segments from diarization
            
        Returns:
            Transcription segments with speaker labels assigned
        """
        # Import here to avoid circular dependency
        from .timestamp_aligner import TranscriptionSegment
        
        if not speaker_segments:
            logger.warning("No speaker segments provided")
            return segments
        
        result = []
        
        for trans_seg in segments:
            # Find speaker segment with most overlap
            best_speaker = None
            max_overlap = 0.0
            
            for spk_seg in speaker_segments:
                overlap = self._calculate_overlap(
                    trans_seg.start, trans_seg.end,
                    spk_seg.start, spk_seg.end
                )
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = spk_seg.speaker
            
            # Create new segment with speaker label
            new_seg = TranscriptionSegment(
                text=trans_seg.text,
                start=trans_seg.start,
                end=trans_seg.end,
                words=trans_seg.words,
                speaker=best_speaker,
                language=trans_seg.language
            )
            result.append(new_seg)
        
        return result
    
    def _calculate_overlap(
        self,
        start1: float,
        end1: float,
        start2: float,
        end2: float
    ) -> float:
        """
        Calculate overlap duration between two time intervals.
        
        Args:
            start1, end1: First interval
            start2, end2: Second interval
            
        Returns:
            Overlap duration in seconds
        """
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_start < overlap_end:
            return overlap_end - overlap_start
        return 0.0
    
    def get_speaker_statistics(
        self,
        speaker_segments: List[SpeakerSegment]
    ) -> Dict[str, Dict[str, float]]:
        """
        Get statistics about speaker activity.
        
        Args:
            speaker_segments: List of speaker segments
            
        Returns:
            Dictionary mapping speaker to statistics (total_time, num_segments)
        """
        stats = defaultdict(lambda: {"total_time": 0.0, "num_segments": 0})
        
        for seg in speaker_segments:
            duration = seg.end - seg.start
            stats[seg.speaker]["total_time"] += duration
            stats[seg.speaker]["num_segments"] += 1
        
        return dict(stats)
