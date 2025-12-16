"""
Evidence Retrieval System for Recruitment Emotion Scoring.

This module provides video clip extraction, caching, and mapping between
criterion scores and supporting video segments to provide evidence for
scoring decisions.

Requirements: 14.5
"""

import os
import cv2
import hashlib
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict

from .models import CriterionScore, KeyMoment, EmotionReport


@dataclass
class VideoClipMetadata:
    """
    Metadata for an extracted video clip.
    
    Attributes:
        clip_path: Path to the extracted clip file
        source_video: Path to source video
        start_time: Start timestamp in seconds
        end_time: End timestamp in seconds
        criterion: Associated criterion name
        score: Associated score value
        cache_key: Unique cache key for this clip
    """
    clip_path: str
    source_video: str
    start_time: float
    end_time: float
    criterion: str
    score: float
    cache_key: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VideoClipMetadata':
        """Create from dictionary."""
        return cls(**data)


class EvidenceRetrievalSystem:
    """
    System for retrieving video evidence for emotion scoring.
    
    Provides functionality to:
    - Extract video clips for specific timestamps
    - Map criterion scores to supporting video segments
    - Cache clips to avoid redundant extraction
    
    Requirements:
        - 14.5: Show video clips corresponding to specific emotional assessments
    """
    
    def __init__(
        self,
        cache_dir: str = "reports/video_clips",
        cache_index_file: str = "reports/clip_cache_index.json"
    ):
        """
        Initialize Evidence Retrieval System.
        
        Args:
            cache_dir: Directory for storing extracted video clips
            cache_index_file: Path to cache index file
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_index_file = Path(cache_index_file)
        self.cache_index_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load cache index
        self.cache_index: Dict[str, VideoClipMetadata] = self._load_cache_index()
        
        print(f"EvidenceRetrievalSystem initialized")
        print(f"  Cache directory: {self.cache_dir}")
        print(f"  Cache index: {self.cache_index_file}")
        print(f"  Cached clips: {len(self.cache_index)}")
    
    def extract_clip_for_timestamp(
        self,
        video_path: str,
        timestamp: float,
        criterion: str,
        score: float,
        clip_duration: float = 3.0
    ) -> Optional[str]:
        """
        Extract video clip for a specific timestamp.
        
        Extracts a video clip centered around the given timestamp.
        Uses caching to avoid redundant extraction.
        
        Args:
            video_path: Path to source video file
            timestamp: Center timestamp in seconds
            criterion: Associated criterion name
            score: Associated score value
            clip_duration: Total duration of clip in seconds
        
        Returns:
            Path to extracted clip file, or None if extraction fails
        
        Requirements:
            - 14.5: Show video clips corresponding to specific emotional assessments
        """
        # Calculate start and end times
        start_time = max(0, timestamp - clip_duration / 2)
        end_time = timestamp + clip_duration / 2
        
        # Generate cache key
        cache_key = self._generate_cache_key(
            video_path, start_time, end_time, criterion
        )
        
        # Check cache
        if cache_key in self.cache_index:
            cached_clip = self.cache_index[cache_key]
            if os.path.exists(cached_clip.clip_path):
                print(f"  Using cached clip: {cached_clip.clip_path}")
                return cached_clip.clip_path
            else:
                # Remove invalid cache entry
                del self.cache_index[cache_key]
                self._save_cache_index()
        
        # Extract new clip
        clip_path = self._extract_video_segment(
            video_path, start_time, end_time, criterion, timestamp
        )
        
        if clip_path:
            # Add to cache
            metadata = VideoClipMetadata(
                clip_path=clip_path,
                source_video=video_path,
                start_time=start_time,
                end_time=end_time,
                criterion=criterion,
                score=score,
                cache_key=cache_key
            )
            self.cache_index[cache_key] = metadata
            self._save_cache_index()
            
            print(f"  Extracted and cached clip: {clip_path}")
        
        return clip_path
    
    def create_criterion_evidence_mapping(
        self,
        report: EmotionReport,
        video_path: str,
        clip_duration: float = 3.0
    ) -> Dict[str, List[str]]:
        """
        Create mapping between criterion scores and supporting video segments.
        
        For each criterion in the report, extracts video clips that provide
        evidence for the score. Uses key moments and evidence timestamps.
        
        Args:
            report: EmotionReport containing scores and key moments
            video_path: Path to source video file
            clip_duration: Duration of each clip in seconds
        
        Returns:
            Dictionary mapping criterion names to lists of clip paths
        
        Requirements:
            - 14.5: Show video clips corresponding to specific emotional assessments
        """
        print(f"\nCreating criterion evidence mapping...")
        
        criterion_clips: Dict[str, List[str]] = {}
        
        # Process each criterion
        for criterion_name, criterion_score in report.criterion_scores.items():
            clips = []
            
            # Extract clips from evidence timestamps
            timestamps = criterion_score.evidence_timestamps[:5]  # Limit to 5 clips per criterion
            
            for timestamp in timestamps:
                clip_path = self.extract_clip_for_timestamp(
                    video_path=video_path,
                    timestamp=timestamp,
                    criterion=criterion_name,
                    score=criterion_score.score,
                    clip_duration=clip_duration
                )
                
                if clip_path:
                    clips.append(clip_path)
            
            # Also extract clips from key moments related to this criterion
            criterion_moments = [
                m for m in report.key_moments 
                if m.criterion == criterion_name
            ]
            
            for moment in criterion_moments[:3]:  # Limit to 3 additional clips
                if moment.timestamp not in timestamps:
                    clip_path = self.extract_clip_for_timestamp(
                        video_path=video_path,
                        timestamp=moment.timestamp,
                        criterion=criterion_name,
                        score=criterion_score.score,
                        clip_duration=clip_duration
                    )
                    
                    if clip_path:
                        clips.append(clip_path)
            
            criterion_clips[criterion_name] = clips
            print(f"  {criterion_name}: {len(clips)} clips")
        
        print(f"  Total clips extracted: {sum(len(clips) for clips in criterion_clips.values())}")
        
        return criterion_clips
    
    def retrieve_evidence_for_criterion(
        self,
        criterion_name: str,
        criterion_score: CriterionScore,
        video_path: str,
        max_clips: int = 5,
        clip_duration: float = 3.0
    ) -> List[str]:
        """
        Retrieve video evidence for a specific criterion.
        
        Extracts video clips that provide evidence for the given criterion score.
        
        Args:
            criterion_name: Name of the criterion
            criterion_score: CriterionScore object
            video_path: Path to source video file
            max_clips: Maximum number of clips to extract
            clip_duration: Duration of each clip in seconds
        
        Returns:
            List of paths to extracted video clips
        
        Requirements:
            - 14.5: Show video clips corresponding to specific emotional assessments
        """
        print(f"\nRetrieving evidence for criterion: {criterion_name}")
        
        clips = []
        timestamps = criterion_score.evidence_timestamps[:max_clips]
        
        for timestamp in timestamps:
            clip_path = self.extract_clip_for_timestamp(
                video_path=video_path,
                timestamp=timestamp,
                criterion=criterion_name,
                score=criterion_score.score,
                clip_duration=clip_duration
            )
            
            if clip_path:
                clips.append(clip_path)
        
        print(f"  Retrieved {len(clips)} evidence clips")
        
        return clips
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cached video clips.
        
        Args:
            older_than_days: Only clear clips older than this many days.
                           If None, clears all clips.
        
        Returns:
            Number of clips removed
        """
        print(f"\nClearing cache...")
        
        removed_count = 0
        
        if older_than_days is None:
            # Remove all clips
            for cache_key, metadata in list(self.cache_index.items()):
                if os.path.exists(metadata.clip_path):
                    try:
                        os.remove(metadata.clip_path)
                        removed_count += 1
                    except Exception as e:
                        print(f"  Warning: Could not remove {metadata.clip_path}: {e}")
                
                del self.cache_index[cache_key]
        else:
            # Remove old clips (implementation would check file modification time)
            import time
            cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
            
            for cache_key, metadata in list(self.cache_index.items()):
                if os.path.exists(metadata.clip_path):
                    file_mtime = os.path.getmtime(metadata.clip_path)
                    if file_mtime < cutoff_time:
                        try:
                            os.remove(metadata.clip_path)
                            removed_count += 1
                            del self.cache_index[cache_key]
                        except Exception as e:
                            print(f"  Warning: Could not remove {metadata.clip_path}: {e}")
        
        self._save_cache_index()
        
        print(f"  Removed {removed_count} cached clips")
        
        return removed_count
    
    def get_cache_stats(self) -> Dict[str, any]:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        total_clips = len(self.cache_index)
        total_size = 0
        criteria_counts = {}
        
        for metadata in self.cache_index.values():
            if os.path.exists(metadata.clip_path):
                total_size += os.path.getsize(metadata.clip_path)
            
            criterion = metadata.criterion
            criteria_counts[criterion] = criteria_counts.get(criterion, 0) + 1
        
        return {
            'total_clips': total_clips,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'clips_per_criterion': criteria_counts,
            'cache_dir': str(self.cache_dir),
        }
    
    # Private helper methods
    
    def _generate_cache_key(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        criterion: str
    ) -> str:
        """
        Generate unique cache key for a video clip.
        
        Args:
            video_path: Path to source video
            start_time: Start timestamp
            end_time: End timestamp
            criterion: Criterion name
        
        Returns:
            Unique cache key string
        """
        # Create a unique identifier based on video path, timestamps, and criterion
        key_string = f"{video_path}:{start_time:.3f}:{end_time:.3f}:{criterion}"
        
        # Hash to create shorter key
        hash_obj = hashlib.md5(key_string.encode())
        cache_key = hash_obj.hexdigest()
        
        return cache_key
    
    def _extract_video_segment(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        criterion: str,
        center_timestamp: float
    ) -> Optional[str]:
        """
        Extract a video segment from source video.
        
        Args:
            video_path: Path to source video file
            start_time: Start time in seconds
            end_time: End time in seconds
            criterion: Criterion name for filename
            center_timestamp: Center timestamp for filename
        
        Returns:
            Path to extracted clip, or None if extraction fails
        """
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"  Error: Could not open video: {video_path}")
                return None
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate frame numbers
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Generate output filename
            video_name = Path(video_path).stem
            clip_filename = f"{video_name}_{criterion}_{center_timestamp:.2f}s.mp4"
            clip_path = str(self.cache_dir / clip_filename)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
            
            # Set to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Extract frames
            frames_written = 0
            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
                frames_written += 1
            
            # Release resources
            out.release()
            cap.release()
            
            if frames_written == 0:
                print(f"  Warning: No frames extracted for clip")
                if os.path.exists(clip_path):
                    os.remove(clip_path)
                return None
            
            return clip_path
            
        except Exception as e:
            print(f"  Error extracting video segment: {e}")
            return None
    
    def _load_cache_index(self) -> Dict[str, VideoClipMetadata]:
        """
        Load cache index from file.
        
        Returns:
            Dictionary mapping cache keys to VideoClipMetadata
        """
        if not self.cache_index_file.exists():
            return {}
        
        try:
            with open(self.cache_index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            cache_index = {}
            for cache_key, metadata_dict in data.items():
                cache_index[cache_key] = VideoClipMetadata.from_dict(metadata_dict)
            
            return cache_index
            
        except Exception as e:
            print(f"  Warning: Could not load cache index: {e}")
            return {}
    
    def _save_cache_index(self) -> None:
        """Save cache index to file."""
        try:
            data = {
                cache_key: metadata.to_dict()
                for cache_key, metadata in self.cache_index.items()
            }
            
            with open(self.cache_index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"  Warning: Could not save cache index: {e}")
