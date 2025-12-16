"""
Transcription Cache module for video transcription.

This module caches transcription results for fast retrieval using LRU eviction policy.
"""

import hashlib
import time
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path
import diskcache


@dataclass
class CacheInfo:
    """Cache information."""
    total_entries: int
    total_size_mb: float
    hit_rate: float
    miss_rate: float


class TranscriptionCache:
    """Caches transcription results with LRU eviction policy."""
    
    def __init__(
        self,
        cache_dir: str = ".cache/transcriptions",
        max_size_mb: int = 1000
    ):
        """
        Initialize transcription cache.
        
        Args:
            cache_dir: Cache directory path
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_mb = max_size_mb
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize diskcache with LRU eviction policy
        # size_limit is in bytes
        self.cache = diskcache.Cache(
            str(self.cache_dir),
            size_limit=max_size_mb * 1024 * 1024,
            eviction_policy='least-recently-used'
        )
        
        # Statistics tracking
        self._hits = 0
        self._misses = 0
    
    def get(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        Get cached transcription.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Cached transcription or None if cache miss
        """
        start_time = time.time()
        
        # Compute file hash as cache key
        try:
            cache_key = self._compute_file_hash(video_path)
        except (FileNotFoundError, IOError):
            # File doesn't exist or can't be read
            self._misses += 1
            return None
        
        # Try to retrieve from cache
        cached_data = self.cache.get(cache_key)
        
        if cached_data is not None:
            # Validate cache entry
            if self._validate_cache_entry(video_path, cached_data):
                self._hits += 1
                
                # Ensure retrieval is fast (< 100ms as per requirements)
                elapsed = time.time() - start_time
                if elapsed > 0.1:
                    # Log warning but still return cached data
                    print(f"Warning: Cache retrieval took {elapsed:.3f}s (> 100ms)")
                
                return cached_data
            else:
                # Invalid cache entry, remove it
                self.invalidate(video_path)
                self._misses += 1
                return None
        else:
            self._misses += 1
            return None
    
    def put(
        self,
        video_path: str,
        transcription: Dict[str, Any]
    ) -> None:
        """
        Store transcription in cache.
        
        Args:
            video_path: Path to video file
            transcription: Transcription result
        """
        try:
            cache_key = self._compute_file_hash(video_path)
            
            # Add metadata for validation
            cache_entry = {
                'transcription': transcription,
                'video_path': video_path,
                'file_hash': cache_key,
                'cached_at': time.time(),
                'file_size': os.path.getsize(video_path)
            }
            
            # Store in cache (LRU eviction happens automatically)
            self.cache.set(cache_key, cache_entry)
            
        except (FileNotFoundError, IOError) as e:
            # Can't compute hash or access file
            print(f"Warning: Failed to cache transcription for {video_path}: {e}")
    
    def invalidate(self, video_path: str) -> None:
        """
        Invalidate cache entry.
        
        Args:
            video_path: Path to video file
        """
        try:
            cache_key = self._compute_file_hash(video_path)
            self.cache.delete(cache_key)
        except (FileNotFoundError, IOError):
            # File doesn't exist, nothing to invalidate
            pass
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_cache_info(self) -> CacheInfo:
        """
        Get cache information.
        
        Returns:
            Cache information including size, entries, and hit rate
        """
        # Get cache statistics
        total_entries = len(self.cache)
        
        # Calculate total size in MB
        total_size_bytes = self.cache.volume()
        total_size_mb = total_size_bytes / (1024 * 1024)
        
        # Calculate hit and miss rates
        total_requests = self._hits + self._misses
        if total_requests > 0:
            hit_rate = self._hits / total_requests
            miss_rate = self._misses / total_requests
        else:
            hit_rate = 0.0
            miss_rate = 0.0
        
        return CacheInfo(
            total_entries=total_entries,
            total_size_mb=total_size_mb,
            hit_rate=hit_rate,
            miss_rate=miss_rate
        )
    
    def _compute_file_hash(self, video_path: str) -> str:
        """
        Compute hash of video file.
        
        Uses SHA256 hash of file content for cache key.
        For large files, only hashes first and last 1MB for performance.
        
        Args:
            video_path: Path to video file
            
        Returns:
            File hash (hex string)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file can't be read
        """
        path = Path(video_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        file_size = path.stat().st_size
        hash_obj = hashlib.sha256()
        
        # For performance, hash file metadata + sample of content
        # This is faster than hashing entire large video files
        hash_obj.update(str(file_size).encode())
        hash_obj.update(str(path.stat().st_mtime).encode())
        
        # Hash first 1MB
        chunk_size = 1024 * 1024  # 1MB
        with open(video_path, 'rb') as f:
            chunk = f.read(chunk_size)
            hash_obj.update(chunk)
            
            # If file is large, also hash last 1MB
            if file_size > chunk_size * 2:
                f.seek(-chunk_size, 2)  # Seek to 1MB before end
                chunk = f.read(chunk_size)
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    def _validate_cache_entry(
        self,
        video_path: str,
        cache_entry: Dict[str, Any]
    ) -> bool:
        """
        Validate cache entry.
        
        Checks if the cached entry is still valid by comparing file hash
        and file size.
        
        Args:
            video_path: Path to video file
            cache_entry: Cached entry to validate
            
        Returns:
            True if cache entry is valid, False otherwise
        """
        try:
            # Check if file still exists
            path = Path(video_path)
            if not path.exists():
                return False
            
            # Check if file size matches
            current_size = path.stat().st_size
            cached_size = cache_entry.get('file_size', 0)
            if current_size != cached_size:
                return False
            
            # Check if file hash matches
            current_hash = self._compute_file_hash(video_path)
            cached_hash = cache_entry.get('file_hash', '')
            if current_hash != cached_hash:
                return False
            
            # Cache entry is valid
            return True
            
        except (FileNotFoundError, IOError, KeyError):
            # Any error means cache is invalid
            return False
    
    def close(self) -> None:
        """Close the cache and release resources."""
        self.cache.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
