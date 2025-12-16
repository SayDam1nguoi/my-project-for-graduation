# -*- coding: utf-8 -*-
"""
MongoDB Manager Module

Quản lý kết nối MongoDB và lưu trữ kết quả thu âm, transcription, và phân tích cảm xúc.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import gridfs
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)


class MongoDBManager:
    """Quản lý kết nối và thao tác với MongoDB."""
    
    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        database_name: str = "emotion_scanner",
        timeout_ms: int = 5000
    ):
        """
        Khởi tạo MongoDB Manager.
        
        Args:
            connection_string: MongoDB connection string
            database_name: Tên database
            timeout_ms: Timeout cho kết nối (milliseconds)
        """
        self.connection_string = connection_string
        self.database_name = database_name
        self.timeout_ms = timeout_ms
        
        self.client: Optional[MongoClient] = None
        self.db = None
        self.fs: Optional[gridfs.GridFS] = None
        self._is_connected = False
        
        # Collection names
        self.RECORDINGS_COLLECTION = "audio_recordings"
        self.TRANSCRIPTIONS_COLLECTION = "transcriptions"
        self.EMOTIONS_COLLECTION = "emotion_analysis"
        self.VIDEOS_COLLECTION = "video_analysis"
    
    def connect(self) -> bool:
        """
        Kết nối đến MongoDB.
        
        Returns:
            True nếu kết nối thành công, False nếu thất bại
        """
        try:
            # Tạo MongoDB client
            self.client = MongoClient(
                self.connection_string,
                serverSelectionTimeoutMS=self.timeout_ms
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            # Get database
            self.db = self.client[self.database_name]
            
            # Initialize GridFS for file storage
            self.fs = gridfs.GridFS(self.db)
            
            # Create indexes
            self._create_indexes()
            
            self._is_connected = True
            logger.info(f"✅ Đã kết nối MongoDB: {self.database_name}")
            
            return True
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"❌ Không thể kết nối MongoDB: {e}")
            self._is_connected = False
            return False
        except Exception as e:
            logger.error(f"❌ Lỗi kết nối MongoDB: {e}")
            self._is_connected = False
            return False
    
    def _create_indexes(self):
        """Tạo indexes cho các collection."""
        try:
            # Audio recordings indexes
            self.db[self.RECORDINGS_COLLECTION].create_index([("created_at", DESCENDING)])
            self.db[self.RECORDINGS_COLLECTION].create_index([("file_name", ASCENDING)])
            
            # Transcriptions indexes
            self.db[self.TRANSCRIPTIONS_COLLECTION].create_index([("recording_id", ASCENDING)])
            self.db[self.TRANSCRIPTIONS_COLLECTION].create_index([("created_at", DESCENDING)])
            
            # Emotions indexes
            self.db[self.EMOTIONS_COLLECTION].create_index([("recording_id", ASCENDING)])
            self.db[self.EMOTIONS_COLLECTION].create_index([("created_at", DESCENDING)])
            
            logger.info("✅ Đã tạo indexes cho MongoDB")
            
        except Exception as e:
            logger.warning(f"⚠️ Không thể tạo indexes: {e}")
    
    def is_connected(self) -> bool:
        """Kiểm tra trạng thái kết nối."""
        if not self._is_connected or not self.client:
            return False
        
        try:
            # Ping server to check connection
            self.client.admin.command('ping')
            return True
        except Exception:
            self._is_connected = False
            return False
    
    def disconnect(self):
        """Ngắt kết nối MongoDB."""
        if self.client:
            self.client.close()
            self._is_connected = False
            logger.info("🔌 Đã ngắt kết nối MongoDB")
    
    def save_audio_recording(
        self,
        file_path: str,
        duration_seconds: float,
        sample_rate: int,
        bit_depth: int,
        channels: int,
        file_size_bytes: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Lưu thông tin bản thu âm vào MongoDB.
        
        Args:
            file_path: Đường dẫn file audio
            duration_seconds: Thời lượng (giây)
            sample_rate: Sample rate (Hz)
            bit_depth: Bit depth
            channels: Số channels
            file_size_bytes: Kích thước file (bytes)
            metadata: Metadata bổ sung
            
        Returns:
            ID của document đã lưu, hoặc None nếu thất bại
        """
        if not self.is_connected():
            logger.error("❌ Chưa kết nối MongoDB")
            return None
        
        try:
            file_path_obj = Path(file_path)
            
            # Prepare document
            document = {
                "file_name": file_path_obj.name,
                "file_path": str(file_path_obj.absolute()),
                "duration_seconds": duration_seconds,
                "sample_rate": sample_rate,
                "bit_depth": bit_depth,
                "channels": channels,
                "file_size_bytes": file_size_bytes,
                "created_at": datetime.now(),
                "metadata": metadata or {}
            }
            
            # Insert document
            result = self.db[self.RECORDINGS_COLLECTION].insert_one(document)
            
            logger.info(f"✅ Đã lưu recording vào MongoDB: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"❌ Lỗi lưu recording: {e}")
            return None
    
    def save_transcription(
        self,
        recording_id: str,
        transcription_text: str,
        language: str = "vi",
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Lưu kết quả transcription vào MongoDB.
        
        Args:
            recording_id: ID của bản thu âm
            transcription_text: Nội dung transcription
            language: Ngôn ngữ
            confidence: Độ tin cậy (0-1)
            metadata: Metadata bổ sung
            
        Returns:
            ID của document đã lưu, hoặc None nếu thất bại
        """
        if not self.is_connected():
            logger.error("❌ Chưa kết nối MongoDB")
            return None
        
        try:
            document = {
                "recording_id": recording_id,
                "text": transcription_text,
                "language": language,
                "confidence": confidence,
                "word_count": len(transcription_text.split()),
                "char_count": len(transcription_text),
                "created_at": datetime.now(),
                "metadata": metadata or {}
            }
            
            result = self.db[self.TRANSCRIPTIONS_COLLECTION].insert_one(document)
            
            logger.info(f"✅ Đã lưu transcription vào MongoDB: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"❌ Lỗi lưu transcription: {e}")
            return None
    
    def save_emotion_analysis(
        self,
        recording_id: str,
        emotions: Dict[str, float],
        dominant_emotion: str,
        analysis_type: str = "audio",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Lưu kết quả phân tích cảm xúc vào MongoDB.
        
        Args:
            recording_id: ID của bản thu âm/video
            emotions: Dictionary các cảm xúc và điểm số
            dominant_emotion: Cảm xúc chủ đạo
            analysis_type: Loại phân tích ("audio", "video", "combined")
            metadata: Metadata bổ sung
            
        Returns:
            ID của document đã lưu, hoặc None nếu thất bại
        """
        if not self.is_connected():
            logger.error("❌ Chưa kết nối MongoDB")
            return None
        
        try:
            document = {
                "recording_id": recording_id,
                "emotions": emotions,
                "dominant_emotion": dominant_emotion,
                "analysis_type": analysis_type,
                "created_at": datetime.now(),
                "metadata": metadata or {}
            }
            
            result = self.db[self.EMOTIONS_COLLECTION].insert_one(document)
            
            logger.info(f"✅ Đã lưu emotion analysis vào MongoDB: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"❌ Lỗi lưu emotion analysis: {e}")
            return None
    
    def get_recording_by_id(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """Lấy thông tin recording theo ID."""
        if not self.is_connected():
            return None
        
        try:
            from bson.objectid import ObjectId
            result = self.db[self.RECORDINGS_COLLECTION].find_one({"_id": ObjectId(recording_id)})
            return result
        except Exception as e:
            logger.error(f"❌ Lỗi lấy recording: {e}")
            return None
    
    def get_recent_recordings(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Lấy danh sách recordings gần đây."""
        if not self.is_connected():
            return []
        
        try:
            cursor = self.db[self.RECORDINGS_COLLECTION].find().sort("created_at", DESCENDING).limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"❌ Lỗi lấy recordings: {e}")
            return []
    
    def get_transcription_by_recording_id(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """Lấy transcription theo recording ID."""
        if not self.is_connected():
            return None
        
        try:
            result = self.db[self.TRANSCRIPTIONS_COLLECTION].find_one({"recording_id": recording_id})
            return result
        except Exception as e:
            logger.error(f"❌ Lỗi lấy transcription: {e}")
            return None
    
    def get_emotion_analysis_by_recording_id(self, recording_id: str) -> Optional[Dict[str, Any]]:
        """Lấy emotion analysis theo recording ID."""
        if not self.is_connected():
            return None
        
        try:
            result = self.db[self.EMOTIONS_COLLECTION].find_one({"recording_id": recording_id})
            return result
        except Exception as e:
            logger.error(f"❌ Lỗi lấy emotion analysis: {e}")
            return None
    
    def delete_recording(self, recording_id: str) -> bool:
        """Xóa recording và các dữ liệu liên quan."""
        if not self.is_connected():
            return False
        
        try:
            from bson.objectid import ObjectId
            
            # Delete recording
            self.db[self.RECORDINGS_COLLECTION].delete_one({"_id": ObjectId(recording_id)})
            
            # Delete related transcriptions
            self.db[self.TRANSCRIPTIONS_COLLECTION].delete_many({"recording_id": recording_id})
            
            # Delete related emotion analysis
            self.db[self.EMOTIONS_COLLECTION].delete_many({"recording_id": recording_id})
            
            logger.info(f"✅ Đã xóa recording và dữ liệu liên quan: {recording_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Lỗi xóa recording: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Lấy thống kê database."""
        if not self.is_connected():
            return {}
        
        try:
            stats = {
                "total_recordings": self.db[self.RECORDINGS_COLLECTION].count_documents({}),
                "total_transcriptions": self.db[self.TRANSCRIPTIONS_COLLECTION].count_documents({}),
                "total_emotions": self.db[self.EMOTIONS_COLLECTION].count_documents({}),
                "database_name": self.database_name,
                "connection_string": self.connection_string
            }
            return stats
        except Exception as e:
            logger.error(f"❌ Lỗi lấy stats: {e}")
            return {}


# Singleton instance
_mongodb_manager: Optional[MongoDBManager] = None


def get_mongodb_manager(
    connection_string: str = "mongodb://localhost:27017/",
    database_name: str = "emotion_scanner"
) -> MongoDBManager:
    """
    Lấy singleton MongoDB manager instance.
    
    Args:
        connection_string: MongoDB connection string
        database_name: Tên database
        
    Returns:
        MongoDBManager instance
    """
    global _mongodb_manager
    
    if _mongodb_manager is None:
        _mongodb_manager = MongoDBManager(connection_string, database_name)
        _mongodb_manager.connect()
    
    return _mongodb_manager
