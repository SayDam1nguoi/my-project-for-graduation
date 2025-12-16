# -*- coding: utf-8 -*-
"""
Data Models for Appearance Assessment

Defines dataclasses for lighting, clothing, and overall appearance assessments.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class LightingAssessment:
    """Đánh giá chất lượng ánh sáng."""
    score: float  # 0-100
    brightness: float  # 0-255
    contrast: float  # 0-100
    uniformity: float  # 0-100
    is_good: bool
    issues: List[str]
    recommendations: List[str]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LightingAssessment':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ClothingAssessment:
    """Đánh giá trang phục chuyên nghiệp."""
    score: float  # 0-100 (overall)
    dominant_color: str
    
    # New detailed fields
    clothing_type: str = ""  # formal_shirt, casual_shirt, t_shirt, suit_jacket, casual_jacket
    neatness_score: float = 0.0  # 0-100
    pattern_type: str = ""  # solid, striped, checkered, complex
    accessories: List[str] = field(default_factory=list)  # tie, glasses, etc.
    
    # Sub-scores breakdown
    color_score: float = 0.0  # 0-100
    type_score: float = 0.0  # 0-100
    pattern_score: float = 0.0  # 0-100
    accessories_score: float = 0.0  # 0-100
    
    # Legacy fields (for backward compatibility)
    professional_color_ratio: float = 0.0  # 0-100
    color_appropriateness: float = 0.0  # 0-100
    
    is_professional: bool = False
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClothingAssessment':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AppearanceAssessment:
    """Đánh giá tổng thể về ánh sáng và quần áo."""
    lighting: Optional[LightingAssessment]
    clothing: Optional[ClothingAssessment]
    overall_score: float  # 0-100
    is_interview_ready: bool
    timestamp: float
    frame_number: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            'lighting': self.lighting.to_dict() if self.lighting else None,
            'clothing': self.clothing.to_dict() if self.clothing else None,
            'overall_score': self.overall_score,
            'is_interview_ready': self.is_interview_ready,
            'timestamp': self.timestamp,
            'frame_number': self.frame_number
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppearanceAssessment':
        """Create from dictionary."""
        lighting = LightingAssessment.from_dict(data['lighting']) if data.get('lighting') else None
        clothing = ClothingAssessment.from_dict(data['clothing']) if data.get('clothing') else None
        return cls(
            lighting=lighting,
            clothing=clothing,
            overall_score=data['overall_score'],
            is_interview_ready=data['is_interview_ready'],
            timestamp=data['timestamp'],
            frame_number=data['frame_number']
        )


@dataclass
class AppearanceWarning:
    """Cảnh báo về vấn đề ánh sáng hoặc quần áo."""
    warning_type: str  # "lighting" or "clothing"
    severity: str  # "low", "medium", "high"
    message: str
    recommendations: List[str]
    timestamp: float
    should_display: bool  # Based on duration and threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppearanceWarning':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AppearanceReport:
    """Báo cáo đánh giá ánh sáng và quần áo cho một phiên."""
    session_id: str
    candidate_id: str
    start_time: datetime
    end_time: datetime
    total_frames: int
    
    # Aggregated scores
    average_lighting_score: float
    average_clothing_score: float
    average_overall_score: float
    
    # Timeline data
    lighting_timeline: List[tuple]  # (timestamp, score)
    clothing_timeline: List[tuple]
    
    # Issues summary
    lighting_issues_count: Dict[str, int]
    clothing_issues_count: Dict[str, int]
    
    # Final assessment
    is_interview_ready: bool
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'session_id': self.session_id,
            'candidate_id': self.candidate_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'total_frames': self.total_frames,
            'average_lighting_score': self.average_lighting_score,
            'average_clothing_score': self.average_clothing_score,
            'average_overall_score': self.average_overall_score,
            'lighting_timeline': self.lighting_timeline,
            'clothing_timeline': self.clothing_timeline,
            'lighting_issues_count': self.lighting_issues_count,
            'clothing_issues_count': self.clothing_issues_count,
            'is_interview_ready': self.is_interview_ready,
            'recommendations': self.recommendations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AppearanceReport':
        """Create from dictionary."""
        return cls(
            session_id=data['session_id'],
            candidate_id=data['candidate_id'],
            start_time=datetime.fromisoformat(data['start_time']),
            end_time=datetime.fromisoformat(data['end_time']),
            total_frames=data['total_frames'],
            average_lighting_score=data['average_lighting_score'],
            average_clothing_score=data['average_clothing_score'],
            average_overall_score=data['average_overall_score'],
            lighting_timeline=data['lighting_timeline'],
            clothing_timeline=data['clothing_timeline'],
            lighting_issues_count=data['lighting_issues_count'],
            clothing_issues_count=data['clothing_issues_count'],
            is_interview_ready=data['is_interview_ready'],
            recommendations=data['recommendations']
        )
