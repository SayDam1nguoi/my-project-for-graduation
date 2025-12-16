"""
Report Generator for Recruitment Emotion Scoring System.

This module generates comprehensive emotion reports with visualizations,
explanations, and evidence extraction for video interview assessments.

Requirements: 14.1, 14.2, 14.3
"""

import json
import cv2
import numpy as np
import os
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualizations will be disabled.")

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not available. PDF export will use text format.")

from .models import (
    EmotionReport,
    CriterionScore,
    KeyMoment,
    FacialData,
)


class ReportGenerator:
    """
    Generates comprehensive emotion reports with visualizations.
    
    This class creates detailed reports from emotion scoring results,
    including:
    - Visual score charts and graphs
    - Key moment identification
    - Detailed explanations for each criterion
    - Evidence extraction (video clips)
    
    Requirements:
    - 14.1: Display all 12 criterion scores with visual indicators
    - 14.2: Provide explanations for each score
    - 14.3: Include timestamps of key emotional moments
    """

    
    def __init__(self, output_dir: str = "reports"):
        """
        Initialize ReportGenerator.
        
        Args:
            output_dir: Directory for saving reports and visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.viz_dir = self.output_dir / "visualizations"
        self.clips_dir = self.output_dir / "video_clips"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"ReportGenerator initialized")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Visualizations: {self.viz_dir}")
        print(f"  Video clips: {self.clips_dir}")
    
    def generate_report(
        self,
        scores: Dict[str, float],
        evidence: Dict[str, Any],
        candidate_id: str,
        video_path: str,
        facial_data_list: Optional[List[FacialData]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> EmotionReport:
        """
        Generate comprehensive emotion report.
        
        Creates a complete EmotionReport with all criterion scores,
        explanations, key moments, and metadata.
        
        Args:
            scores: Dictionary mapping criterion names to score values (0-10)
            evidence: Supporting evidence data for each criterion
            candidate_id: Unique identifier for the candidate
            video_path: Path to the analyzed video file
            facial_data_list: Optional list of FacialData for key moment extraction
            weights: Optional dictionary of criterion weights (defaults to 4-criteria weights)
        
        Returns:
            EmotionReport with complete assessment data
        
        Requirements:
            - 5.5: Display all 4 criterion scores with weights and visual indicators
            - 6.1: Display criterion scores with their respective weights
            - 6.2: Provide explanations for each score
            - 6.3: Include timestamps of key emotional moments
        """
        print(f"\nGenerating comprehensive report for candidate: {candidate_id}")
        
        # Use default 4-criteria weights if not provided AND we have 4 criteria
        if weights is None:
            # Check if we have exactly 4 criteria (new system)
            if len(scores) == 4 and all(c in scores for c in ['confidence', 'positivity', 'professionalism', 'engagement']):
                weights = {
                    'confidence': 0.325,
                    'positivity': 0.275,
                    'professionalism': 0.225,
                    'engagement': 0.175,
                }
            else:
                # For other cases (12-criteria or custom), use equal weights
                weights = {criterion: 1.0 / len(scores) for criterion in scores.keys()}
        
        # Create CriterionScore objects with explanations and weights
        criterion_scores = {}
        for criterion_name, score in scores.items():
            criterion_evidence = evidence.get(criterion_name, {})
            criterion_weight = weights.get(criterion_name, 0.0)
            
            criterion_scores[criterion_name] = CriterionScore(
                criterion_name=criterion_name,
                score=score,
                confidence=self._calculate_confidence(criterion_evidence),
                explanation=self._generate_detailed_explanation(
                    criterion_name, score, criterion_evidence
                ),
                evidence_timestamps=criterion_evidence.get('timestamps', []),
                supporting_data=criterion_evidence,
                weight=criterion_weight
            )
        
        # Calculate total score (weighted average)
        total_score = self._calculate_total_score(scores, weights)
        
        # Extract key moments
        key_moments = []
        if facial_data_list:
            key_moments = self.extract_key_moments(
                facial_data_list=facial_data_list,
                criterion_scores=criterion_scores,
                top_n=20
            )
        
        # Gather metadata
        metadata = {
            'report_generated_at': datetime.now().isoformat(),
            'video_path': video_path,
            'total_criteria': len(scores),
            'average_confidence': np.mean([
                cs.confidence for cs in criterion_scores.values()
            ]),
            'criterion_weights': weights,
        }
        
        if facial_data_list:
            metadata.update({
                'total_frames_analyzed': len(facial_data_list),
                'video_duration_seconds': facial_data_list[-1].timestamp if facial_data_list else 0.0,
            })
        
        # Create EmotionReport
        report = EmotionReport(
            candidate_id=candidate_id,
            video_path=video_path,
            timestamp=datetime.now(),
            criterion_scores=criterion_scores,
            total_score=total_score,
            key_moments=key_moments,
            metadata=metadata
        )
        
        print(f"  Report generated successfully")
        print(f"  Total score: {total_score:.2f}/10")
        print(f"  Key moments identified: {len(key_moments)}")
        
        return report

    
    def create_visualizations(
        self,
        scores: Dict[str, float],
        output_prefix: str = "emotion_report",
        weights: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Create score charts and graphs.
        
        Generates visual representations of the emotion scores including:
        - Bar chart of all criterion scores with weights
        - Radar chart showing score distribution
        - Score distribution histogram
        
        Args:
            scores: Dictionary mapping criterion names to score values (0-10)
            output_prefix: Prefix for output filenames
            weights: Optional dictionary of criterion weights for display
        
        Returns:
            List of paths to generated visualization files
        
        Requirements:
            - 5.5: Display all 4 criterion scores with visual indicators
            - 6.1: Display criterion scores with their respective weights
        """
        if not MATPLOTLIB_AVAILABLE:
            print("Warning: matplotlib not available. Skipping visualizations.")
            return []
        
        print(f"\nCreating visualizations...")
        visualization_paths = []
        
        # Use default 4-criteria weights if not provided
        if weights is None:
            weights = {
                'confidence': 0.325,
                'positivity': 0.275,
                'professionalism': 0.225,
                'engagement': 0.175,
            }
        
        # 1. Bar chart of criterion scores with weights
        bar_chart_path = self._create_bar_chart(scores, output_prefix, weights)
        if bar_chart_path:
            visualization_paths.append(bar_chart_path)
        
        # 2. Radar chart
        radar_chart_path = self._create_radar_chart(scores, output_prefix, weights)
        if radar_chart_path:
            visualization_paths.append(radar_chart_path)
        
        # 3. Score distribution
        distribution_path = self._create_score_distribution(scores, output_prefix)
        if distribution_path:
            visualization_paths.append(distribution_path)
        
        print(f"  Created {len(visualization_paths)} visualizations")
        
        return visualization_paths
    
    def extract_key_moments(
        self,
        facial_data_list: List[FacialData],
        criterion_scores: Dict[str, CriterionScore],
        top_n: int = 20
    ) -> List[KeyMoment]:
        """
        Identify significant timestamps in the interview.
        
        Extracts key moments that provide evidence for scoring decisions,
        including:
        - Peak emotional expressions
        - Significant behavioral changes
        - Notable criterion-specific events
        
        Args:
            facial_data_list: List of FacialData objects from video analysis
            criterion_scores: Dictionary of CriterionScore objects
            top_n: Maximum number of key moments to extract
        
        Returns:
            List of KeyMoment objects sorted by significance
        
        Requirements:
            - 14.3: Include timestamps of key emotional moments
        """
        print(f"\nExtracting key moments from {len(facial_data_list)} frames...")
        
        key_moments = []
        
        # Extract moments for each criterion
        for criterion_name, criterion_score in criterion_scores.items():
            moments = self._extract_criterion_moments(
                criterion_name=criterion_name,
                facial_data_list=facial_data_list,
                criterion_score=criterion_score
            )
            key_moments.extend(moments)
        
        # Sort by intensity and select top N
        key_moments.sort(key=lambda m: m.intensity, reverse=True)
        key_moments = key_moments[:top_n]
        
        # Sort by timestamp for chronological order
        key_moments.sort(key=lambda m: m.timestamp)
        
        print(f"  Extracted {len(key_moments)} key moments")
        
        return key_moments

    
    def export_to_json(
        self,
        report: EmotionReport,
        output_path: Optional[str] = None
    ) -> str:
        """
        Export report to JSON format with proper serialization and validation.
        
        Creates a JSON file containing the complete emotion report with all
        criterion scores, key moments, and metadata. Validates the exported
        file to ensure it's valid JSON.
        
        Args:
            report: EmotionReport to export
            output_path: Optional custom output path
        
        Returns:
            Path to exported JSON file
            
        Raises:
            ValueError: If report validation fails
            IOError: If file cannot be written
            
        Requirements:
            - 14.4: Allow export to JSON format
        """
        print(f"\nExporting report to JSON...")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(
                self.output_dir / f"emotion_report_{report.candidate_id}_{timestamp}.json"
            )
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert report to JSON
            json_data = report.to_json(indent=2)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(json_data)
            
            # Validate exported file
            if not self._validate_json_export(output_path):
                raise ValueError("Exported JSON file validation failed")
            
            print(f"  ✓ JSON report exported successfully: {output_path}")
            print(f"  File size: {os.path.getsize(output_path)} bytes")
            
            return output_path
            
        except Exception as e:
            print(f"  ✗ Error exporting JSON report: {e}")
            raise
    
    def export_to_pdf(
        self,
        report: EmotionReport,
        output_path: Optional[str] = None,
        include_visualizations: bool = True
    ) -> str:
        """
        Export report to PDF format using reportlab.
        
        Creates a professionally formatted PDF document containing the complete
        emotion report with criterion scores, explanations, key moments, and
        optional visualizations. Falls back to text format if reportlab is
        not available.
        
        Args:
            report: EmotionReport to export
            output_path: Optional custom output path
            include_visualizations: Whether to include score visualizations
        
        Returns:
            Path to exported PDF file (or text file if reportlab unavailable)
            
        Raises:
            IOError: If file cannot be written
            
        Requirements:
            - 14.4: Allow export to PDF format
        """
        print(f"\nExporting report to PDF...")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if REPORTLAB_AVAILABLE:
                output_path = str(
                    self.output_dir / f"emotion_report_{report.candidate_id}_{timestamp}.pdf"
                )
            else:
                output_path = str(
                    self.output_dir / f"emotion_report_{report.candidate_id}_{timestamp}.txt"
                )
        
        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if REPORTLAB_AVAILABLE:
                # Generate PDF using reportlab
                self._generate_pdf_report(report, output_path, include_visualizations)
                
                # Validate exported file
                if not self._validate_pdf_export(output_path):
                    raise ValueError("Exported PDF file validation failed")
                
                print(f"  ✓ PDF report exported successfully: {output_path}")
            else:
                # Fallback to text format
                print(f"  ⚠ reportlab not available, exporting as text format")
                text_report = self._generate_text_report(report)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(text_report)
                
                print(f"  ✓ Text report exported: {output_path}")
                print(f"  Note: Install reportlab for PDF export: pip install reportlab")
            
            print(f"  File size: {os.path.getsize(output_path)} bytes")
            
            return output_path
            
        except Exception as e:
            print(f"  ✗ Error exporting PDF report: {e}")
            raise
    
    def extract_video_clips(
        self,
        video_path: str,
        key_moments: List[KeyMoment],
        clip_duration: float = 3.0
    ) -> Dict[str, str]:
        """
        Extract video clips for key moments.
        
        Creates short video clips around each key moment timestamp
        to provide visual evidence for scoring decisions.
        
        Args:
            video_path: Path to source video file
            key_moments: List of KeyMoment objects
            clip_duration: Duration of each clip in seconds
        
        Returns:
            Dictionary mapping KeyMoment timestamps to clip file paths
        """
        print(f"\nExtracting video clips for {len(key_moments)} key moments...")
        
        clip_paths = {}
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  Error: Could not open video file: {video_path}")
            return clip_paths
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        for i, moment in enumerate(key_moments):
            # Calculate start and end frames
            start_time = max(0, moment.timestamp - clip_duration / 2)
            end_time = moment.timestamp + clip_duration / 2
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            
            # Generate clip filename
            clip_filename = f"clip_{i:03d}_{moment.criterion}_{moment.timestamp:.2f}s.mp4"
            clip_path = str(self.clips_dir / clip_filename)
            
            # Extract clip
            success = self._extract_clip(
                cap, start_frame, end_frame, clip_path, fps
            )
            
            if success:
                clip_paths[moment.timestamp] = clip_path
                # Update KeyMoment with clip path
                moment.video_clip_path = clip_path
        
        cap.release()
        
        print(f"  Extracted {len(clip_paths)} video clips")
        
        return clip_paths

    
    # Private helper methods
    
    def _calculate_confidence(self, evidence: Dict[str, Any]) -> float:
        """Calculate confidence score from evidence data."""
        if 'error' in evidence:
            return 0.0
        
        total_frames = evidence.get('total_frames', 0)
        
        if total_frames == 0:
            return 0.5  # Default confidence
        elif total_frames < 30:
            return 0.6
        elif total_frames < 100:
            return 0.8
        else:
            return 0.95
    
    def _calculate_total_score(self, scores: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate weighted average total score.
        
        Args:
            scores: Dictionary of criterion scores
            weights: Optional dictionary of criterion weights
            
        Returns:
            Weighted average score (0-10)
        """
        if len(scores) == 0:
            return 0.0
        
        # Use default 4-criteria weights if not provided AND we have 4 criteria
        if weights is None:
            # Check if we have exactly 4 criteria (new system)
            if len(scores) == 4 and all(c in scores for c in ['confidence', 'positivity', 'professionalism', 'engagement']):
                weights = {
                    'confidence': 0.325,
                    'positivity': 0.275,
                    'professionalism': 0.225,
                    'engagement': 0.175,
                }
            else:
                # For other cases (12-criteria or custom), use equal weights
                weights = {}
        
        # Calculate weighted average
        total = 0.0
        total_weight = 0.0
        
        for criterion_name, score in scores.items():
            weight = weights.get(criterion_name, 0.0)
            total += score * weight
            total_weight += weight
        
        # Normalize if weights don't sum to 1.0 or use simple average if no weights
        if total_weight > 0:
            return float(total / total_weight)
        else:
            # Fallback to simple average (equal weights)
            return float(np.mean(list(scores.values())))
    
    def _generate_detailed_explanation(
        self,
        criterion_name: str,
        score: float,
        evidence: Dict[str, Any]
    ) -> str:
        """
        Generate detailed explanation for a criterion score.
        
        Requirements:
            - 6.2: Provide explanations for each score
        """
        # Determine score level
        if score >= 7.0:
            level = "high"
            performance = "excellent"
        elif score >= 4.0:
            level = "moderate"
            performance = "satisfactory"
        else:
            level = "low"
            performance = "needs improvement"
        
        # Criterion-specific explanations for 4-criteria system
        explanations = {
            'confidence': (
                f"The candidate showed {level} confidence with a score of {score:.1f}/10. "
                f"This {performance} assessment is based on voice tone, posture, eye contact, and facial expressions. "
                f"Eye contact was maintained {evidence.get('eye_contact_percentage', 0):.1f}% of the time, "
                f"and posture indicators showed {evidence.get('posture_score', 0):.2f} alignment. "
                f"{'The candidate demonstrated strong self-assurance and clear communication' if score >= 7 else 'The candidate showed adequate presence with some hesitation' if score >= 4 else 'The candidate appeared uncertain and lacked assertiveness'}."
            ),
            'positivity': (
                f"Positivity was {level} with a score of {score:.1f}/10. "
                f"The candidate's warmth and friendly demeanor were {performance}. "
                f"Genuine smiles were detected in {evidence.get('genuine_smile_count', 0)} frames, "
                f"with {evidence.get('smile_authenticity', 0):.1f}% being authentic. "
                f"{'The candidate created a warm, approachable impression with natural positive expressions' if score >= 7 else 'The candidate showed moderate warmth with some genuine positive moments' if score >= 4 else 'The candidate appeared reserved with limited positive expressions'}."
            ),
            'professionalism': (
                f"Professionalism was {level} with a score of {score:.1f}/10. "
                f"The candidate's professional demeanor and emotional control were {performance}. "
                f"Appropriate emotions were displayed {evidence.get('appropriate_frequency', 0)*100:.1f}% of the time, "
                f"with composure maintained at {evidence.get('composure_score', 0):.2f}. "
                f"{'The candidate exhibited strong professional standards with excellent emotional regulation' if score >= 7 else 'The candidate showed adequate professionalism with generally appropriate behavior' if score >= 4 else 'The candidate displayed unprofessional moments and poor emotional control'}."
            ),
            'engagement': (
                f"Engagement was {level} with a score of {score:.1f}/10. "
                f"The candidate's attention and interaction level were {performance}. "
                f"Eye contact was maintained {evidence.get('eye_contact_percentage', 0):.1f}% of the time, "
                f"with facial responsiveness at {evidence.get('responsiveness_rate', 0)*100:.1f}%. "
                f"Head movements indicated {evidence.get('head_movement_score', 0):.2f} active listening. "
                f"{'The candidate was highly engaged with strong attention and active participation' if score >= 7 else 'The candidate showed moderate engagement with adequate responsiveness' if score >= 4 else 'The candidate appeared disengaged with limited attention and interaction'}."
            ),
            # Legacy 12-criteria support (for backward compatibility)
            'calmness': (
                f"The candidate demonstrated {level} calmness with a score of {score:.1f}/10. "
                f"This {performance} result indicates that the candidate "
                f"{'maintained composure well' if score >= 7 else 'showed some signs of stress' if score >= 4 else 'appeared anxious'} "
                f"throughout the interview. "
                f"Calm expressions were observed in {evidence.get('calm_frequency', 0)*100:.1f}% of analyzed frames."
            ),
            'expression_control': (
                f"Expression control was {level} with a score of {score:.1f}/10. "
                f"The candidate's ability to maintain appropriate facial expressions was {performance}. "
                f"Emotional changes occurred at a rate of {evidence.get('emotion_change_rate', 0)*100:.1f}%, "
                f"{'indicating good emotional regulation' if score >= 7 else 'suggesting moderate control' if score >= 4 else 'showing difficulty in expression management'}."
            ),
            'friendliness': (
                f"Friendliness was {level} with a score of {score:.1f}/10. "
                f"The candidate's warmth and approachability were {performance}. "
                f"Genuine smiles were detected in {evidence.get('genuine_smile_count', 0)} frames, "
                f"{'creating a positive impression' if score >= 7 else 'showing moderate warmth' if score >= 4 else 'appearing somewhat reserved'}."
            ),
            'enthusiasm': (
                f"The candidate displayed {level} enthusiasm with a score of {score:.1f}/10. "
                f"Energy levels and engagement were {performance}. "
                f"Average energy level was {evidence.get('avg_energy', 0):.2f}, "
                f"{'indicating strong interest' if score >= 7 else 'showing moderate engagement' if score >= 4 else 'suggesting low energy'}."
            ),
            'authenticity': (
                f"Authenticity was {level} with a score of {score:.1f}/10. "
                f"The genuineness of emotional expressions was {performance}. "
                f"Micro-expression consistency was {evidence.get('micro_expr_consistency', 0):.2f}, "
                f"{'suggesting genuine emotions' if score >= 7 else 'indicating moderate authenticity' if score >= 4 else 'raising some concerns about sincerity'}."
            ),
            'eye_contact': (
                f"Eye contact was {level} with a score of {score:.1f}/10. "
                f"The candidate's gaze patterns were {performance}. "
                f"Eye contact was maintained {evidence.get('eye_contact_percentage', 0):.1f}% of the time, "
                f"{'demonstrating excellent engagement' if score >= 7 else 'showing adequate attention' if score >= 4 else 'indicating difficulty maintaining eye contact'}."
            ),
            'facial_interaction': (
                f"Facial interaction was {level} with a score of {score:.1f}/10. "
                f"The candidate's facial responsiveness was {performance}. "
                f"Responsiveness rate was {evidence.get('responsiveness_rate', 0)*100:.1f}%, "
                f"{'showing strong engagement' if score >= 7 else 'indicating moderate interaction' if score >= 4 else 'suggesting limited responsiveness'}."
            ),
            'contextual_appropriateness': (
                f"Contextual appropriateness was {level} with a score of {score:.1f}/10. "
                f"The alignment of expressions with conversation context was {performance}. "
                f"Appropriate expressions were shown {evidence.get('appropriate_frequency', 0)*100:.1f}% of the time, "
                f"{'demonstrating excellent emotional intelligence' if score >= 7 else 'showing adequate awareness' if score >= 4 else 'indicating some misalignment'}."
            ),
            'emotion_consistency': (
                f"Emotion consistency was {level} with a score of {score:.1f}/10. "
                f"The stability of emotional states was {performance}. "
                f"Emotional changes occurred at a rate of {evidence.get('change_rate', 0)*100:.1f}%, "
                f"{'indicating strong emotional stability' if score >= 7 else 'showing moderate consistency' if score >= 4 else 'suggesting emotional volatility'}."
            ),
            'sentiment_alignment': (
                f"Sentiment alignment was {level} with a score of {score:.1f}/10. "
                f"The candidate's emotional alignment with the interviewer was {performance}. "
                f"Alignment frequency was {evidence.get('alignment_frequency', 0)*100:.1f}%, "
                f"{'demonstrating excellent empathy' if score >= 7 else 'showing adequate responsiveness' if score >= 4 else 'indicating limited emotional attunement'}."
            ),
        }
        
        return explanations.get(
            criterion_name,
            f"{criterion_name.replace('_', ' ').title()} score: {score:.1f}/10. {performance.title()} performance."
        )

    
    def _extract_criterion_moments(
        self,
        criterion_name: str,
        facial_data_list: List[FacialData],
        criterion_score: CriterionScore
    ) -> List[KeyMoment]:
        """Extract key moments for a specific criterion."""
        moments = []
        
        # 4-criteria system
        if criterion_name == 'confidence':
            # Find moments of strong confidence indicators
            for fd in facial_data_list:
                pitch, yaw, roll = fd.head_pose
                gaze_x, gaze_y = fd.gaze_direction
                gaze_magnitude = np.sqrt(gaze_x**2 + gaze_y**2)
                
                posture_score = 1.0 - (abs(pitch) + abs(yaw)) / 180.0
                gaze_score = 1.0 - min(gaze_magnitude, 1.0)
                confidence_score = (posture_score + gaze_score) / 2.0
                
                if confidence_score > 0.7:
                    moments.append(KeyMoment(
                        timestamp=fd.timestamp,
                        frame_number=fd.frame_number,
                        description="Strong confidence indicators",
                        criterion=criterion_name,
                        emotion='neutral',
                        intensity=confidence_score
                    ))
        
        elif criterion_name == 'positivity':
            # Find genuine smile moments and warm expressions
            for fd in facial_data_list:
                au12 = fd.action_units.get('AU12', 0)  # Lip corner puller
                au6 = fd.action_units.get('AU6', 0)    # Cheek raiser
                smile_score = au12 * au6
                happy_score = fd.emotion_probabilities.get('happy', 0)
                positivity_score = max(smile_score, happy_score)
                
                if positivity_score > 0.5:
                    moments.append(KeyMoment(
                        timestamp=fd.timestamp,
                        frame_number=fd.frame_number,
                        description="Positive expression detected",
                        criterion=criterion_name,
                        emotion='happy',
                        intensity=positivity_score
                    ))
        
        elif criterion_name == 'professionalism':
            # Find moments of professional composure
            for fd in facial_data_list:
                neutral_score = fd.emotion_probabilities.get('neutral', 0)
                happy_score = fd.emotion_probabilities.get('happy', 0)
                # Professional = balanced neutral and appropriate positive
                professional_score = (neutral_score * 0.6 + happy_score * 0.4)
                
                if professional_score > 0.6:
                    moments.append(KeyMoment(
                        timestamp=fd.timestamp,
                        frame_number=fd.frame_number,
                        description="Professional demeanor observed",
                        criterion=criterion_name,
                        emotion='neutral',
                        intensity=professional_score
                    ))
        
        elif criterion_name == 'engagement':
            # Find moments of high engagement
            for fd in facial_data_list:
                gaze_x, gaze_y = fd.gaze_direction
                gaze_magnitude = np.sqrt(gaze_x**2 + gaze_y**2)
                gaze_score = 1.0 - min(gaze_magnitude, 1.0)
                
                # Check for facial responsiveness
                emotion_intensity = max(fd.emotion_probabilities.values())
                engagement_score = (gaze_score * 0.6 + emotion_intensity * 0.4)
                
                if engagement_score > 0.7:
                    moments.append(KeyMoment(
                        timestamp=fd.timestamp,
                        frame_number=fd.frame_number,
                        description="High engagement detected",
                        criterion=criterion_name,
                        emotion='neutral',
                        intensity=engagement_score
                    ))
        
        # Legacy 12-criteria support
        elif criterion_name == 'calmness':
            # Find moments of peak calmness
            for fd in facial_data_list:
                calm_score = fd.emotion_probabilities.get('neutral', 0)
                if calm_score > 0.7:
                    moments.append(KeyMoment(
                        timestamp=fd.timestamp,
                        frame_number=fd.frame_number,
                        description="High calmness detected",
                        criterion=criterion_name,
                        emotion='neutral',
                        intensity=calm_score
                    ))
        
        elif criterion_name == 'friendliness':
            # Find genuine smile moments
            for fd in facial_data_list:
                au12 = fd.action_units.get('AU12', 0)  # Lip corner puller
                au6 = fd.action_units.get('AU6', 0)    # Cheek raiser
                smile_score = au12 * au6
                
                if smile_score > 0.3:
                    moments.append(KeyMoment(
                        timestamp=fd.timestamp,
                        frame_number=fd.frame_number,
                        description="Genuine smile detected",
                        criterion=criterion_name,
                        emotion='happy',
                        intensity=smile_score
                    ))
        
        elif criterion_name == 'enthusiasm':
            # Find moments of high energy
            for fd in facial_data_list:
                happy_score = fd.emotion_probabilities.get('happy', 0)
                surprise_score = fd.emotion_probabilities.get('surprise', 0)
                energy_score = max(happy_score, surprise_score)
                
                if energy_score > 0.6:
                    moments.append(KeyMoment(
                        timestamp=fd.timestamp,
                        frame_number=fd.frame_number,
                        description="High enthusiasm detected",
                        criterion=criterion_name,
                        emotion='happy' if happy_score > surprise_score else 'surprise',
                        intensity=energy_score
                    ))
        
        # Limit to top 3 moments per criterion
        moments.sort(key=lambda m: m.intensity, reverse=True)
        return moments[:3]

    
    def _create_bar_chart(
        self,
        scores: Dict[str, float],
        output_prefix: str,
        weights: Optional[Dict[str, float]] = None
    ) -> Optional[str]:
        """Create bar chart of criterion scores with weights."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Prepare data
            criteria = list(scores.keys())
            values = list(scores.values())
            
            # Create color map based on score levels
            colors = []
            for score in values:
                if score >= 7.0:
                    colors.append('#2ecc71')  # Green for high
                elif score >= 4.0:
                    colors.append('#f39c12')  # Orange for moderate
                else:
                    colors.append('#e74c3c')  # Red for low
            
            # Create bar chart
            bars = ax.bar(range(len(criteria)), values, color=colors, alpha=0.8)
            
            # Customize chart
            ax.set_xlabel('Emotion Criteria', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score (0-10)', fontsize=12, fontweight='bold')
            ax.set_title('Emotion Scoring Results by Criterion', fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(criteria)))
            
            # Add weights to labels if provided
            if weights:
                labels = []
                for c in criteria:
                    weight = weights.get(c, 0.0)
                    label = f"{c.replace('_', ' ').title()}\n({weight*100:.1f}%)"
                    labels.append(label)
                ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            else:
                ax.set_xticklabels([c.replace('_', ' ').title() for c in criteria], rotation=45, ha='right')
            
            ax.set_ylim(0, 10)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add score labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.1f}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            # Add legend
            high_patch = mpatches.Patch(color='#2ecc71', label='High (7-10)')
            med_patch = mpatches.Patch(color='#f39c12', label='Moderate (4-7)')
            low_patch = mpatches.Patch(color='#e74c3c', label='Low (0-4)')
            ax.legend(handles=[high_patch, med_patch, low_patch], loc='upper right')
            
            plt.tight_layout()
            
            # Save figure
            output_path = str(self.viz_dir / f"{output_prefix}_bar_chart.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    Created bar chart: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"    Error creating bar chart: {e}")
            return None
    
    def _create_radar_chart(
        self,
        scores: Dict[str, float],
        output_prefix: str,
        weights: Optional[Dict[str, float]] = None
    ) -> Optional[str]:
        """Create radar chart of criterion scores with weights."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # Prepare data
            criteria = list(scores.keys())
            values = list(scores.values())
            
            # Number of variables
            num_vars = len(criteria)
            
            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            
            # Complete the circle
            values += values[:1]
            angles += angles[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
            ax.fill(angles, values, alpha=0.25, color='#3498db')
            
            # Fix axis to go in the right order
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            
            # Draw axis lines for each angle and label
            ax.set_xticks(angles[:-1])
            
            # Add weights to labels if provided
            if weights:
                labels = []
                for c in criteria:
                    weight = weights.get(c, 0.0)
                    label = f"{c.replace('_', ' ').title()}\n({weight*100:.1f}%)"
                    labels.append(label)
                ax.set_xticklabels(labels, size=9)
            else:
                ax.set_xticklabels([c.replace('_', ' ').title() for c in criteria], size=9)
            
            # Set y-axis limits
            ax.set_ylim(0, 10)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.set_yticklabels(['2', '4', '6', '8', '10'], size=8)
            
            # Add title
            ax.set_title('Emotion Scoring Radar Chart', size=14, fontweight='bold', pad=20)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # Save figure
            output_path = str(self.viz_dir / f"{output_prefix}_radar_chart.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    Created radar chart: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"    Error creating radar chart: {e}")
            return None

    
    def _create_score_distribution(
        self,
        scores: Dict[str, float],
        output_prefix: str
    ) -> Optional[str]:
        """Create score distribution visualization."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            values = list(scores.values())
            
            # 1. Histogram
            ax1.hist(values, bins=10, range=(0, 10), color='#3498db', alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Score Range', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax1.set_title('Score Distribution', fontsize=13, fontweight='bold')
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add vertical lines for thresholds
            ax1.axvline(x=4, color='orange', linestyle='--', linewidth=2, label='Low/Moderate')
            ax1.axvline(x=7, color='green', linestyle='--', linewidth=2, label='Moderate/High')
            ax1.legend()
            
            # 2. Box plot
            ax2.boxplot(values, patch_artist=True,
                       boxprops=dict(facecolor='#3498db', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       whiskerprops=dict(color='black', linewidth=1.5),
                       capprops=dict(color='black', linewidth=1.5))
            ax2.set_ylabel('Score (0-10)', fontsize=12, fontweight='bold')
            ax2.set_title('Score Statistics', fontsize=13, fontweight='bold')
            ax2.set_ylim(0, 10)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add statistics text
            mean_score = np.mean(values)
            median_score = np.median(values)
            std_score = np.std(values)
            
            stats_text = f'Mean: {mean_score:.2f}\nMedian: {median_score:.2f}\nStd Dev: {std_score:.2f}'
            ax2.text(1.15, 5, stats_text, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            
            # Save figure
            output_path = str(self.viz_dir / f"{output_prefix}_distribution.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"    Created distribution chart: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"    Error creating distribution chart: {e}")
            return None
    
    def _extract_clip(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        output_path: str,
        fps: float
    ) -> bool:
        """Extract a video clip from source video."""
        try:
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Set to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Extract frames
            for frame_num in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            out.release()
            return True
            
        except Exception as e:
            print(f"    Error extracting clip: {e}")
            return False
    
    def _generate_text_report(self, report: EmotionReport) -> str:
        """Generate text-based report (fallback for PDF)."""
        lines = []
        lines.append("=" * 80)
        lines.append("EMOTION SCORING REPORT")
        lines.append("=" * 80)
        lines.append(f"\nCandidate ID: {report.candidate_id}")
        lines.append(f"Video: {report.video_path}")
        lines.append(f"Report Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"\nTotal Score: {report.total_score:.2f}/10")
        lines.append("\n" + "-" * 80)
        lines.append("CRITERION SCORES")
        lines.append("-" * 80)
        
        for criterion_name, criterion_score in report.criterion_scores.items():
            lines.append(f"\n{criterion_name.replace('_', ' ').title()}")
            lines.append(f"  Score: {criterion_score.score:.2f}/10")
            lines.append(f"  Confidence: {criterion_score.confidence:.2f}")
            lines.append(f"  Explanation: {criterion_score.explanation}")
        
        lines.append("\n" + "-" * 80)
        lines.append("KEY MOMENTS")
        lines.append("-" * 80)
        
        for i, moment in enumerate(report.key_moments, 1):
            lines.append(f"\n{i}. {moment.description}")
            lines.append(f"   Time: {moment.timestamp:.2f}s")
            lines.append(f"   Criterion: {moment.criterion}")
            lines.append(f"   Emotion: {moment.emotion}")
            lines.append(f"   Intensity: {moment.intensity:.2f}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)
    
    def _validate_json_export(self, file_path: str) -> bool:
        """
        Validate exported JSON file.
        
        Checks that the file exists, is readable, contains valid JSON,
        and can be deserialized back to an EmotionReport.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                print(f"    Validation failed: File does not exist")
                return False
            
            # Check file is not empty
            if os.path.getsize(file_path) == 0:
                print(f"    Validation failed: File is empty")
                return False
            
            # Read and parse JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                json_content = f.read()
                data = json.loads(json_content)
            
            # Verify required fields exist
            required_fields = ['candidate_id', 'video_path', 'timestamp', 
                             'criterion_scores', 'total_score']
            for field in required_fields:
                if field not in data:
                    print(f"    Validation failed: Missing required field '{field}'")
                    return False
            
            # Try to deserialize back to EmotionReport
            EmotionReport.from_dict(data)
            
            print(f"    ✓ JSON validation passed")
            return True
            
        except json.JSONDecodeError as e:
            print(f"    Validation failed: Invalid JSON - {e}")
            return False
        except Exception as e:
            print(f"    Validation failed: {e}")
            return False
    
    def _validate_pdf_export(self, file_path: str) -> bool:
        """
        Validate exported PDF file.
        
        Checks that the file exists, is readable, and has the PDF signature.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            # Check file exists
            if not os.path.exists(file_path):
                print(f"    Validation failed: File does not exist")
                return False
            
            # Check file is not empty
            if os.path.getsize(file_path) == 0:
                print(f"    Validation failed: File is empty")
                return False
            
            # Check PDF signature (first 4 bytes should be %PDF)
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    print(f"    Validation failed: Not a valid PDF file")
                    return False
            
            print(f"    ✓ PDF validation passed")
            return True
            
        except Exception as e:
            print(f"    Validation failed: {e}")
            return False
    
    def _generate_pdf_report(
        self,
        report: EmotionReport,
        output_path: str,
        include_visualizations: bool
    ) -> None:
        """
        Generate PDF report using reportlab.
        
        Creates a professionally formatted PDF with all report data.
        
        Args:
            report: EmotionReport to export
            output_path: Path for output PDF file
            include_visualizations: Whether to include score charts
        """
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        # Container for PDF elements
        story = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER,
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=12,
        )
        normal_style = styles['Normal']
        
        # Title
        story.append(Paragraph("Emotion Scoring Report", title_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Candidate Information
        story.append(Paragraph("Candidate Information", heading_style))
        info_data = [
            ['Candidate ID:', report.candidate_id],
            ['Video Path:', report.video_path],
            ['Report Generated:', report.timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            ['Total Score:', f"{report.total_score:.2f}/10"],
        ]
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
            ('ALIGN', (1, 0), (1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(info_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Criterion Scores
        story.append(Paragraph("Criterion Scores", heading_style))
        
        # Create table data with weights
        score_data = [['Criterion', 'Weight', 'Score', 'Confidence', 'Level']]
        for criterion_name, criterion_score in report.criterion_scores.items():
            # Determine level
            if criterion_score.score >= 7.0:
                level = 'High'
            elif criterion_score.score >= 4.0:
                level = 'Moderate'
            else:
                level = 'Low'
            
            score_data.append([
                criterion_name.replace('_', ' ').title(),
                f"{criterion_score.weight*100:.1f}%",
                f"{criterion_score.score:.2f}",
                f"{criterion_score.confidence:.2f}",
                level
            ])
        
        # Create table
        score_table = Table(score_data, colWidths=[2*inch, 0.8*inch, 0.8*inch, 1*inch, 0.8*inch])
        score_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Detailed Explanations
        story.append(PageBreak())
        story.append(Paragraph("Detailed Criterion Explanations", heading_style))
        story.append(Spacer(1, 0.1 * inch))
        
        for criterion_name, criterion_score in report.criterion_scores.items():
            criterion_title = criterion_name.replace('_', ' ').title()
            story.append(Paragraph(
                f"<b>{criterion_title}</b> - Score: {criterion_score.score:.2f}/10",
                normal_style
            ))
            story.append(Spacer(1, 0.05 * inch))
            story.append(Paragraph(criterion_score.explanation, normal_style))
            story.append(Spacer(1, 0.15 * inch))
        
        # Key Moments
        if report.key_moments:
            story.append(PageBreak())
            story.append(Paragraph("Key Moments", heading_style))
            story.append(Spacer(1, 0.1 * inch))
            
            moment_data = [['Time', 'Description', 'Criterion', 'Emotion', 'Intensity']]
            for moment in report.key_moments[:20]:  # Limit to 20 moments
                moment_data.append([
                    f"{moment.timestamp:.2f}s",
                    moment.description,
                    moment.criterion.replace('_', ' ').title(),
                    moment.emotion.title(),
                    f"{moment.intensity:.2f}"
                ])
            
            moment_table = Table(moment_data, colWidths=[0.8*inch, 2*inch, 1.5*inch, 1*inch, 0.8*inch])
            moment_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ecc71')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8f9fa')]),
            ]))
            story.append(moment_table)
        
        # Include visualizations if requested and available
        if include_visualizations and MATPLOTLIB_AVAILABLE:
            story.append(PageBreak())
            story.append(Paragraph("Score Visualizations", heading_style))
            story.append(Spacer(1, 0.2 * inch))
            
            # Generate visualizations
            scores_dict = {name: score.score for name, score in report.criterion_scores.items()}
            viz_paths = self.create_visualizations(scores_dict, f"temp_{report.candidate_id}")
            
            # Add visualizations to PDF
            for viz_path in viz_paths:
                if os.path.exists(viz_path):
                    try:
                        img = Image(viz_path, width=6*inch, height=3*inch)
                        story.append(img)
                        story.append(Spacer(1, 0.2 * inch))
                    except Exception as e:
                        print(f"    Warning: Could not add visualization {viz_path}: {e}")
        
        # Build PDF
        doc.build(story)
        print(f"    ✓ PDF document generated")
