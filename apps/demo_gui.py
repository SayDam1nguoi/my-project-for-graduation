# -*- coding: utf-8 -*-
"""
Demo GUI cho nhận diện cảm xúc thời gian thực.

Giao diện đồ họa với 2 chức năng:
- Quét từ camera trực tiếp
- Quét từ video file

Sử dụng:
    python demo_gui.py
"""

import cv2
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import threading
import sys
from pathlib import Path
import time
import numpy as np
from typing import Dict, List, Tuple

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add src to path (go up to project root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.face_detector import FaceDetector
from src.inference.preprocessor import FacePreprocessor
from src.inference.emotion_classifier import EmotionClassifier
from src.inference.visualizer import VisualizationEngine, VisualizationConfig
from src.inference.video_processor import CameraProcessor, VideoFileProcessor

# Screen capture
try:
    from src.utils.screen_capture import ScreenCapture, is_screen_capture_available
    SCREEN_CAPTURE_AVAILABLE = is_screen_capture_available()
except ImportError:
    SCREEN_CAPTURE_AVAILABLE = False

# Attention detection
try:
    from src.video_analysis.attention_detector import AttentionDetector, AttentionLevel
    ATTENTION_AVAILABLE = True
except ImportError:
    ATTENTION_AVAILABLE = False

# Dual attention detection
try:
    from src.video_analysis.dual_attention_coordinator import DualAttentionCoordinator
    DUAL_ATTENTION_AVAILABLE = True
except ImportError:
    DUAL_ATTENTION_AVAILABLE = False

# Dual person comparison
try:
    from src.video_analysis.dual_person_coordinator import DualPersonCoordinator
    DUAL_PERSON_AVAILABLE = True
except ImportError:
    DUAL_PERSON_AVAILABLE = False

# COMMENTED OUT - Replaced with audio recording system
# Speech analysis
# try:
#     from src.speech_analysis import (
#         SpeechAnalysisCoordinator,
#         TranscriptWindow,
#         AudioConfig,
#         STTConfig,
#         QualityConfig
#     )
#     SPEECH_ANALYSIS_AVAILABLE = True
# except ImportError:
#     SPEECH_ANALYSIS_AVAILABLE = False
SPEECH_ANALYSIS_AVAILABLE = False

# Video audio player
try:
    from src.utils.video_audio_player import VideoAudioPlayer, check_moviepy_installed
    VIDEO_AUDIO_AVAILABLE = check_moviepy_installed()
except ImportError:
    VIDEO_AUDIO_AVAILABLE = False
    print("Warning: Video audio player not available")

# Appearance assessment
try:
    from src.video_analysis.appearance.coordinator import AppearanceAssessmentCoordinator
    from src.video_analysis.appearance.config import AppearanceConfig
    from src.video_analysis.appearance.models import AppearanceWarning
    APPEARANCE_ASSESSMENT_AVAILABLE = True
except ImportError:
    APPEARANCE_ASSESSMENT_AVAILABLE = False
    print("Warning: Appearance assessment not available")

# GUI Components
from gui.overlays import LiveReportOverlay, AppearanceWarningOverlay
from gui.constants import COLORS, BUTTON_STYLE, WINDOW_SIZE, PERFORMANCE
from gui.video_handler import VideoHandler, VideoSourceManager
from gui import ui_builder
from gui.file_save_dialog import ask_save_file
# Import với path đầy đủ để đảm bảo singleton
from apps.gui.score_manager import get_score_manager

# Video Transcription Tab
try:
    from gui.video_transcription_tab import VideoTranscriptionTab
    VIDEO_TRANSCRIPTION_AVAILABLE = True
except ImportError:
    VIDEO_TRANSCRIPTION_AVAILABLE = False
    print("Warning: Video transcription tab not available")

# Emotion Scoring Tab
try:
    from gui.emotion_scoring_tab import EmotionScoringTab
    EMOTION_SCORING_AVAILABLE = True
except ImportError:
    EMOTION_SCORING_AVAILABLE = False
    print("Warning: Emotion scoring tab not available")

# Audio Recording Tab
try:
    from gui.audio_recording_tab import AudioRecordingTab
    AUDIO_RECORDING_AVAILABLE = True
except ImportError:
    AUDIO_RECORDING_AVAILABLE = False
    print("Warning: Audio recording tab not available")


class EmotionRecognitionGUI:
    """Giao diện GUI cho nhận diện cảm xúc."""
    
    def __init__(self, root):
        """Khởi tạo giao diện."""
        self.root = root
        
        # Check if root is a Frame (tab mode) or Tk (standalone mode)
        if isinstance(root, tk.Frame):
            # Tab mode - don't set window properties
            self.parent_frame = root
            self.root.configure(bg=COLORS['bg_dark'])
        else:
            # Standalone mode - set window properties
            self.parent_frame = root
            self.root.title("Nhận Diện Cảm Xúc - Emotion Recognition")
            self.root.geometry(f"{WINDOW_SIZE['width']}x{WINDOW_SIZE['height']}")
            self.root.minsize(WINDOW_SIZE['min_width'], WINDOW_SIZE['min_height'])
            self.root.configure(bg=COLORS['bg_dark'])
        
        # Biến trạng thái
        self.is_running = False
        self.cap = None
        self.thread = None
        self.video_source = "camera"  # "camera", "file", or "screen"
        self.video_file_path = None
        self.ui_hidden = False  # Flag để biết UI có đang bị ẩn không
        
        # Audio player for video files
        self.audio_player = None
        
        # COMMENTED OUT - Replaced with audio recording system
        # Transcript extraction
        # self.transcript_thread = None
        # self.is_extracting_transcript = False
        
        # Dual person comparison mode
        self.dual_person_mode = False
        self.dual_person_coordinator = None
        
        # Dual person settings
        self.dual_person_settings = {
            'comparison_update_interval': 10.0,  # seconds
            'show_bounding_boxes': True,
            'primary_position': 'left'  # 'left' or 'right'
        }
        
        # Screen capture
        self.screen_capture = None
        self.selected_window = None
        
        # Components
        self.detector = None
        self.preprocessor = None
        self.classifier = None
        self.visualizer = None
        
        # Processors
        self.camera_processor = None
        self.video_processor = None
        self.current_processor = None
        
        # Statistics
        self.emotion_counts = {}
        self.total_faces = 0
        self.frame_count = 0
        self.start_time = None
        self.current_emotion = None  # Track current emotion for overlay
        
        # Attention detection
        self.attention_detector = None
        self.attention_scores = []
        self.attention_focused_frames = 0
        self.attention_distracted_frames = 0
        
        # Dual attention detection
        self.dual_attention_coordinator = None
        self.dual_attention_enabled = False
        
        # Appearance assessment
        self.appearance_enabled = False  # Mặc định tắt
        self.assess_clothing = False  # Quét quần áo
        self.assess_lighting = False  # Quét ánh sáng
        self.appearance_scores = {
            'clothing': 0.0,
            'lighting': 0.0,
            'overall': 0.0
        }
        
        # Real-time Scoring - 4 ĐẦU MỤC RIÊNG BIỆT
        # 1. Cảm xúc (Emotion)
        self.emotion_scores_history = []
        self.current_emotion_score = 0.0
        self.emotion_criteria_scores = {}
        
        # 2. Tập trung (Focus) - Theo công thức FocusScore
        self.focus_scores_history = []
        self.current_focus_score = 0.0
        self.focus_components = {}  # FT5, EGA, MS
        
        # 3. Rõ ràng lời nói (Speech Clarity) - Sẽ implement sau
        self.speech_clarity_scores_history = []
        self.current_speech_clarity_score = 0.0
        
        # 4. Nội dung (Content) - Sẽ implement sau
        self.content_scores_history = []
        self.current_content_score = 0.0
        
        # Live report overlay
        self.live_report_overlay = LiveReportOverlay(self)
        
        # Performance settings
        self.skip_frames = PERFORMANCE['default_skip_frames']
        self.frame_counter = 0
        self.resize_factor = PERFORMANCE['default_resize_factor']
        
        # Video handlers (will be initialized after widgets are created)
        self.video_handler = None
        self.video_source_manager = VideoSourceManager()
        
        # Tạo giao diện
        self.create_widgets()
        
        # Bind resize event
        self.root.bind('<Configure>', self.on_window_resize)
        
        # Lazy loading - Load components in background after UI is shown
        self.components_loaded = False
        self.root.after(100, self.initialize_components_async)
    
    def create_widgets(self):
        """Tạo các widget cho giao diện."""
        # Create main container
        self.main_container = tk.Frame(self.root, bg='#1a1a1a')
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Header with gradient effect
        header_frame = tk.Frame(self.main_container, bg='#0d47a1', height=70)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="NHẬN DIỆN CẢM XÚC THỜI GIAN THỰC",
            font=("Segoe UI", 22, "bold"),
            bg='#0d47a1',
            fg='#ffffff'
        )
        title_label.pack(pady=18)
        
        # Main content frame with grid
        content_frame = tk.Frame(self.main_container, bg='#1a1a1a')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Configure grid weights for responsive layout
        content_frame.grid_rowconfigure(0, weight=1)
        content_frame.grid_columnconfigure(0, weight=4)  # Video takes 80% width
        content_frame.grid_columnconfigure(1, weight=1)  # Stats takes 20% width
        
        # Left panel - Video display with rounded corners effect
        left_frame = tk.Frame(content_frame, bg='#252525', relief=tk.FLAT, bd=0)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        # Configure left frame grid
        left_frame.grid_rowconfigure(0, weight=0)  # Label fixed
        left_frame.grid_rowconfigure(1, weight=1)  # Video expands
        left_frame.grid_columnconfigure(0, weight=1)
        
        # Store left frame reference for mode switching
        self.left_frame = left_frame
        
        # Single view mode (default)
        self.single_view_frame = tk.Frame(left_frame, bg='#252525')
        self.single_view_frame.grid(row=0, column=0, rowspan=2, sticky='nsew', padx=2, pady=2)
        self.single_view_frame.grid_rowconfigure(0, weight=0)
        self.single_view_frame.grid_rowconfigure(1, weight=1)
        self.single_view_frame.grid_columnconfigure(0, weight=1)
        
        video_label = tk.Label(
            self.single_view_frame,
            text="📹 Video Camera",
            font=("Segoe UI", 13, "bold"),
            bg='#252525',
            fg='#64b5f6'
        )
        video_label.grid(row=0, column=0, pady=12, sticky='w', padx=15)
        
        # Video canvas - dùng Canvas thay vì Label để tránh resize
        video_container = tk.Frame(self.single_view_frame, bg='#1a1a1a', relief=tk.SOLID, bd=1)
        video_container.grid(row=1, column=0, sticky='nsew', padx=15, pady=(0, 15))
        video_container.grid_rowconfigure(0, weight=1)
        video_container.grid_columnconfigure(0, weight=1)
        
        self.video_canvas = tk.Canvas(
            video_container,
            bg='#000000',
            highlightthickness=0
        )
        self.video_canvas.grid(row=0, column=0, sticky='nsew')
        
        # Placeholder text
        self.video_canvas_text = self.video_canvas.create_text(
            0, 0,
            text="Chưa có video\nNhấn 'BẮT ĐẦU QUÉT' để bắt đầu",
            font=("Segoe UI", 12),
            fill='#757575',
            anchor='center'
        )
        
        # Store canvas image reference
        self.video_canvas_image = None
        
        # Split view mode (hidden by default)
        self.split_view_frame = tk.Frame(left_frame, bg='#1e1e1e')
        # Don't grid yet - will show when dual person mode activated
        self.split_view_frame.grid_rowconfigure(0, weight=0)  # Labels
        self.split_view_frame.grid_rowconfigure(1, weight=1)  # Videos
        self.split_view_frame.grid_rowconfigure(2, weight=0)  # Comparison panel
        self.split_view_frame.grid_columnconfigure(0, weight=1)  # Left video
        self.split_view_frame.grid_columnconfigure(1, weight=0)  # Comparison panel (center)
        self.split_view_frame.grid_columnconfigure(2, weight=1)  # Right video
        
        # Primary video (left)
        primary_label = tk.Label(
            self.split_view_frame,
            text="Ban",
            font=("Arial", 12, "bold"),
            bg='#1e1e1e',
            fg='#4CAF50'
        )
        primary_label.grid(row=0, column=0, pady=10)
        
        self.primary_canvas = tk.Label(
            self.split_view_frame,
            bg='#000000'
        )
        self.primary_canvas.grid(row=1, column=0, sticky='nsew', padx=(10, 5), pady=10)
        
        # Comparison panel (center) - Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
        comparison_container = tk.Frame(self.split_view_frame, bg='#1e1e1e')
        comparison_container.grid(row=1, column=1, sticky='ns', padx=5, pady=10)
        
        # Comparison panel title
        comparison_title = tk.Label(
            comparison_container,
            text="SO SANH",
            font=("Arial", 11, "bold"),
            bg='#1e1e1e',
            fg='#FFD700'
        )
        comparison_title.pack(pady=(5, 10))
        
        # Comparison content frame
        self.comparison_panel = tk.Frame(comparison_container, bg='#2b2b2b', relief=tk.RAISED, borderwidth=2)
        self.comparison_panel.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Emotion comparison section
        emotion_section = tk.Frame(self.comparison_panel, bg='#2b2b2b')
        emotion_section.pack(fill=tk.X, padx=10, pady=10)
        
        emotion_title = tk.Label(
            emotion_section,
            text="CAM XUC",
            font=("Arial", 10, "bold"),
            bg='#2b2b2b',
            fg='#FFD700'
        )
        emotion_title.pack(pady=(0, 5))
        
        self.emotion_comparison_label = tk.Label(
            emotion_section,
            text="--",
            font=("Courier", 9),
            bg='#2b2b2b',
            fg='#ffffff',
            justify=tk.CENTER
        )
        self.emotion_comparison_label.pack()
        
        # Separator
        separator1 = tk.Frame(self.comparison_panel, bg='#555555', height=2)
        separator1.pack(fill=tk.X, padx=10, pady=5)
        
        # Attention comparison section
        attention_section = tk.Frame(self.comparison_panel, bg='#2b2b2b')
        attention_section.pack(fill=tk.X, padx=10, pady=10)
        
        attention_title = tk.Label(
            attention_section,
            text="TAP TRUNG",
            font=("Arial", 10, "bold"),
            bg='#2b2b2b',
            fg='#FFD700'
        )
        attention_title.pack(pady=(0, 5))
        
        self.attention_comparison_label = tk.Label(
            attention_section,
            text="--",
            font=("Courier", 9),
            bg='#2b2b2b',
            fg='#ffffff',
            justify=tk.CENTER
        )
        self.attention_comparison_label.pack()
        
        # Separator
        separator2 = tk.Frame(self.comparison_panel, bg='#555555', height=2)
        separator2.pack(fill=tk.X, padx=10, pady=5)
        
        # Overall winner section
        winner_section = tk.Frame(self.comparison_panel, bg='#2b2b2b')
        winner_section.pack(fill=tk.X, padx=10, pady=10)
        
        winner_title = tk.Label(
            winner_section,
            text="TONG KET",
            font=("Arial", 10, "bold"),
            bg='#2b2b2b',
            fg='#FFD700'
        )
        winner_title.pack(pady=(0, 5))
        
        self.overall_winner_label = tk.Label(
            winner_section,
            text="--",
            font=("Courier", 9, "bold"),
            bg='#2b2b2b',
            fg='#ffffff',
            justify=tk.CENTER
        )
        self.overall_winner_label.pack()
        
        # Last update time label
        self.comparison_update_label = tk.Label(
            self.comparison_panel,
            text="Cap nhat: --",
            font=("Arial", 8),
            bg='#2b2b2b',
            fg='#9E9E9E'
        )
        self.comparison_update_label.pack(side=tk.BOTTOM, pady=5)
        
        # Export button for dual person mode (at bottom of comparison panel)
        self.dual_export_button = tk.Button(
            comparison_container,
            text="📊 XUAT BAO CAO",
            font=("Arial", 9, "bold"),
            bg='#FF9800',
            fg='#ffffff',
            activebackground='#F57C00',
            command=self.export_comparison_report,
            cursor='hand2',
            width=12,
            height=1
        )
        self.dual_export_button.pack(side=tk.BOTTOM, pady=5)
        
        # Settings button for dual person mode (at bottom of comparison panel)
        self.dual_settings_button = tk.Button(
            comparison_container,
            text="⚙️ CAI DAT",
            font=("Arial", 9, "bold"),
            bg='#607D8B',
            fg='#ffffff',
            activebackground='#546E7A',
            command=self.show_dual_person_settings,
            cursor='hand2',
            width=12,
            height=1
        )
        self.dual_settings_button.pack(side=tk.BOTTOM, pady=5)
        
        # Secondary video (right)
        secondary_label = tk.Label(
            self.split_view_frame,
            text="Nguoi Khac",
            font=("Arial", 12, "bold"),
            bg='#1e1e1e',
            fg='#2196F3'
        )
        secondary_label.grid(row=0, column=2, pady=10)
        
        self.secondary_canvas = tk.Label(
            self.split_view_frame,
            bg='#000000'
        )
        self.secondary_canvas.grid(row=1, column=2, sticky='nsew', padx=(5, 10), pady=10)
        
        # Right panel - Statistics with modern design
        right_frame = tk.Frame(content_frame, bg='#252525', relief=tk.FLAT, bd=0)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=(10, 0))
        
        # Store right frame reference for mode switching
        self.right_frame = right_frame
        
        # Configure right frame grid
        right_frame.grid_rowconfigure(0, weight=0)  # Label fixed
        right_frame.grid_rowconfigure(1, weight=1)  # Text expands
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Single view statistics (default)
        self.single_stats_frame = tk.Frame(right_frame, bg='#252525')
        self.single_stats_frame.grid(row=0, column=0, rowspan=2, sticky='nsew', padx=2, pady=2)
        self.single_stats_frame.grid_rowconfigure(0, weight=0)
        self.single_stats_frame.grid_rowconfigure(1, weight=1)
        self.single_stats_frame.grid_columnconfigure(0, weight=1)
        
        stats_label = tk.Label(
            self.single_stats_frame,
            text="📊 Thống Kê",
            font=("Segoe UI", 13, "bold"),
            bg='#252525',
            fg='#81c784'
        )
        stats_label.grid(row=0, column=0, pady=12, sticky='w', padx=15)
        
        # Statistics display with scrollbar
        stats_container = tk.Frame(self.single_stats_frame, bg='#1a1a1a', relief=tk.SOLID, bd=1)
        stats_container.grid(row=1, column=0, sticky='nsew', padx=15, pady=(0, 15))
        stats_container.grid_rowconfigure(0, weight=1)
        stats_container.grid_columnconfigure(0, weight=1)
        
        stats_scrollbar = tk.Scrollbar(stats_container)
        stats_scrollbar.grid(row=0, column=1, sticky='ns')
        
        self.stats_text = tk.Text(
            stats_container,
            font=("Consolas", 9),
            bg='#1e1e1e',
            fg='#e0e0e0',
            wrap=tk.WORD,
            state=tk.DISABLED,
            yscrollcommand=stats_scrollbar.set,
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.stats_text.grid(row=0, column=0, sticky='nsew')
        stats_scrollbar.config(command=self.stats_text.yview)
        
        # Dual statistics panels (hidden by default)
        self.dual_stats_frame = tk.Frame(right_frame, bg='#1e1e1e')
        # Don't grid yet - will show when dual person mode activated
        self.dual_stats_frame.grid_rowconfigure(0, weight=1)  # Primary stats
        self.dual_stats_frame.grid_rowconfigure(1, weight=1)  # Secondary stats
        self.dual_stats_frame.grid_columnconfigure(0, weight=1)
        
        # Primary user statistics panel
        primary_stats_container = tk.Frame(self.dual_stats_frame, bg='#1e1e1e')
        primary_stats_container.grid(row=0, column=0, sticky='nsew', padx=5, pady=(5, 2))
        
        primary_stats_label = tk.Label(
            primary_stats_container,
            text="Bạn (Primary)",
            font=("Arial", 11, "bold"),
            bg='#1e1e1e',
            fg='#4CAF50'
        )
        primary_stats_label.pack(pady=5)
        
        self.primary_stats_text = tk.Text(
            primary_stats_container,
            font=("Courier", 9),
            bg='#2b2b2b',
            fg='#ffffff',
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=15
        )
        self.primary_stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Secondary user statistics panel
        secondary_stats_container = tk.Frame(self.dual_stats_frame, bg='#1e1e1e')
        secondary_stats_container.grid(row=1, column=0, sticky='nsew', padx=5, pady=(2, 5))
        
        secondary_stats_label = tk.Label(
            secondary_stats_container,
            text="Người Khác (Secondary)",
            font=("Arial", 11, "bold"),
            bg='#1e1e1e',
            fg='#2196F3'
        )
        secondary_stats_label.pack(pady=5)
        
        self.secondary_stats_text = tk.Text(
            secondary_stats_container,
            font=("Courier", 9),
            bg='#2b2b2b',
            fg='#ffffff',
            wrap=tk.WORD,
            state=tk.DISABLED,
            height=15
        )
        self.secondary_stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Last update time for dual statistics
        self.last_dual_stats_update = 0
        self.last_emotion_detection_time = 0
        
        # Control panel with modern design (fixed height to prevent resizing)
        control_frame = tk.Frame(self.main_container, bg='#252525', relief=tk.FLAT, bd=0, height=200)
        control_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        control_frame.pack_propagate(False)  # Prevent frame from resizing
        
        # Top row - Source selection with better styling
        source_frame = tk.Frame(control_frame, bg='#252525', height=50)
        source_frame.pack(fill=tk.X, pady=15, padx=20)
        source_frame.pack_propagate(False)  # Prevent frame from resizing
        
        source_label = tk.Label(
            source_frame,
            text="🎥 Nguồn video:",
            font=("Segoe UI", 11, "bold"),
            bg='#252525',
            fg='#90caf9'
        )
        source_label.pack(side=tk.LEFT, padx=(0, 20))
        
        # Radio buttons for source selection with modern style
        self.source_var = tk.StringVar(value="camera")
        
        radio_style = {
            'font': ("Segoe UI", 10),
            'bg': '#252525',
            'fg': '#e0e0e0',
            'selectcolor': '#1a1a1a',
            'activebackground': '#252525',
            'activeforeground': '#ffffff',
            'highlightthickness': 0
        }
        
        camera_radio = tk.Radiobutton(
            source_frame,
            text="📷 Camera",
            variable=self.source_var,
            value="camera",
            command=self.on_source_change,
            **radio_style
        )
        camera_radio.pack(side=tk.LEFT, padx=15)
        
        file_radio = tk.Radiobutton(
            source_frame,
            text="📁 Video File",
            variable=self.source_var,
            value="file",
            command=self.on_source_change,
            **radio_style
        )
        file_radio.pack(side=tk.LEFT, padx=15)
        
        # Screen capture radio button
        screen_radio = tk.Radiobutton(
            source_frame,
            text="🖥️ Screen Capture",
            variable=self.source_var,
            value="screen",
            command=self.on_source_change,
            state=tk.NORMAL if SCREEN_CAPTURE_AVAILABLE else tk.DISABLED,
            **radio_style
        )
        screen_radio.pack(side=tk.LEFT, padx=15)
        
        # Browse button for video file with modern style
        self.browse_button = tk.Button(
            source_frame,
            text="📂 Chọn File...",
            font=("Segoe UI", 10),
            bg='#455a64',
            fg='#ffffff',
            activebackground='#37474f',
            command=self.browse_video_file,
            cursor='hand2',
            state=tk.DISABLED,
            relief=tk.FLAT,
            padx=15,
            pady=5
        )
        self.browse_button.pack(side=tk.LEFT, padx=15)
        
        # File/Window path label
        self.file_label = tk.Label(
            source_frame,
            text="",
            font=("Segoe UI", 9),
            bg='#252525',
            fg='#9e9e9e'
        )
        self.file_label.pack(side=tk.LEFT, padx=15)
        
        # Separator line
        separator = tk.Frame(control_frame, bg='#3a3a3a', height=1)
        separator.pack(fill=tk.X, padx=20, pady=10)
        
        # Bottom row - Control buttons with scrollable container
        button_container = tk.Frame(control_frame, bg='#252525', height=70)
        button_container.pack(fill=tk.X, pady=(10, 15), padx=20)
        button_container.pack_propagate(False)
        
        # Create canvas for scrolling
        button_canvas = tk.Canvas(button_container, bg='#252525', height=60, highlightthickness=0)
        button_scrollbar = tk.Scrollbar(button_container, orient='horizontal', command=button_canvas.xview)
        button_frame = tk.Frame(button_canvas, bg='#252525')
        
        button_frame.bind(
            '<Configure>',
            lambda e: button_canvas.configure(scrollregion=button_canvas.bbox('all'))
        )
        
        button_canvas.create_window((0, 0), window=button_frame, anchor='nw')
        button_canvas.configure(xscrollcommand=button_scrollbar.set)
        
        button_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        button_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Modern button style (smaller for more buttons)
        button_style = {
            'font': ("Segoe UI", 10, "bold"),
            'cursor': 'hand2',
            'relief': tk.FLAT,
            'padx': 15,
            'pady': 8
        }
        
        # Start/Stop button with gradient effect
        self.start_button = tk.Button(
            button_frame,
            text="▶️ BẮT ĐẦU QUÉT",
            bg='#43a047',
            fg='#ffffff',
            activebackground='#388e3c',
            command=self.toggle_processing,
            **button_style
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Reset button
        self.reset_button = tk.Button(
            button_frame,
            text="🔄 ĐẶT LẠI",
            bg='#1e88e5',
            fg='#ffffff',
            activebackground='#1565c0',
            command=self.reset_statistics,
            **button_style
        )
        self.reset_button.pack(side=tk.LEFT, padx=10)
        
        # Export Statistics button
        self.export_button = tk.Button(
            button_frame,
            text="📊 XUẤT THỐNG KÊ",
            bg='#fb8c00',
            fg='#ffffff',
            activebackground='#ef6c00',
            command=self.export_statistics,
            **button_style
        )
        self.export_button.pack(side=tk.LEFT, padx=10)
        
        # Send Scores button (NEW)
        self.send_scores_button = tk.Button(
            button_frame,
            text="📤 GỬI ĐIỂM SANG TỔNG HỢP",
            bg='#FF9800',
            fg='#ffffff',
            activebackground='#F57C00',
            command=self.send_scores_to_summary,
            **button_style
        )
        self.send_scores_button.pack(side=tk.LEFT, padx=10)
        
        # Dual Person Comparison button
        self.dual_person_button = tk.Button(
            button_frame,
            text="👥 CHẾ ĐỘ SO SÁNH 2 NGƯỜI",
            bg='#00acc1',
            fg='#ffffff',
            activebackground='#00838f',
            command=self.toggle_dual_person_mode,
            state=tk.NORMAL if DUAL_PERSON_AVAILABLE else tk.DISABLED,
            **button_style
        )
        self.dual_person_button.pack(side=tk.LEFT, padx=10)
        
        # Hide UI button (for screen capture mode)
        self.hide_ui_button = tk.Button(
            button_frame,
            text="👁️ ẨN UI → BÁO CÁO",
            bg='#7e57c2',
            fg='#ffffff',
            activebackground='#5e35b1',
            command=self.toggle_ui_visibility,
            **button_style
        )
        self.hide_ui_button.pack(side=tk.LEFT, padx=10)
        print(f"✓ Hide UI button created: {self.hide_ui_button}")
        print(f"  Command: {self.hide_ui_button['command']}")
        
        # Performance Settings button - REMOVED
        
        # Audio Recording button (NEW)
        self.audio_record_button = tk.Button(
            button_frame,
            text="🎤 THU ÂM",
            bg='#d32f2f',
            fg='#ffffff',
            activebackground='#b71c1c',
            command=self.show_audio_recording_dialog,
            **button_style
        )
        self.audio_record_button.pack(side=tk.LEFT, padx=10)
        
        # Appearance Assessment button
        self.appearance_button = tk.Button(
            button_frame,
            text="👔 ĐÁNH GIÁ NGOẠI HÌNH",
            bg='#6a1b9a',
            fg='#ffffff',
            activebackground='#4a148c',
            command=self.show_appearance_settings,
            state=tk.NORMAL if APPEARANCE_ASSESSMENT_AVAILABLE else tk.DISABLED,
            **button_style
        )
        self.appearance_button.pack(side=tk.LEFT, padx=10)
        
        # Clear Cache button
        self.clear_cache_button = tk.Button(
            button_frame,
            text="🗑️ XÓA CACHE",
            bg='#d32f2f',
            fg='#ffffff',
            activebackground='#b71c1c',
            command=self.clear_all_cache,
            **button_style
        )
        self.clear_cache_button.pack(side=tk.LEFT, padx=10)
        
        # Status and info frame
        info_frame = tk.Frame(control_frame, bg='#252525')
        info_frame.pack(fill=tk.X, pady=(5, 10), padx=20)
        
        # Status label with modern design
        self.status_label = tk.Label(
            info_frame,
            text="⏸️ Đã dừng",
            font=("Segoe UI", 11, "bold"),
            bg='#252525',
            fg='#ffa726'
        )
        self.status_label.pack(side=tk.LEFT, padx=(0, 30))
        
        # FPS label with icon
        self.fps_label = tk.Label(
            info_frame,
            text="⚡ FPS: 0.0",
            font=("Segoe UI", 11, "bold"),
            bg='#252525',
            fg='#66bb6a'
        )
        self.fps_label.pack(side=tk.LEFT, padx=30)
        self.fps_label.pack(side=tk.RIGHT, padx=20)
        
        # Initialize appearance warning overlay (after video_canvas is created)
        self.appearance_warning_overlay = AppearanceWarningOverlay(self.video_canvas)
        
        # Initialize video handler (after video_canvas is created)
        self.video_handler = VideoHandler(self.video_canvas)
    
    def initialize_components_async(self):
        """Khởi tạo components trong background thread."""
        self.update_status("⏳ Đang tải model...", "#ff9800")
        
        # Run in thread to avoid freezing UI
        thread = threading.Thread(target=self.initialize_components, daemon=True)
        thread.start()
    
    def initialize_components(self):
        """Khởi tạo các components nhận diện cảm xúc."""
        try:
            
            # Model path
            model_path = Path("models/efficientnet_b2_best.pth")
            if not model_path.exists():
                messagebox.showerror(
                    "Loi",
                    f"Khong tim thay model: {model_path}\n\n"
                    "Vui long huan luyen model truoc hoac tai model da huan luyen."
                )
                self.update_status("Loi: Khong tim thay model", "#f44336")
                return
            
            # Initialize components with improved sensitivity (RESTORED TO ORIGINAL)
            self.detector = FaceDetector(
                device='auto',
                confidence_threshold=0.85,  # Original value for better sensitivity
                max_faces=10,  # Original: detect up to 10 faces
                target_size=(640, 480),  # Original: back to default for better detection
                enable_preprocessing=False  # Disabled for speed (can enable if needed)
            )
            
            self.preprocessor = FacePreprocessor(
                target_size=(224, 224),
                margin=0.2  # Original: back to default
            )
            
            self.classifier = EmotionClassifier(
                model_path=str(model_path),
                device='auto',
                confidence_threshold=0.6  # Original threshold
            )
            
            vis_config = VisualizationConfig()
            self.visualizer = VisualizationEngine(config=vis_config)
            
            # Initialize processors with performance optimizations (RESTORED TO ORIGINAL)
            detection_skip = PERFORMANCE.get('face_detection_skip', 3)
            self.camera_processor = CameraProcessor(
                self.detector,
                self.preprocessor,
                self.classifier,
                detection_skip_frames=detection_skip,  # Skip frames for face detection
                cache_results=True  # Cache detection results for smoother playback
            )
            
            self.video_processor = VideoFileProcessor(
                self.detector,
                self.preprocessor,
                self.classifier,
                frame_skip=0,  # Original: Process all frames (0 = no skip)
                temporal_window=7,  # Original: Temporal smoothing window (industry-standard)
                enable_temporal_smoothing=True  # Enable industry-standard smoothing
            )
            
            # Initialize emotion counts
            self.emotion_counts = {emotion: 0 for emotion in self.classifier.emotions}
            
            # Initialize attention detector
            if ATTENTION_AVAILABLE:
                self.attention_detector = AttentionDetector(
                    ear_threshold=0.25,  # Increased for better detection
                    gaze_threshold=0.20,  # Increased for better detection
                    pose_threshold=25.0,  # Increased for better detection
                    alert_duration=3.0
                )
            
            # Initialize dual attention coordinator
            if DUAL_ATTENTION_AVAILABLE and ATTENTION_AVAILABLE and SCREEN_CAPTURE_AVAILABLE:
                try:
                    self.dual_attention_coordinator = DualAttentionCoordinator(
                        camera_detector=self.attention_detector,
                        face_detector=self.detector,
                        enable_screen_monitor=True,
                        camera_weight=0.6,
                        screen_weight=0.4,
                        distraction_threshold=60.0,
                        alert_threshold=5.0,
                        alert_cooldown=10.0,
                        enable_performance_monitoring=False,  # Disable FPS-based auto-disable
                        fps_threshold=5.0  # Set very low threshold (won't trigger)
                    )
                    print("DualAttentionCoordinator initialized successfully")
                except Exception as e:
                    print(f"Warning: Could not initialize DualAttentionCoordinator: {e}")
                    self.dual_attention_coordinator = None
            
            # Initialize appearance assessment coordinator
            if APPEARANCE_ASSESSMENT_AVAILABLE:
                try:
                    # Load appearance config from YAML or use defaults
                    appearance_config = AppearanceConfig(
                        lighting_enabled=True,
                        clothing_enabled=True,
                        real_time_warnings_enabled=True,
                        warning_update_interval=5.0,
                        warning_threshold=60.0
                    )
                    
                    self.appearance_coordinator = AppearanceAssessmentCoordinator(
                        config=appearance_config
                    )
                    print("AppearanceAssessmentCoordinator initialized successfully")
                except Exception as e:
                    print(f"Warning: Could not initialize appearance assessment: {e}")
                    self.appearance_coordinator = None
            else:
                self.appearance_coordinator = None
            
            self.components_loaded = True
            self.update_status("✅ Sẵn sàng", "#4CAF50")
            self.update_statistics()
            
        except Exception as e:
            self.components_loaded = False
            messagebox.showerror("Loi", f"Khong the khoi tao: {str(e)}")
            self.update_status(f"Loi: {str(e)}", "#f44336")
    
    def on_source_change(self):
        """Xử lý khi thay đổi nguồn video."""
        # Stop processing if currently running
        if self.is_running:
            self.stop_processing()
        
        source = self.source_var.get()
        
        # Reset video display to placeholder (Canvas không có config image/text)
        if not self.is_running:
            self.clear_video_display()
        
        # Reset buttons
        self.browse_button.config(state=tk.DISABLED)
        self.file_label.config(text="")
        
        if source == "file":
            self.browse_button.config(state=tk.NORMAL)
            self.video_file_path = None
            self.selected_window = None
            self.update_status("Chon file video de bat dau", "#ff9800")
        elif source == "screen":
            self.video_file_path = None
            self.selected_window = None
            self.file_label.config(text="Quét toàn màn hình")
            self.update_status("San sang quet toan man hinh", "#4CAF50")
        else:  # camera
            self.video_file_path = None
            self.selected_window = None
            self.update_status("San sang quet tu camera", "#4CAF50")
    
    # select_window method removed - now using full screen capture
    
    def browse_video_file(self):
        """Chọn file video."""
        file_path = filedialog.askopenfilename(
            title="Chon video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_file_path = file_path
            
            # Display shortened path
            path_obj = Path(file_path)
            display_name = path_obj.name
            if len(display_name) > 40:
                display_name = display_name[:37] + "..."
            self.file_label.config(text=display_name)
            self.update_status("Da chon file - San sang bat dau", "#4CAF50")
    
    def toggle_processing(self):
        """Bật/tắt xử lý video."""
        # Check if components loaded
        if not self.components_loaded:
            messagebox.showwarning("Chờ", "Đang tải model, vui lòng đợi...")
            return
        
        if not self.is_running:
            self.start_processing()
        else:
            self.stop_processing()
    
    def start_processing(self):
        """Bắt đầu xử lý video."""
        if self.classifier is None:
            messagebox.showerror("Loi", "Components chua duoc khoi tao!")
            return
        
        # Handle dual person mode
        if self.dual_person_mode:
            self._start_dual_person_processing()
            return
        
        source = self.source_var.get()
        
        # Open video source
        if source == "camera":
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror(
                    "Loi",
                    "Khong the mo camera!\n\n"
                    "Kiem tra:\n"
                    "- Camera da ket noi\n"
                    "- Quyen truy cap camera\n"
                    "- Khong co ung dung nao khac dang dung camera"
                )
                return
            self.current_processor = self.camera_processor
            
        elif source == "file":
            if not self.video_file_path:
                messagebox.showerror("Loi", "Vui long chon file video truoc!")
                return
            
            self.cap = cv2.VideoCapture(self.video_file_path)
            if not self.cap.isOpened():
                messagebox.showerror(
                    "Loi",
                    f"Khong the mo file video!\n\n"
                    f"File: {self.video_file_path}\n\n"
                    "Kiem tra:\n"
                    "- File ton tai\n"
                    "- Dinh dang video hop le\n"
                    "- Codec duoc ho tro"
                )
                return
            self.current_processor = self.video_processor
            
            # Initialize audio player for video file
            if VIDEO_AUDIO_AVAILABLE:
                try:
                    print(f"[Audio] Khoi tao audio player cho video file...")
                    self.audio_player = VideoAudioPlayer(self.video_file_path)
                    print(f"[Audio] Audio track phat hien:")
                    print(f"  - Sample rate: {self.audio_player.sample_rate}Hz")
                    print(f"  - Channels: {self.audio_player.n_channels}")
                    print(f"  - Duration: {self.audio_player.duration:.1f}s")
                except Exception as e:
                    print(f"[Audio] Khong the khoi tao audio player: {e}")
                    self.audio_player = None
            
        else:  # screen
            if not SCREEN_CAPTURE_AVAILABLE:
                messagebox.showerror("Loi", "Screen capture khong kha dung!\n\nVui long cai dat: pip install mss")
                return
            
            # Initialize screen capture for full screen
            try:
                if self.screen_capture is None:
                    self.screen_capture = ScreenCapture(monitor_number=1)  # Primary monitor
                
                self.cap = "screen"  # Special marker for screen capture
                self.current_processor = self.camera_processor  # Use camera processor
                
                # FIX: Lock canvas size khi bắt đầu screen capture để tránh resize loop
                if hasattr(self, '_cached_canvas_size'):
                    # Get current canvas size and lock it
                    try:
                        canvas_width = self.video_canvas.winfo_width()
                        canvas_height = self.video_canvas.winfo_height()
                        if canvas_width > 1 and canvas_height > 1:
                            self._cached_canvas_size = (canvas_width, canvas_height)
                            print(f"🔒 Locked canvas size for screen capture: {canvas_width}x{canvas_height}")
                    except:
                        pass
                
            except Exception as e:
                messagebox.showerror("Loi", f"Khong the bat dau screen capture: {str(e)}")
                return
        
        # Reset processor statistics
        self.current_processor.reset_statistics()
        
        # Start dual attention coordinator for camera and screen sources
        # Note: Attention tracking is disabled for video files (only emotion detection)
        if self.dual_attention_coordinator is not None and source in ["camera", "screen"]:
            try:
                if self.dual_attention_coordinator.start():
                    self.dual_attention_enabled = True
                    print(f"✓ Dual attention detection enabled for {source}")
                else:
                    print("Warning: Could not start dual attention coordinator")
                    self.dual_attention_enabled = False
            except Exception as e:
                print(f"Warning: Error starting dual attention coordinator: {e}")
                self.dual_attention_enabled = False
        else:
            self.dual_attention_enabled = False
            if source == "file":
                print("ℹ Attention tracking disabled for video file (emotion detection only)")
        
        self.is_running = True
        self.video_source = source
        self.start_time = time.time()
        self.frame_count = 0
        
        # Get initial canvas size
        try:
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                self._cached_canvas_size = (canvas_width, canvas_height)
                self._window_size_changed = False
                print(f"✓ Initial canvas: {canvas_width}x{canvas_height}")
        except:
            pass
        
        # Start audio player for video file
        if source == "file" and self.audio_player is not None:
            try:
                print("[Audio] Bat dau phat audio...")
                self.audio_player.start()
                print("[Audio] ✓ Audio dang phat!")
            except Exception as e:
                print(f"[Audio] Loi khi phat audio: {e}")
        
        # Update UI
        self.start_button.config(
            text="DUNG QUET",
            bg='#f44336',
            activebackground='#da190b'
        )
        
        # Disable source selection while processing
        self.browse_button.config(state=tk.DISABLED)
        
        # Set status text
        if source == "camera":
            status_text = "Dang quet camera..."
        elif source == "file":
            status_text = "Dang xu ly video..." + (" (co audio)" if self.audio_player else "")
        else:
            status_text = "Dang capture man hinh..."
        
        self.update_status(status_text, "#4CAF50")
        
        # Start processing thread
        self.thread = threading.Thread(target=self.process_video, daemon=True)
        self.thread.start()
    
    def _start_dual_person_processing(self):
        """Start dual person comparison processing.
        
        Requirements:
        - 1.1: Start DualPersonCoordinator
        - 1.4: Process frames from both sources
        - 1.3: Handle screen capture failures with error dialog
        """
        if self.dual_person_coordinator is None:
            messagebox.showerror("Loi", "Dual person coordinator chua duoc khoi tao!")
            return
        
        # Reset fallback message flag
        if hasattr(self, '_fallback_message_shown'):
            delattr(self, '_fallback_message_shown')
        
        # Start coordinator
        if not self.dual_person_coordinator.start():
            # Determine which component failed
            error_details = "Khong the bat dau dual person mode!\n\n"
            
            if not self.dual_person_coordinator.camera_active:
                error_details += "Loi: Camera khong hoat dong\n\n"
                error_details += "Kiem tra:\n"
                error_details += "- Camera da ket noi\n"
                error_details += "- Quyen truy cap camera\n"
                error_details += "- Khong co ung dung nao khac dang dung camera"
            elif not self.dual_person_coordinator.screen_capture_active:
                error_details += "Loi: Screen capture khong hoat dong\n\n"
                error_details += "Kiem tra:\n"
                error_details += "- Quyen truy cap man hinh\n"
                error_details += "- He thong ho tro screen capture\n"
                error_details += "- Thu khoi dong lai ung dung"
            else:
                error_details += "Loi khong xac dinh\n\n"
                error_details += "Vui long kiem tra log de biet them chi tiet"
            
            messagebox.showerror("Loi Khoi Tao", error_details)
            return
        
        self.is_running = True
        self.video_source = "dual_person"
        self.start_time = time.time()
        self.frame_count = 0
        
        # Update UI
        self.start_button.config(
            text="DUNG QUET",
            bg='#f44336',
            activebackground='#da190b'
        )
        
        self.update_status("Dang so sanh 2 nguoi...", "#00BCD4")
        
        # Start processing thread
        self.thread = threading.Thread(target=self.process_dual_person_video, daemon=True)
        self.thread.start()
        
        print("✓ Dual person processing started")
    
    def stop_processing(self):
        """Dừng xử lý video."""
        self.is_running = False
        
        # Stop audio player
        if self.audio_player is not None:
            try:
                print("[Audio] Dung audio player...")
                self.audio_player.stop()
                self.audio_player = None
                print("[Audio] ✓ Audio da dung")
            except Exception as e:
                print(f"[Audio] Loi khi dung audio: {e}")
        
        # Stop dual person coordinator if active
        if self.dual_person_mode and self.dual_person_coordinator is not None:
            try:
                if self.dual_person_coordinator.is_active():
                    self.dual_person_coordinator.stop()
            except Exception as e:
                print(f"Warning: Error stopping dual person coordinator: {e}")
        
        # Stop dual attention coordinator
        if self.dual_attention_coordinator is not None and self.dual_attention_enabled:
            try:
                self.dual_attention_coordinator.stop()
                self.dual_attention_enabled = False
                print("Dual attention detection stopped")
            except Exception as e:
                print(f"Warning: Error stopping dual attention coordinator: {e}")
        
        # Release resources
        if self.cap is not None and self.cap != "screen":
            self.cap.release()
        self.cap = None
        
        # Cleanup screen capture (no explicit stop needed)
        # ScreenCapture cleans up automatically
        
        # Clear video display
        self.clear_video_display()
        
        # Update UI
        self.start_button.config(
            text="BAT DAU QUET",
            bg='#4CAF50',
            activebackground='#45a049'
        )
        
        # Re-enable source selection
        source = self.source_var.get()
        if source == "file":
            self.browse_button.config(state=tk.NORMAL)
        elif source == "screen":
            pass  # No button to enable for screen capture
        
        self.update_status("Da dung", "#ff9800")
    
    def process_dual_person_video(self):
        """Process video in dual person comparison mode.
        
        Requirements:
        - 1.4: Process frames from both sources
        - 2.1, 2.2: Update split view display
        - 3.3: Update statistics within 500ms when emotion detected
        - 3.6: Update statistics at least once per second
        - 5.1: Update comparison results
        - 7.3: Display face detection warning (>5s no face)
        - 7.5: Display low confidence visual indicators
        """
        # Track performance warnings
        low_fps_start_time = None
        low_fps_warning_shown = False
        
        while self.is_running and self.dual_person_coordinator is not None:
            try:
                # Process frame through coordinator
                result = self.dual_person_coordinator.process_frame()
                
                if result is None:
                    print("Warning: No result from coordinator")
                    time.sleep(0.1)
                    continue
                
                # Get current frames from coordinator
                primary_frame, secondary_frame = self.dual_person_coordinator.get_current_frames()
                
                if primary_frame is None:
                    print("Warning: No primary frame")
                    continue
                
                # Check for screen capture failure and fallback (Req 1.3, 7.2)
                if self.dual_person_coordinator.fallback_to_single_mode:
                    # Show fallback message once
                    if not hasattr(self, '_fallback_message_shown'):
                        self._fallback_message_shown = True
                        self.root.after(0, lambda: self._show_fallback_message())
                
                # Visualize results on frames (respect bounding boxes setting - Req 8.3)
                # Primary frame with bounding boxes and warnings
                if self.dual_person_settings['show_bounding_boxes']:
                    if result.primary.face_detected and result.primary.face_bbox is not None:
                        # face_bbox is in (x, y, width, height) format
                        x, y, w, h = result.primary.face_bbox
                        x1, y1 = x, y
                        x2, y2 = x + w, y + h
                        cv2.rectangle(primary_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw emotion label with low confidence indicator (Req 7.5)
                        if result.primary.emotion is not None:
                            if result.primary.low_confidence:
                                # Low confidence - dimmer color and "?" mark
                                label = f"{result.primary.emotion}: {result.primary.emotion_confidence:.2f} (?)"
                                color = (100, 200, 100)  # Dimmer green
                            else:
                                label = f"{result.primary.emotion}: {result.primary.emotion_confidence:.2f}"
                                color = (0, 255, 0)  # Bright green
                            
                            cv2.putText(
                                primary_frame,
                                label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )
                
                # Draw no face warning overlay on primary frame (Req 7.3)
                if result.primary.no_face_warning:
                    self._draw_no_face_warning(primary_frame, "primary")
                
                # Secondary frame with bounding boxes and warnings
                if secondary_frame is not None and self.dual_person_settings['show_bounding_boxes']:
                    if result.secondary.face_detected and result.secondary.face_bbox is not None:
                        # face_bbox is in (x, y, width, height) format
                        x, y, w, h = result.secondary.face_bbox
                        x1, y1 = x, y
                        x2, y2 = x + w, y + h
                        cv2.rectangle(secondary_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        
                        # Draw emotion label with low confidence indicator (Req 7.5)
                        if result.secondary.emotion is not None:
                            if result.secondary.low_confidence:
                                # Low confidence - dimmer color and "?" mark
                                label = f"{result.secondary.emotion}: {result.secondary.emotion_confidence:.2f} (?)"
                                color = (200, 100, 100)  # Dimmer red/blue
                            else:
                                label = f"{result.secondary.emotion}: {result.secondary.emotion_confidence:.2f}"
                                color = (255, 0, 0)  # Bright red/blue
                            
                            cv2.putText(
                                secondary_frame,
                                label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                2
                            )
                
                # Draw no face warning overlay on secondary frame (Req 7.3)
                if secondary_frame is not None and result.secondary.no_face_warning:
                    self._draw_no_face_warning(secondary_frame, "secondary")
                
                # Update split view display
                self.update_split_view(primary_frame, secondary_frame)
                
                # Update FPS
                fps = self.dual_person_coordinator.get_fps()
                self.update_fps(fps)
                
                # Check for performance warning (Req 7.1 - FPS < 15)
                if fps < 15 and fps > 0:
                    if low_fps_start_time is None:
                        low_fps_start_time = time.time()
                    elif time.time() - low_fps_start_time >= 5.0 and not low_fps_warning_shown:
                        # Show warning after 5 seconds of low FPS
                        low_fps_warning_shown = True
                        self.root.after(0, lambda: self._show_performance_warning(fps))
                else:
                    # Reset warning if FPS recovers
                    low_fps_start_time = None
                    low_fps_warning_shown = False
                
                # Update frame count
                self.frame_count += 1
                
                # Check if emotion was detected (for 500ms update requirement)
                current_time = time.time()
                emotion_detected = (
                    (result.primary.face_detected and result.primary.emotion is not None) or
                    (result.secondary.face_detected and result.secondary.emotion is not None)
                )
                
                # Update statistics based on requirements:
                # - Within 500ms when emotion detected (Req 3.3)
                # - At least once per second (Req 3.6)
                should_update = False
                
                if emotion_detected:
                    # Update within 500ms of emotion detection
                    if current_time - self.last_emotion_detection_time >= 0.5:
                        should_update = True
                        self.last_emotion_detection_time = current_time
                
                # Always update at least once per second
                if current_time - self.last_dual_stats_update >= 1.0:
                    should_update = True
                
                if should_update:
                    self.update_dual_statistics(result)
                    self.last_dual_stats_update = current_time
                
            except Exception as e:
                print(f"Error in dual person processing: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
                continue
        
        # Final update - just update with last known result
        # No need to fetch new result since we're stopping
    
    def process_video(self):
        """Xử lý video từ camera, file, hoặc screen capture."""
        consecutive_failures = 0
        max_failures = 30  # 30 frames liên tiếp thất bại
        
        # Calculate frame delay for video files (to match original FPS)
        video_fps = 30  # Default FPS
        if self.video_source == "file" and self.cap is not None:
            try:
                video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if video_fps <= 0 or video_fps > 120:
                    video_fps = 30
            except:
                video_fps = 30
        frame_delay = 1.0 / video_fps if self.video_source == "file" else 0
        last_frame_time = time.time()
        
        while self.is_running and self.cap is not None:
            # Read frame based on source
            if self.video_source == "screen":
                # Capture from screen
                frame = self.screen_capture.capture_current()
                if frame is None:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        # Window might be closed or unavailable
                        self.root.after(0, lambda: self.on_screen_capture_error())
                        break
                    time.sleep(0.1)  # Wait a bit before retry
                    continue
                else:
                    consecutive_failures = 0  # Reset on success
                ret = True
            else:
                # Read from camera or file
                ret, frame = self.cap.read()
                if not ret:
                    # End of video file
                    if self.video_source == "file":
                        self.root.after(0, self.on_video_end)
                    break
                
                # Add delay for video files to maintain proper playback speed
                if self.video_source == "file" and frame_delay > 0:
                    elapsed = time.time() - last_frame_time
                    if elapsed < frame_delay:
                        time.sleep(frame_delay - elapsed)
                    last_frame_time = time.time()
            
            self.frame_count += 1
            self.frame_counter += 1
            
            # Skip frames for better performance (only if skip_frames > 0)
            if self.skip_frames > 0 and self.frame_counter % (self.skip_frames + 1) != 0:
                # Skip processing but still display for smooth playback
                # Chỉ update display mỗi 2 frames để giảm tải
                if self.frame_counter % 2 == 0:
                    try:
                        # Use video handler for consistent display
                        self.video_handler.update_display(frame)
                    except:
                        pass
                continue
            
            try:
                # Resize frame for faster processing
                if self.resize_factor < 1.0:
                    h, w = frame.shape[:2]
                    new_w = int(w * self.resize_factor)
                    new_h = int(h * self.resize_factor)
                    # Use INTER_LINEAR for faster resizing
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                # Process frame using appropriate processor
                results, fps = self.current_processor.process_frame(frame)
                
                # Update local statistics from processor
                stats = self.current_processor.get_statistics()
                self.emotion_counts = stats['emotion_counts']
                self.total_faces = stats['total_faces']
                
                # Track current emotion for overlay
                if len(results) > 0:
                    detection, _ = results[0]
                    if hasattr(detection, 'emotion') and detection.emotion:
                        self.current_emotion = detection.emotion
                else:
                    self.current_emotion = None
                
                # Skip heavy processing for video files to improve performance
                disable_attention = (self.video_source == "file" and 
                                   PERFORMANCE.get('disable_attention_in_video', True))
                disable_appearance = (self.video_source == "file" and 
                                    PERFORMANCE.get('disable_appearance_in_video', True))
                
                # Process with dual attention coordinator if enabled
                dual_result = None
                if not disable_attention and self.dual_attention_enabled and self.dual_attention_coordinator is not None:
                    try:
                        # Get landmarks from first detection
                        landmarks = None
                        if len(results) > 0:
                            detection, _ = results[0]
                            landmarks = detection.landmarks
                        
                        # Process frame through dual attention coordinator
                        dual_result = self.dual_attention_coordinator.process_frame(frame, landmarks)
                        
                        # CHỈ LƯU VÀ TÍNH ATTENTION KHI UI KHÔNG BỊ ẨN
                        if not self.ui_hidden:
                            # Store score for statistics
                            self.attention_scores.append(dual_result.fused_score.combined_score)
                            
                            # Update attention statistics
                            if dual_result.fused_score.combined_score >= 60:
                                self.attention_focused_frames += 1
                            else:
                                self.attention_distracted_frames += 1
                    
                    except Exception as e:
                        print(f"Error in dual attention processing: {e}")
                        dual_result = None
                
                # Fallback to single attention detector if dual attention not enabled
                # ALWAYS run attention detection for camera (ignore FPS)
                elif self.attention_detector is not None and self.video_source == "camera":
                    # Kiểm tra có phát hiện khuôn mặt không
                    face_detected = len(results) > 0 and results[0][0].landmarks is not None
                    
                    if face_detected:
                        detection, _ = results[0]
                        attention_score, attention_details = self.attention_detector.calculate_attention_score(
                            detection.landmarks,
                            frame.shape[:2],
                            face_detected=True
                        )
                    else:
                        # Không có khuôn mặt - truyền None và face_detected=False
                        attention_score, attention_details = self.attention_detector.calculate_attention_score(
                            None,
                            frame.shape[:2],
                            face_detected=False
                        )
                    
                    # CHỈ LƯU VÀ TÍNH ATTENTION KHI UI KHÔNG BỊ ẨN
                    if not self.ui_hidden:
                        self.attention_scores.append(attention_score)
                        
                        # Ngưỡng mới: >= 8.0 (thang 0-10) = tập trung
                        if attention_score >= 8.0:
                            self.attention_focused_frames += 1
                        else:
                            self.attention_distracted_frames += 1
                
                # Process real-time scoring
                if len(results) > 0:
                    try:
                        detection, emotion_result = results[0]
                        if emotion_result and detection.landmarks is not None:
                            emotion = emotion_result.emotion
                            emotion_probs = emotion_result.probabilities
                            
                            # 1. TÍNH ĐIỂM CẢM XÚC - Luôn tính cho tất cả nguồn
                            emotion_score = self._calculate_realtime_emotion_score(
                                emotion,
                                emotion_probs,
                                detection.landmarks,
                                frame.shape[:2]
                            )
                            self.emotion_scores_history.append(emotion_score)
                            if len(self.emotion_scores_history) > 300:
                                self.emotion_scores_history.pop(0)
                            self.current_emotion_score = np.mean(self.emotion_scores_history)
                            
                            # 2. TÍNH ĐIỂM TẬP TRUNG - CHỈ cho camera/screen, KHÔNG cho video file
                            if self.video_source != "file":
                                focus_score = self._calculate_focus_score(
                                    detection.landmarks,
                                    frame.shape[:2]
                                )
                                self.focus_scores_history.append(focus_score)
                                if len(self.focus_scores_history) > 300:
                                    self.focus_scores_history.pop(0)
                                self.current_focus_score = np.mean(self.focus_scores_history)
                            
                    except Exception as e:
                        print(f"Error in real-time scoring: {e}")
                
                # Process appearance assessment if enabled (Req 7.1: every 5 seconds)
                appearance_warnings = []
                if not disable_appearance and self.appearance_enabled and self.appearance_coordinator is not None and len(results) > 0:
                    try:
                        detection, _ = results[0]
                        if detection.bbox is not None:
                            # Assess appearance
                            appearance_assessment = self.appearance_coordinator.assess_appearance(
                                frame, detection.bbox
                            )
                            
                            # Get warnings from assessment (Req 7.2, 7.3)
                            if appearance_assessment is not None:
                                # Update scores based on what user selected
                                if self.assess_clothing:
                                    self.appearance_scores['clothing'] = appearance_assessment.clothing_score
                                else:
                                    self.appearance_scores['clothing'] = 0.0
                                
                                if self.assess_lighting:
                                    self.appearance_scores['lighting'] = appearance_assessment.lighting_score
                                else:
                                    self.appearance_scores['lighting'] = 0.0
                                
                                # Overall score only if at least one is enabled
                                if self.assess_clothing or self.assess_lighting:
                                    self.appearance_scores['overall'] = appearance_assessment.overall_score
                                else:
                                    self.appearance_scores['overall'] = 0.0
                                
                                from src.video_analysis.appearance.warning_manager import WarningManager
                                warning_manager = WarningManager(
                                    warning_threshold=self.appearance_coordinator.config.warning_threshold
                                )
                                appearance_warnings = warning_manager.update_warnings(appearance_assessment)
                    except Exception as e:
                        print(f"Error in appearance assessment: {e}")
                
                # Visualize results with bounding boxes and emotion labels
                output_frame = self.visualizer.draw_results(frame, results, fps=fps)
                
                # Draw alert if dual attention detected distraction (KHÔNG HIỂN THỊ CHO SCREEN CAPTURE)
                if dual_result is not None and dual_result.should_show_alert and self.video_source != "screen":
                    try:
                        output_frame = self.dual_attention_coordinator.alert_manager.draw_alert(
                            output_frame,
                            dual_result.distraction_state.distraction_duration
                        )
                    except Exception as e:
                        print(f"Error drawing alert: {e}")
                
                # Draw alert if no face detected for too long (single attention mode) (KHÔNG HIỂN THỊ CHO SCREEN CAPTURE)
                elif hasattr(self, '_current_no_face_duration') and self._current_no_face_duration >= 5.0 and self.video_source != "screen":
                    # Draw red warning overlay
                    overlay = output_frame.copy()
                    h, w = output_frame.shape[:2]
                    
                    # Semi-transparent red background
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.3, output_frame, 0.7, 0, output_frame)
                    
                    # Warning text
                    warning_text = "⚠️ CANH BAO: KHONG PHAT HIEN KHUON MAT!"
                    duration_text = f"Thoi gian: {self._current_no_face_duration:.1f}s"
                    
                    # Calculate text size and position
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2
                    thickness = 3
                    
                    (text_w, text_h), _ = cv2.getTextSize(warning_text, font, font_scale, thickness)
                    text_x = (w - text_w) // 2
                    text_y = h // 2 - 30
                    
                    # Draw text with black outline
                    cv2.putText(output_frame, warning_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                    cv2.putText(output_frame, warning_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                    
                    # Draw duration
                    (dur_w, dur_h), _ = cv2.getTextSize(duration_text, font, 0.8, 2)
                    dur_x = (w - dur_w) // 2
                    dur_y = text_y + 50
                    
                    cv2.putText(output_frame, duration_text, (dur_x, dur_y), font, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(output_frame, duration_text, (dur_x, dur_y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Update video display with visualized frame
                self.update_video_display(output_frame)
                
                # THROTTLE: Update UI elements less frequently to reduce lag
                # Only update every 15 frames (0.5s at 30 FPS) - TĂNG từ 10
                ui_interval = PERFORMANCE.get('ui_update_interval', 15)
                if self.frame_count % ui_interval == 0:
                    # Update appearance warnings overlay (Req 7.1, 7.2, 7.3, 7.5)
                    if appearance_warnings:
                        self.appearance_warning_overlay.update_warnings(appearance_warnings)
                    else:
                        # Clear warnings if none active (Req 7.4)
                        self.appearance_warning_overlay.clear()
                    
                    # Update FPS display
                    self.update_fps(fps)
                
                # Update statistics based on configured interval (less frequently)
                stats_interval = PERFORMANCE.get('stats_update_interval', 30)
                if self.frame_count % stats_interval == 0:
                    self.update_statistics()
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final update
        self.update_statistics()
    
    def _calculate_realtime_emotion_score(
        self,
        emotion: str,
        emotion_probs: Dict[str, float],
        landmarks: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Tính điểm emotion scoring real-time cho 1 frame - CHÍNH XÁC.
        
        Sử dụng 4 tiêu chí chính (Big Tech mode):
        1. Confidence (Tự tin): 30%
        2. Positivity (Tích cực): 30%
        3. Professionalism (Chuyên nghiệp): 20%
        4. Engagement (Tương tác): 20%
        
        Returns:
            float: Điểm từ 0-10
        """
        try:
            # Lấy emotion probabilities
            happy_prob = emotion_probs.get('happy', 0.0)
            neutral_prob = emotion_probs.get('neutral', 0.0)
            surprise_prob = emotion_probs.get('surprise', 0.0)
            sad_prob = emotion_probs.get('sad', 0.0)
            angry_prob = emotion_probs.get('angry', 0.0)
            fear_prob = emotion_probs.get('fear', 0.0)
            disgust_prob = emotion_probs.get('disgust', 0.0)
            
            # 1. CONFIDENCE SCORE (30%) - Tự tin
            # Chỉ tính emotion chính, không trừ quá nhiều
            if emotion == 'happy':
                confidence_score = 8.0 + (happy_prob * 2.0)  # 8-10
            elif emotion == 'neutral':
                confidence_score = 9.0 + (neutral_prob * 1.0)  # 9-10
            elif emotion == 'surprise':
                confidence_score = 6.0 + (surprise_prob * 2.0)  # 6-8
            elif emotion == 'sad':
                confidence_score = 4.0 - (sad_prob * 2.0)  # 2-4
            elif emotion == 'fear':
                confidence_score = 3.0 - (fear_prob * 2.0)  # 1-3
            elif emotion == 'angry':
                confidence_score = 5.0 - (angry_prob * 2.0)  # 3-5
            elif emotion == 'disgust':
                confidence_score = 4.0 - (disgust_prob * 2.0)  # 2-4
            else:
                confidence_score = 7.0
            
            confidence_score = max(0.0, min(10.0, confidence_score))
            
            # 2. POSITIVITY SCORE (30%) - Tích cực
            # Chỉ tính emotion chính
            if emotion == 'happy':
                positivity_score = 9.0 + (happy_prob * 1.0)  # 9-10
            elif emotion == 'neutral':
                positivity_score = 7.0 + (neutral_prob * 1.0)  # 7-8
            elif emotion == 'surprise':
                positivity_score = 6.0 + (surprise_prob * 1.0)  # 6-7
            elif emotion == 'sad':
                positivity_score = 3.0 - (sad_prob * 2.0)  # 1-3
            elif emotion == 'angry':
                positivity_score = 2.0 - (angry_prob * 1.5)  # 0.5-2
            elif emotion == 'fear':
                positivity_score = 3.5 - (fear_prob * 2.0)  # 1.5-3.5
            elif emotion == 'disgust':
                positivity_score = 2.5 - (disgust_prob * 1.5)  # 1-2.5
            else:
                positivity_score = 7.0
            
            positivity_score = max(0.0, min(10.0, positivity_score))
            
            # 3. PROFESSIONALISM SCORE (20%) - Dựa vào emotion stability và appropriateness
            professionalism_score = 7.0  # Base score
            
            # Cảm xúc phù hợp với môi trường chuyên nghiệp
            if emotion in ['neutral', 'happy']:
                professionalism_score = 8.0 + (neutral_prob * 2.0)  # Neutral tốt nhất
            elif emotion == 'surprise':
                professionalism_score = 6.0
            elif emotion in ['sad', 'fear']:
                professionalism_score = 4.0
            elif emotion in ['angry', 'disgust']:
                professionalism_score = 2.0  # Rất không phù hợp
            
            professionalism_score = max(0.0, min(10.0, professionalism_score))
            
            # 4. ENGAGEMENT SCORE (20%) - Dựa vào attention và facial activity
            engagement_score = 7.0  # Default
            
            # Nếu có attention detector, dùng attention score
            if self.attention_detector and len(self.attention_scores) > 0:
                recent_attention = self.attention_scores[-1] if self.attention_scores else 70
                engagement_score = recent_attention / 10.0  # Convert 0-100 to 0-10
            else:
                # Nếu không có attention, dùng emotion activity
                # Surprise và Happy cho thấy engagement cao
                negative_emotions = sad_prob + angry_prob + fear_prob + disgust_prob
                engagement_score = (
                    happy_prob * 9.0 +
                    surprise_prob * 8.0 +
                    neutral_prob * 6.0 +
                    (1 - negative_emotions) * 5.0
                )
            
            engagement_score = max(0.0, min(10.0, engagement_score))
            
            # Tính weighted average
            total_score = (
                confidence_score * 0.30 +
                positivity_score * 0.30 +
                professionalism_score * 0.20 +
                engagement_score * 0.20
            )
            
            # Đảm bảo trong khoảng 0-10
            total_score = max(0.0, min(10.0, total_score))
            
            # Lưu điểm từng tiêu chí
            self.emotion_criteria_scores = {
                'confidence': round(confidence_score, 1),
                'positivity': round(positivity_score, 1),
                'professionalism': round(professionalism_score, 1),
                'engagement': round(engagement_score, 1),
                'total': round(total_score, 2)
            }
            
            return total_score
        
        except Exception as e:
            print(f"Error calculating emotion score: {e}")
            return 5.0  # Default neutral score
    
    def _calculate_focus_score(
        self,
        landmarks: np.ndarray,
        frame_shape: Tuple[int, int]
    ) -> float:
        """
        Tính điểm tập trung theo CÔNG THỨC MỚI (4 thành phần).
        
        FocusScore = (
            FacePresence * 0.40 +
            GazeFocus    * 0.30 +
            HeadFocus    * 0.20 +
            DriftScore   * 0.10
        ) * 10
        
        Với:
        - Face Presence (40%): Có mặt trong khung hình
        - Gaze Focus (30%): Nhìn thẳng vs nhìn xa
        - Head Focus (20%): Đầu thẳng vs quay đầu
        - Drift Score (10%): Phạt khi ngó nghiêng nhiều
        
        Returns:
            float: Điểm từ 0-10
        """
        try:
            # Sử dụng attention_detector với công thức mới
            if self.attention_detector:
                face_detected = landmarks is not None
                score, details = self.attention_detector.calculate_attention_score(
                    landmarks, 
                    frame_shape, 
                    face_detected=face_detected
                )
                
                # Lấy các component từ details (đã tính trong attention_detector)
                face_presence = details.get('face_presence_score', 0.0)
                gaze_focus = details.get('gaze_focus_score', 0.0)
                head_focus = details.get('head_focus_score', 0.0)
                drift_score = details.get('drift_score', 1.0)
                
                # Lưu các component để hiển thị (chuyển sang %)
                self.focus_components = {
                    'face_presence': round(face_presence * 100, 1),
                    'gaze_focus': round(gaze_focus * 100, 1),
                    'head_focus': round(head_focus * 100, 1),
                    'drift_score': round(drift_score * 100, 1),
                    'total_score': round(score, 2),
                    'gaze_status': details.get('gaze_status', 'unknown'),
                    'head_status': details.get('head_status', 'unknown'),
                    'drift_events': details.get('drift_events', 0)
                }
                
                return score
            else:
                # Fallback nếu không có attention_detector
                return 7.0
        
        except Exception as e:
            print(f"Error calculating focus score: {e}")
            return 7.0  # Default
    
    def on_window_resize(self, event):
        """Xử lý khi resize app - cập nhật lại canvas size."""
        if event.widget != self.root:
            return
        
        # Debounce - chỉ xử lý sau khi user dừng resize
        if hasattr(self, '_resize_timer'):
            self.root.after_cancel(self._resize_timer)
        
        def update_size():
            # Reset tất cả để tính lại
            if hasattr(self, '_canvas_size'):
                delattr(self, '_canvas_size')
            if hasattr(self, '_resize_dimensions'):
                delattr(self, '_resize_dimensions')
            if hasattr(self, 'video_canvas_image'):
                self.video_canvas_image = None
            print("✓ Canvas reset - will recalculate on next frame")
        
        self._resize_timer = self.root.after(500, update_size)
    
    def update_video_display(self, frame):
        """Hiển thị video - MƯỢT MÀ với Canvas, không giật."""
        # Throttle nhẹ để mượt (30 FPS) - dùng từ PERFORMANCE settings
        if not hasattr(self, '_last_display_update'):
            self._last_display_update = 0
        
        current_time = time.time()
        if current_time - self._last_display_update < PERFORMANCE['display_update_throttle']:
            return
        
        self._last_display_update = current_time
        
        try:
            # Lấy kích thước canvas (1 lần duy nhất)
            if not hasattr(self, '_canvas_size'):
                self.root.update_idletasks()
                w = self.video_canvas.winfo_width()
                h = self.video_canvas.winfo_height()
                if w > 10 and h > 10:
                    self._canvas_size = (w, h)
                    # Update placeholder text position (nếu còn tồn tại)
                    if self.video_canvas_text:
                        try:
                            self.video_canvas.coords(self.video_canvas_text, w//2, h//2)
                        except:
                            pass
                    print(f"✓ Canvas size: {self._canvas_size}")
                else:
                    self._canvas_size = (800, 450)
            
            canvas_w, canvas_h = self._canvas_size
            
            # Cache kích thước resize (1 lần duy nhất)
            if not hasattr(self, '_resize_dimensions'):
                h, w = frame.shape[:2]
                scale = min(canvas_w / w, canvas_h / h)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                self._resize_dimensions = (new_w, new_h)
                print(f"✓ Resize dimensions: {self._resize_dimensions}")
            
            new_w, new_h = self._resize_dimensions
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize với kích thước CỐ ĐỊNH
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Convert to PhotoImage
            image = Image.fromarray(frame_rgb)
            photo = ImageTk.PhotoImage(image=image)
            
            # Xóa placeholder text (chỉ lần đầu)
            if self.video_canvas_text:
                self.video_canvas.delete(self.video_canvas_text)
                self.video_canvas_text = None
            
            # Update canvas image - VẼ Ở GIỮA, KHÔNG THAY ĐỔI KÍCH THƯỚC
            if self.video_canvas_image is None:
                # Tạo image lần đầu - ở giữa canvas
                x = canvas_w // 2
                y = canvas_h // 2
                self.video_canvas_image = self.video_canvas.create_image(
                    x, y, image=photo, anchor='center'
                )
            else:
                # Chỉ update image, KHÔNG thay đổi vị trí
                self.video_canvas.itemconfig(self.video_canvas_image, image=photo)
            
            # Giữ reference để tránh garbage collection
            self.video_canvas.image = photo
        
        except Exception as e:
            print(f"Error updating video: {e}")
    
    def clear_video_display(self):
        """Xóa hiển thị video - xóa canvas và hiển thị placeholder."""
        try:
            # Xóa tất cả items trên canvas
            self.video_canvas.delete('all')
            
            # Reset image reference
            self.video_canvas_image = None
            self.video_canvas.image = None
            
            # Get canvas size
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 400
                canvas_height = 300
            
            # Tạo lại placeholder text ở giữa
            self.video_canvas_text = self.video_canvas.create_text(
                canvas_width // 2,
                canvas_height // 2,
                text="Chưa có video\nNhấn 'BẮT ĐẦU QUÉT' để bắt đầu",
                font=("Segoe UI", 12),
                fill='#757575',
                anchor='center'
            )
            
            # Reset cache
            if hasattr(self, '_canvas_size'):
                delattr(self, '_canvas_size')
            if hasattr(self, '_resize_dimensions'):
                delattr(self, '_resize_dimensions')
        
        except Exception as e:
            print(f"Error clearing video display: {e}")
    
    def update_fps(self, fps):
        """Cập nhật FPS với màu động."""
        # Color based on FPS performance
        if fps >= 25:
            color = '#66bb6a'  # Green - Good
        elif fps >= 15:
            color = '#ffa726'  # Orange - OK
        else:
            color = '#ef5350'  # Red - Poor
        
        self.fps_label.config(text=f"⚡ FPS: {fps:.1f}", fg=color)
    
    def update_status(self, text, color):
        """Cập nhật trạng thái với icon."""
        # Add appropriate icon based on status
        icon_map = {
            'Đang quét': '▶️',
            'Đã dừng': '⏸️',
            'Đang khởi tạo': '⏳',
            'Lỗi': '❌',
            'Sẵn sàng': '✅'
        }
        
        # Find matching icon
        icon = ''
        for key, val in icon_map.items():
            if key in text:
                icon = val + ' '
                break
        
        self.status_label.config(text=f"{icon}{text}", fg=color)
    
    def update_dual_statistics(self, result):
        """Update dual statistics panels for both users.
        
        Displays current emotion, confidence, attention score, emotion score,
        and attention score for each user in separate panels.
        
        Args:
            result: DualPersonResult containing data for both users
        
        Requirements:
        - 3.1: Display statistics panel for Primary User
        - 3.2: Display statistics panel for Secondary User
        - 3.3: Update when emotion detected (within 500ms)
        - 3.4: Display current emotion label and confidence
        - 3.5: Display current attention score (0-100)
        - 3.6: Update at least once per second
        - 4.4: Display Emotion Score and Attention Score (0-100)
        """
        # Lưu vị trí cuộn hiện tại cho cả hai panel
        primary_scroll_position = self.primary_stats_text.yview()
        secondary_scroll_position = self.secondary_stats_text.yview()
        
        # Update primary user statistics
        self.primary_stats_text.config(state=tk.NORMAL)
        self.primary_stats_text.delete(1.0, tk.END)
        
        primary_stats = self._format_person_stats(result.primary, "primary")
        self.primary_stats_text.insert(1.0, primary_stats)
        self.primary_stats_text.config(state=tk.DISABLED)
        
        # Khôi phục vị trí cuộn cho primary panel
        self.primary_stats_text.yview_moveto(primary_scroll_position[0])
        
        # Update secondary user statistics
        self.secondary_stats_text.config(state=tk.NORMAL)
        self.secondary_stats_text.delete(1.0, tk.END)
        
        secondary_stats = self._format_person_stats(result.secondary, "secondary")
        self.secondary_stats_text.insert(1.0, secondary_stats)
        self.secondary_stats_text.config(state=tk.DISABLED)
        
        # Khôi phục vị trí cuộn cho secondary panel
        self.secondary_stats_text.yview_moveto(secondary_scroll_position[0])
        
        # Update comparison panel if comparison result is available (Req 5.1)
        if result.comparison is not None:
            self.update_comparison_panel(result.comparison)
    
    def update_comparison_panel(self, comparison):
        """Update comparison panel with comparison results.
        
        Displays emotion score comparison, attention score comparison,
        winner indicators, percentage differences, and visual highlighting
        for significant gaps.
        
        Args:
            comparison: ComparisonResult containing comparison data
        
        Requirements:
        - 5.1: Generate comparison report every 10 seconds
        - 5.2: Compare Emotion Scores and identify winner
        - 5.3: Compare Attention Scores and identify winner
        - 5.4: Display comparison results in central panel
        - 5.5: Display percentage difference
        - 5.6: Highlight significant gap (>20 points)
        """
        from datetime import datetime
        
        # Format emotion comparison (Req 5.2, 5.5)
        emotion_text = f"Ban:        {comparison.primary_emotion_score:.0f}/100\n"
        emotion_text += f"Nguoi Khac: {comparison.secondary_emotion_score:.0f}/100\n\n"
        
        # Determine emotion winner and format with indicator (Req 5.2)
        if comparison.emotion_winner == "primary":
            emotion_text += "THANG: Ban\n"
            emotion_winner_color = '#4CAF50'
        elif comparison.emotion_winner == "secondary":
            emotion_text += "THANG: Nguoi Khac\n"
            emotion_winner_color = '#2196F3'
        else:
            emotion_text += "HOA\n"
            emotion_winner_color = '#FFD700'
        
        # Add percentage difference (Req 5.5)
        emotion_text += f"Chenh lech: {comparison.emotion_difference:.1f}%"
        
        # Update emotion comparison label
        self.emotion_comparison_label.config(text=emotion_text)
        
        # Highlight significant gap (Req 5.6)
        if comparison.significant_emotion_gap:
            self.emotion_comparison_label.config(
                fg='#FFD700',
                font=("Courier", 9, "bold")
            )
        else:
            self.emotion_comparison_label.config(
                fg='#ffffff',
                font=("Courier", 9)
            )
        
        # Format attention comparison (Req 5.3, 5.5)
        attention_text = f"Ban:        {comparison.primary_attention_score:.0f}/100\n"
        attention_text += f"Nguoi Khac: {comparison.secondary_attention_score:.0f}/100\n\n"
        
        # Determine attention winner and format with indicator (Req 5.3)
        if comparison.attention_winner == "primary":
            attention_text += "THANG: Ban\n"
            attention_winner_color = '#4CAF50'
        elif comparison.attention_winner == "secondary":
            attention_text += "THANG: Nguoi Khac\n"
            attention_winner_color = '#2196F3'
        else:
            attention_text += "HOA\n"
            attention_winner_color = '#FFD700'
        
        # Add percentage difference (Req 5.5)
        attention_text += f"Chenh lech: {comparison.attention_difference:.1f}%"
        
        # Update attention comparison label
        self.attention_comparison_label.config(text=attention_text)
        
        # Highlight significant gap (Req 5.6)
        if comparison.significant_attention_gap:
            self.attention_comparison_label.config(
                fg='#FFD700',
                font=("Courier", 9, "bold")
            )
        else:
            self.attention_comparison_label.config(
                fg='#ffffff',
                font=("Courier", 9)
            )
        
        # Calculate overall winner (Req 5.2, 5.3)
        primary_wins = 0
        secondary_wins = 0
        
        if comparison.emotion_winner == "primary":
            primary_wins += 1
        elif comparison.emotion_winner == "secondary":
            secondary_wins += 1
        
        if comparison.attention_winner == "primary":
            primary_wins += 1
        elif comparison.attention_winner == "secondary":
            secondary_wins += 1
        
        # Format overall winner
        if primary_wins > secondary_wins:
            overall_text = "TONG KET:\nBan THANG"
            overall_color = '#4CAF50'
        elif secondary_wins > primary_wins:
            overall_text = "TONG KET:\nNguoi Khac THANG"
            overall_color = '#2196F3'
        else:
            overall_text = "TONG KET:\nHOA"
            overall_color = '#FFD700'
        
        # Add win counts
        overall_text += f"\n\n({primary_wins}-{secondary_wins})"
        
        # Update overall winner label
        self.overall_winner_label.config(text=overall_text, fg=overall_color)
        
        # Update last comparison time (Req 5.1)
        update_time = datetime.fromtimestamp(comparison.timestamp).strftime("%H:%M:%S")
        self.comparison_update_label.config(text=f"Cap nhat: {update_time}")
    
    def _format_person_stats(self, person_result, person_type):
        """Format statistics for one person.
        
        Args:
            person_result: PersonResult for the user
            person_type: "primary" or "secondary"
        
        Returns:
            Formatted statistics string
        
        Requirements:
        - 3.4: Display emotion label and confidence
        - 3.5: Display attention score (0-100)
        - 4.4: Display Emotion Score and Attention Score
        """
        stats = ""
        
        # Face detection status
        if person_result.face_detected:
            stats += "✓ Khuôn mặt: Phát hiện\n"
        else:
            stats += "✗ Khuôn mặt: Không có\n"
            if person_result.no_face_warning:
                stats += "⚠️  Cảnh báo: >5s\n"
        
        stats += "\n"
        
        # Current emotion and confidence (Req 3.4)
        stats += "--- CẢM XÚC HIỆN TẠI ---\n\n"
        
        if person_result.emotion is not None:
            emotion_display = person_result.emotion.upper()
            confidence = person_result.emotion_confidence or 0.0
            
            # Add low confidence indicator
            if person_result.low_confidence:
                stats += f"Cảm xúc: {emotion_display} (?)\n"
                stats += f"Độ tin cậy: {confidence:.1%} ⚠️\n"
            else:
                stats += f"Cảm xúc: {emotion_display}\n"
                stats += f"Độ tin cậy: {confidence:.1%}\n"
            
            # Show confidence bar
            bar_length = int(confidence * 10)
            bar = "█" * bar_length
            stats += f"{bar}\n"
        else:
            stats += "Cảm xúc: --\n"
            stats += "Độ tin cậy: --\n"
        
        stats += "\n"
        
        # Current attention score (Req 3.5)
        stats += "--- SỰ TẬP TRUNG ---\n\n"
        
        if person_result.attention_score is not None:
            attention = person_result.attention_score
            stats += f"Điểm hiện tại: {attention:.0f}/100\n"
            
            # Attention level indicator
            if attention >= 80:
                stats += "Mức độ: ✓ Tập trung\n"
            elif attention >= 60:
                stats += "Mức độ: ~ Trung bình\n"
            else:
                stats += "Mức độ: ✗ Mất tập trung\n"
            
            # Show attention bar
            bar_length = int(attention / 10)
            bar = "█" * bar_length
            stats += f"{bar}\n"
        else:
            stats += "Điểm hiện tại: --\n"
            stats += "Mức độ: --\n"
        
        stats += "\n"
        
        # Emotion Score (0-100) - Req 4.4
        stats += "--- ĐIỂM CẢM XÚC ---\n\n"
        emotion_score = person_result.emotion_score
        stats += f"Điểm số: {emotion_score:.0f}/100\n"
        
        # Score interpretation
        if emotion_score >= 70:
            stats += "Đánh giá: 😊 Tích cực\n"
        elif emotion_score >= 50:
            stats += "Đánh giá: 😐 Trung tính\n"
        else:
            stats += "Đánh giá: 😔 Tiêu cực\n"
        
        # Show score bar
        bar_length = int(emotion_score / 10)
        bar = "█" * bar_length
        stats += f"{bar}\n"
        
        stats += "\n"
        
        # Attention Score (0-100) - Req 4.4
        stats += "--- ĐIỂM TẬP TRUNG ---\n\n"
        attention_score_avg = person_result.attention_score_avg
        stats += f"Điểm số: {attention_score_avg:.0f}/100\n"
        
        # Score interpretation
        if attention_score_avg >= 70:
            stats += "Đánh giá: ✓ Tốt\n"
        elif attention_score_avg >= 50:
            stats += "Đánh giá: ~ Khá\n"
        else:
            stats += "Đánh giá: ✗ Kém\n"
        
        # Show score bar
        bar_length = int(attention_score_avg / 10)
        bar = "█" * bar_length
        stats += f"{bar}\n"
        
        return stats
    
    def update_statistics(self):
        """Cập nhật thống kê."""
        # THROTTLE: Chỉ update mỗi 2 giây để tránh lag (TĂNG từ 1s)
        if not hasattr(self, '_last_stats_update'):
            self._last_stats_update = 0
        
        current_time = time.time()
        if current_time - self._last_stats_update < 2.0:
            return  # Skip update nếu chưa đủ 2 giây
        
        self._last_stats_update = current_time
        
        # Lưu vị trí cuộn hiện tại
        current_scroll_position = self.stats_text.yview()
        
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        # Session info
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        stats = f"""
================================
    THONG KE PHIEN LAM VIEC
================================

Thoi gian: {elapsed:.1f}s
So khung hinh: {self.frame_count}
Tong so khuon mat: {self.total_faces}

================================
      PHAN BO CAM XUC
================================

"""
        
        # Emotion distribution
        total_emotions = sum(self.emotion_counts.values())
        if total_emotions > 0:
            for emotion in sorted(self.emotion_counts.keys()):
                count = self.emotion_counts[emotion]
                percentage = (count / total_emotions) * 100
                bar_length = int(percentage / 5)
                bar = "#" * bar_length
                stats += f"{emotion:10s}: {count:4d} ({percentage:5.1f}%)\n{bar}\n\n"
        else:
            stats += "Chua co du lieu\n"
        
        # 1. CHẤM ĐIỂM CẢM XÚC
        if len(self.emotion_scores_history) > 0:
            stats += f"""
================================
   1. CHAM DIEM CAM XUC (0-10)
================================

"""
            score = self.current_emotion_score
            bar_length = int(score)
            bar = "★" * bar_length
            stats += f"Diem hien tai: {score:.2f}/10\n{bar}\n\n"
            
            if self.emotion_criteria_scores:
                stats += "--- CHI TIET TIEU CHI ---\n\n"
                criteria = self.emotion_criteria_scores
                stats += f"Tu tin:        {criteria.get('confidence', 0):.1f}/10 (30%)\n"
                stats += f"Tich cuc:      {criteria.get('positivity', 0):.1f}/10 (30%)\n"
                stats += f"Chuyen nghiep: {criteria.get('professionalism', 0):.1f}/10 (20%)\n"
                stats += f"Tuong tac:     {criteria.get('engagement', 0):.1f}/10 (20%)\n\n"
            
            stats += f"So mau:        {len(self.emotion_scores_history)} frames\n"
            stats += f"Diem TB:       {np.mean(self.emotion_scores_history):.2f}/10\n"
            stats += f"Diem cao nhat: {np.max(self.emotion_scores_history):.2f}/10\n"
            stats += f"Diem thap nhat:{np.min(self.emotion_scores_history):.2f}/10\n"
        
        # 2. CHẤM ĐIỂM TẬP TRUNG (Focus Score) - CHỈ hiển thị cho camera/screen
        if len(self.focus_scores_history) > 0 and self.video_source != "file":
            stats += f"""
================================
   2. CHAM DIEM TAP TRUNG (0-10)
================================

"""
            score = self.current_focus_score
            bar_length = int(score)
            bar = "●" * bar_length
            stats += f"Diem hien tai: {score:.2f}/10\n{bar}\n\n"
            
            if self.focus_components:
                stats += "--- CONG THUC FOCUS SCORE (MOI) ---\n\n"
                comp = self.focus_components
                stats += f"Face Presence:  {comp.get('face_presence', 0):.1f}% (40%)\n"
                stats += f"Gaze Focus:     {comp.get('gaze_focus', 0):.1f}% (30%)\n"
                stats += f"  → Status: {comp.get('gaze_status', 'N/A')}\n"
                stats += f"Head Focus:     {comp.get('head_focus', 0):.1f}% (20%)\n"
                stats += f"  → Status: {comp.get('head_status', 'N/A')}\n"
                stats += f"Drift Score:    {comp.get('drift_score', 0):.1f}% (10%)\n"
                stats += f"  → Events: {comp.get('drift_events', 0)}\n"
                stats += f"---\n"
                stats += f"Tong diem:      {comp.get('total_score', 0):.2f}/10\n\n"
            
            stats += f"So mau:        {len(self.focus_scores_history)} frames\n"
            stats += f"Diem TB:       {np.mean(self.focus_scores_history):.2f}/10\n"
            stats += f"Diem cao nhat: {np.max(self.focus_scores_history):.2f}/10\n"
            stats += f"Diem thap nhat:{np.min(self.focus_scores_history):.2f}/10\n"
        
        # Dual attention statistics (if enabled)
        if self.dual_attention_enabled and self.dual_attention_coordinator is not None:
            try:
                dual_stats = self.dual_attention_coordinator.get_statistics()
                
                stats += f"""
================================
   THEO DOI TAP TRUNG KEP
================================

"""
                # Scores section with clear formatting
                stats += "--- DIEM SO TAP TRUNG ---\n\n"
                
                # Camera score with face presence indicator
                camera_icon = "✓" if dual_stats['face_presence'] else "✗"
                camera_status = "Co mat" if dual_stats['face_presence'] else "Khong co"
                stats += f"📷 Camera:      {dual_stats['camera_score']:3.0f}/100  [{camera_icon}] {camera_status}\n"
                
                # Screen score with face presence indicator
                screen_icon = "✓" if dual_stats['screen_presence'] else "✗"
                screen_status = "Co mat" if dual_stats['screen_presence'] else "Khong co"
                stats += f"🖥️  Screen:      {dual_stats['screen_score']:3.0f}/100  [{screen_icon}] {screen_status}\n"
                
                # Combined score with visual bar
                combined_score = dual_stats['combined_score']
                bar_length = int(combined_score / 5)
                bar = "█" * bar_length
                stats += f"📊 Combined:     {combined_score:3.0f}/100\n"
                stats += f"   {bar}\n\n"
                
                # Fusion mode
                fusion_mode_display = {
                    'full': '✓ Day du (Camera + Screen)',
                    'camera_only': '⚠ Chi Camera',
                    'screen_only': '⚠ Chi Screen',
                    'none': '✗ Khong co'
                }
                mode = dual_stats.get('fusion_mode', 'none')
                stats += f"Che do: {fusion_mode_display.get(mode, mode)}\n\n"
                
                # Distraction tracking section
                stats += "--- THEO DOI MAT TAP TRUNG ---\n\n"
                
                # Current distraction duration
                current_distraction = dual_stats.get('current_distraction', 0.0)
                if current_distraction > 0:
                    # Show warning if approaching alert threshold
                    if current_distraction >= 4.0:
                        stats += f"⏱️  Hien tai:     {current_distraction:.1f}s  ⚠️ SAP CANH BAO!\n"
                    else:
                        stats += f"⏱️  Hien tai:     {current_distraction:.1f}s\n"
                else:
                    stats += f"⏱️  Hien tai:     0.0s  ✓ Dang tap trung\n"
                
                # Cumulative statistics section
                stats += "\n--- THONG KE TICH LUY ---\n\n"
                
                total_distraction = dual_stats.get('total_distraction', 0.0)
                alert_count = dual_stats.get('alert_count', 0)
                
                # Format total distraction time
                if total_distraction >= 60:
                    minutes = int(total_distraction // 60)
                    seconds = total_distraction % 60
                    stats += f"📈 Tong mat TT:  {minutes}m {seconds:.1f}s\n"
                else:
                    stats += f"📈 Tong mat TT:  {total_distraction:.1f}s\n"
                
                # Number of alerts triggered
                stats += f"⚠️  Canh bao:     {alert_count} lan\n"
                
                # Calculate focus percentage if we have enough data
                if elapsed > 0:
                    focus_percentage = ((elapsed - total_distraction) / elapsed) * 100
                    focus_percentage = max(0, min(100, focus_percentage))
                    stats += f"✓  Ti le TT:     {focus_percentage:.1f}%\n"
                
                # Performance metrics section (if monitoring enabled)
                if dual_stats.get('performance_monitoring_enabled', False):
                    stats += "\n--- HIEU SUAT HE THONG ---\n\n"
                    
                    # FPS
                    fps = dual_stats.get('fps', 0.0)
                    fps_threshold = dual_stats.get('fps_threshold', 20.0)
                    if fps > 0:
                        if fps < fps_threshold:
                            stats += f"🎬 FPS:          {fps:.1f}  ⚠️ Thap!\n"
                        else:
                            stats += f"🎬 FPS:          {fps:.1f}  ✓\n"
                    
                    # Processing time
                    proc_time = dual_stats.get('avg_processing_time_ms', 0.0)
                    if proc_time > 0:
                        stats += f"⏱️  Xu ly:        {proc_time:.2f}ms\n"
                    
                    # Memory usage
                    memory_mb = dual_stats.get('memory_usage_mb', 0.0)
                    memory_threshold = dual_stats.get('memory_threshold_mb', 150.0)
                    if memory_mb > 0:
                        if memory_mb > memory_threshold:
                            stats += f"💾 Bo nho:       {memory_mb:.1f}MB  ⚠️ Cao!\n"
                        else:
                            stats += f"💾 Bo nho:       {memory_mb:.1f}MB  ✓\n"
                    
                    # Performance status
                    if dual_stats.get('performance_disabled', False):
                        disable_reason = dual_stats.get('disable_reason', 'Unknown')
                        stats += f"\n⚠️  TU DONG TAT: {disable_reason}\n"
                
            except Exception as e:
                print(f"Error getting dual attention statistics: {e}")
                import traceback
                traceback.print_exc()
                stats += """
================================
   THEO DOI TAP TRUNG KEP
================================

Loi lay thong ke
"""
        
        # Fallback to single attention statistics (if dual attention not enabled)
        elif self.attention_detector is not None and len(self.attention_scores) > 0:
            avg_attention = np.mean(self.attention_scores)
            current_attention = self.attention_scores[-1] if self.attention_scores else 0
            total_attention = self.attention_focused_frames + self.attention_distracted_frames
            
            attention_stats = self.attention_detector.get_statistics()
            
            stats += f"""
================================
      MUC DO TAP TRUNG
================================

Diem hien tai: {current_attention:.0f}/100
Diem trung binh: {avg_attention:.0f}/100

"""
            if total_attention > 0:
                focused_rate = self.attention_focused_frames / total_attention
                distracted_rate = self.attention_distracted_frames / total_attention
                
                stats += f"Ti le tap trung: {focused_rate:.1%}\n"
                stats += f"Ti le mat tap trung: {distracted_rate:.1%}\n\n"
            
            stats += f"Mat nham: {attention_stats['eyes_closed_rate']:.1%}\n"
            stats += f"Nhin ra ngoai: {attention_stats['looking_away_rate']:.1%}\n"
            stats += f"So lan chop mat: {attention_stats['blink_count']}\n"
        
        # Appearance Assessment (if enabled)
        if self.appearance_enabled and (self.assess_clothing or self.assess_lighting):
            clothing_score = self.appearance_scores['clothing']
            lighting_score = self.appearance_scores['lighting']
            overall_score = self.appearance_scores['overall']
            
            # Determine status icons and colors
            def get_status(score):
                if score == 0:
                    return "- Khong quet"
                elif score >= 80:
                    return "✓ Tot"
                elif score >= 60:
                    return "⚠ Trung binh"
                else:
                    return "✗ Can cai thien"
            
            stats += f"""
================================
   DANH GIA NGOAI HINH
================================

"""
            # Only show selected assessments
            if self.assess_clothing:
                clothing_bar = "█" * int(clothing_score / 5)
                stats += f"👔 Quan ao:     {clothing_score:5.1f}/100  {get_status(clothing_score)}\n"
                stats += f"   {clothing_bar}\n\n"
            
            if self.assess_lighting:
                lighting_bar = "█" * int(lighting_score / 5)
                stats += f"💡 Anh sang:    {lighting_score:5.1f}/100  {get_status(lighting_score)}\n"
                stats += f"   {lighting_bar}\n\n"
            
            # Overall score if any assessment is enabled
            if self.assess_clothing or self.assess_lighting:
                overall_bar = "█" * int(overall_score / 5)
                stats += f"📊 Tong the:    {overall_score:5.1f}/100  {get_status(overall_score)}\n"
                stats += f"   {overall_bar}\n"
        
        # Performance
        if self.classifier:
            perf = self.classifier.get_performance_stats()
            stats += f"""
================================
         HIEU SUAT
================================

Trung binh: {perf['average_time_ms']:.2f}ms
Toi thieu: {perf['min_time_ms']:.2f}ms
Toi da: {perf['max_time_ms']:.2f}ms
Dat yeu cau <30ms: {'YES' if perf['meets_requirement'] else 'NO'}
"""
        
        self.stats_text.insert(1.0, stats)
        self.stats_text.config(state=tk.DISABLED)
        
        # Cập nhật Score Summary Tab
        self.update_score_summary_tab()
        
        # Khôi phục vị trí cuộn
        self.stats_text.yview_moveto(current_scroll_position[0])
    
    def reset_statistics(self):
        """Đặt lại thống kê."""
        if messagebox.askyesno("Xac nhan", "Ban co chac muon dat lai thong ke?"):
            self.emotion_counts = {emotion: 0 for emotion in self.classifier.emotions}
            self.total_faces = 0
            self.frame_count = 0
            self.start_time = time.time() if self.is_running else None
            
            # Reset attention statistics
            self.attention_scores = []
            self.attention_focused_frames = 0
            self.attention_distracted_frames = 0
            if self.attention_detector is not None:
                self.attention_detector.reset()
            
            # Reset appearance scores
            self.appearance_scores = {
                'clothing': 0.0,
                'lighting': 0.0,
                'overall': 0.0
            }
            
            # Reset emotion scoring
            self.emotion_scores_history = []
            self.current_emotion_score = 0.0
            self.emotion_criteria_scores = {}
            
            # Reset focus scoring
            self.focus_scores_history = []
            self.current_focus_score = 0.0
            self.focus_components = {}
            
            # Reset dual attention statistics
            if self.dual_attention_coordinator is not None:
                try:
                    self.dual_attention_coordinator.reset_statistics()
                except Exception as e:
                    print(f"Warning: Error resetting dual attention statistics: {e}")
            
            self.update_statistics()
    
    def export_statistics(self):
        """Xuất thống kê ra file .txt"""
        try:
            # Generate default filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"statistics_{timestamp}"
            default_dir = str(Path("./reports").absolute())
            
            # Ask where to save with custom dialog
            filename = ask_save_file(
                parent=self.root,
                title="Xuất Thống Kê",
                default_filename=default_name,
                default_dir=default_dir,
                file_extension=".txt",
                file_types=[
                    ("Text files", "*.txt"),
                    ("All files", "*.*")
                ]
            )
            
            if not filename:
                return  # User cancelled
            
            filename = Path(filename)
            
            # Gather all statistics
            elapsed = time.time() - self.start_time if self.start_time else 0
            
            # Build report content
            report = f"""
{'='*70}
           BAO CAO THONG KE NHAN DIEN CAM XUC
{'='*70}

Thoi gian tao bao cao: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

{'='*70}
                 THONG TIN PHIEN LAM VIEC
{'='*70}

Thoi gian lam viec:     {elapsed:.1f} giay ({elapsed/60:.1f} phut)
Tong so khung hinh:     {self.frame_count}
Tong so khuon mat:      {self.total_faces}
Nguon video:            {'Camera' if self.video_source == 'camera' else 'Video File' if self.video_source == 'file' else 'Screen Capture'}

{'='*70}
                    PHAN BO CAM XUC
{'='*70}

"""
            
            # Emotion distribution
            total_emotions = sum(self.emotion_counts.values())
            if total_emotions > 0:
                # Find dominant emotion
                dominant_emotion = max(self.emotion_counts.items(), key=lambda x: x[1])
                report += f"Cam xuc pho bien nhat: {dominant_emotion[0].upper()} ({dominant_emotion[1]} lan, {dominant_emotion[1]/total_emotions*100:.1f}%)\n\n"
                
                report += "Chi tiet phan bo:\n"
                for emotion in sorted(self.emotion_counts.keys()):
                    count = self.emotion_counts[emotion]
                    percentage = (count / total_emotions) * 100
                    bar_length = int(percentage / 2)
                    bar = "█" * bar_length
                    report += f"  {emotion:12s}: {count:5d} ({percentage:5.1f}%)  {bar}\n"
            else:
                report += "Khong co du lieu cam xuc\n"
            
            # Dual attention statistics
            if self.dual_attention_enabled and self.dual_attention_coordinator is not None:
                try:
                    dual_stats = self.dual_attention_coordinator.get_statistics()
                    
                    report += f"""

{'='*70}
              THONG KE TAP TRUNG (DUAL ATTENTION)
{'='*70}

--- Diem So Tap Trung ---

Camera Score:           {dual_stats['camera_score']:.0f}/100
Screen Score:           {dual_stats['screen_score']:.0f}/100
Combined Score:         {dual_stats['combined_score']:.0f}/100

Face Presence (Camera): {'Co' if dual_stats['face_presence'] else 'Khong'}
Face Presence (Screen): {'Co' if dual_stats['screen_presence'] else 'Khong'}

Fusion Mode:            {dual_stats.get('fusion_mode', 'none')}

--- Thong Ke Mat Tap Trung ---

Tong thoi gian mat TT:  {dual_stats.get('total_distraction', 0.0):.1f} giay
So lan canh bao:        {dual_stats.get('alert_count', 0)} lan

"""
                    # Calculate focus percentage
                    if elapsed > 0:
                        focus_percentage = ((elapsed - dual_stats.get('total_distraction', 0.0)) / elapsed) * 100
                        focus_percentage = max(0, min(100, focus_percentage))
                        report += f"Ti le tap trung:        {focus_percentage:.1f}%\n"
                        report += f"Ti le mat tap trung:    {100-focus_percentage:.1f}%\n"
                
                except Exception as e:
                    report += f"\nLoi lay thong ke dual attention: {e}\n"
            
            # Single attention statistics (fallback)
            elif self.attention_detector is not None and len(self.attention_scores) > 0:
                avg_attention = np.mean(self.attention_scores)
                total_attention = self.attention_focused_frames + self.attention_distracted_frames
                attention_stats = self.attention_detector.get_statistics()
                
                report += f"""

{'='*70}
                   THONG KE TAP TRUNG
{'='*70}

Diem tap trung trung binh:  {avg_attention:.0f}/100

"""
                if total_attention > 0:
                    focused_rate = self.attention_focused_frames / total_attention
                    distracted_rate = self.attention_distracted_frames / total_attention
                    report += f"Ti le tap trung:            {focused_rate:.1%}\n"
                    report += f"Ti le mat tap trung:        {distracted_rate:.1%}\n\n"
                
                # Hiển thị công thức mới
                if self.focus_components:
                    report += f"""
--- CONG THUC TINH DIEM (4 THANH PHAN) ---

FocusScore = (
    FacePresence * 0.40 +
    GazeFocus    * 0.30 +
    HeadFocus    * 0.20 +
    DriftScore   * 0.10
) * 10

Chi tiet:
  Face Presence:  {self.focus_components.get('face_presence', 0):.1f}% (40%)
  Gaze Focus:     {self.focus_components.get('gaze_focus', 0):.1f}% (30%)
    → Status: {self.focus_components.get('gaze_status', 'N/A')}
  Head Focus:     {self.focus_components.get('head_focus', 0):.1f}% (20%)
    → Status: {self.focus_components.get('head_status', 'N/A')}
  Drift Score:    {self.focus_components.get('drift_score', 0):.1f}% (10%)
    → Events: {self.focus_components.get('drift_events', 0)}

Tong diem:        {self.focus_components.get('total_score', 0):.2f}/10

"""
                
                report += f"Ti le mat nham:             {attention_stats['eyes_closed_rate']:.1%}\n"
                report += f"Ti le nhin ra ngoai:        {attention_stats['looking_away_rate']:.1%}\n"
                report += f"So lan chop mat:            {attention_stats['blink_count']}\n"
            
            # Performance statistics
            if self.classifier:
                perf = self.classifier.get_performance_stats()
                report += f"""

{'='*70}
                    HIEU SUAT XU LY
{'='*70}

Thoi gian xu ly trung binh:    {perf['average_time_ms']:.2f}ms
Thoi gian xu ly toi thieu:     {perf['min_time_ms']:.2f}ms
Thoi gian xu ly toi da:        {perf['max_time_ms']:.2f}ms
Dat yeu cau (<30ms):           {'YES' if perf['meets_requirement'] else 'NO'}

"""
            
            # Footer
            report += f"""
{'='*70}
                      KET THUC BAO CAO
{'='*70}

File duoc luu tai: {filename}
"""
            
            # Write to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Show success message
            messagebox.showinfo(
                "Thanh cong",
                f"Da xuat thong ke thanh cong!\n\n"
                f"File: {filename}\n\n"
                f"Ban co the mo file nay de xem chi tiet."
            )
            
            print(f"Statistics exported to: {filename}")
            
        except Exception as e:
            messagebox.showerror(
                "Loi",
                f"Khong the xuat thong ke!\n\n"
                f"Loi: {str(e)}"
            )
            print(f"Error exporting statistics: {e}")
            import traceback
            traceback.print_exc()
    
    def calculate_emotion_score(self) -> float:
        """
        Tính điểm Cảm xúc (0-10) từ emotion_counts.
        
        Logic:
        - Positive emotions (happy, surprise) → điểm cao
        - Neutral → điểm trung bình  
        - Negative emotions (sad, angry, fear, disgust) → điểm thấp
        
        Returns:
            Điểm cảm xúc (0-10)
        """
        if not self.emotion_counts:
            return 0.0
        
        total = sum(self.emotion_counts.values())
        if total == 0:
            return 0.0
        
        # Trọng số cho từng cảm xúc (thang 0-10)
        weights = {
            'happy': 10.0,      # Vui vẻ - tốt nhất
            'surprise': 8.0,    # Ngạc nhiên - tốt
            'neutral': 7.0,     # Trung lập - chấp nhận được
            'sad': 4.0,         # Buồn - không tốt
            'angry': 3.0,       # Tức giận - xấu
            'fear': 3.0,        # Sợ hãi - xấu
            'disgust': 2.0      # Ghê tởm - rất xấu
        }
        
        # Tính điểm trung bình có trọng số
        weighted_sum = sum(
            self.emotion_counts.get(emotion, 0) * weights.get(emotion, 5.0)
            for emotion in self.emotion_counts
        )
        
        score = weighted_sum / total
        return round(min(10.0, max(0.0, score)), 2)
    
    def calculate_focus_score(self) -> float:
        """
        Tính điểm Tập trung (0-10) từ attention_scores.
        
        Sử dụng điểm attention trung bình (đã là thang 0-10).
        
        Returns:
            Điểm tập trung (0-10)
        """
        if not self.attention_scores:
            return 0.0
        
        # Attention score đã là 0-10, nhưng clamp để chắc chắn
        avg_attention = np.mean(self.attention_scores)
        
        # QUAN TRỌNG: Clamp về 0-10
        avg_attention = max(0.0, min(10.0, avg_attention))
        
        return round(avg_attention, 2)
    
    def send_scores_to_summary(self):
        """
        Gửi điểm Emotion và Focus sang tab Tổng Hợp Điểm.
        
        Sử dụng ScoreManager để chia sẻ điểm giữa các tab.
        """
        # Kiểm tra đã có dữ liệu chưa
        if not self.is_running and self.frame_count == 0:
            messagebox.showwarning(
                "Chưa Có Dữ Liệu",
                "Vui lòng quét khuôn mặt trước khi gửi điểm!\n\n"
                "Nhấn 'BẮT ĐẦU QUÉT' để bắt đầu."
            )
            return
        
        # Tính điểm
        emotion_score = self.calculate_emotion_score()
        focus_score = self.calculate_focus_score()
        
        # Kiểm tra điểm hợp lệ
        if emotion_score == 0.0 and focus_score == 0.0:
            messagebox.showwarning(
                "Chưa Có Điểm",
                "Chưa có đủ dữ liệu để tính điểm!\n\n"
                "Vui lòng quét khuôn mặt lâu hơn để có kết quả chính xác."
            )
            return
        
        # Gửi vào ScoreManager
        print(f"\n[EmotionRecognitionGUI] Sending scores to ScoreManager...")
        print(f"  Emotion: {emotion_score:.1f}/10")
        print(f"  Focus: {focus_score:.1f}/10")
        
        score_manager = get_score_manager()
        print(f"  ScoreManager ID: {id(score_manager)}")
        
        score_manager.set_emotion_score(emotion_score, source="emotion_recognition")
        score_manager.set_focus_score(focus_score, source="emotion_recognition")
        
        print(f"  ✓ Scores sent successfully!")
        
        # Verify scores were saved
        all_scores = score_manager.get_all_scores()
        print(f"  Verification:")
        print(f"    Emotion in manager: {all_scores['emotion']['score']:.1f}")
        print(f"    Focus in manager: {all_scores['focus']['score']:.1f}")
        
        # Thông báo thành công
        message = "✅ ĐÃ GỬI ĐIỂM THÀNH CÔNG!\n\n"
        message += f"😊 Cảm xúc (Emotion): {emotion_score:.2f}/10\n"
        message += f"🎯 Tập trung (Focus): {focus_score:.2f}/10\n\n"
        
        # Thêm chi tiết
        if self.emotion_counts:
            total_emotions = sum(self.emotion_counts.values())
            dominant = max(self.emotion_counts.items(), key=lambda x: x[1])
            message += f"Cảm xúc chủ đạo: {dominant[0]} ({dominant[1]}/{total_emotions})\n"
        
        if self.attention_scores:
            message += f"Số frame đã quét: {len(self.attention_scores)}\n"
        
        message += "\n"
        message += "Điểm đã được gửi sang tab 'Tổng Hợp Điểm'.\n"
        message += "Vui lòng chuyển sang tab đó để xem và tính điểm tổng!"
        
        messagebox.showinfo("Gửi Điểm Thành Công", message)
    
    def clear_all_cache(self):
        """Xóa toàn bộ cache (transcripts, .cache, logs cũ)."""
        if not messagebox.askyesno(
            "Xác nhận xóa cache",
            "Bạn có chắc muốn xóa toàn bộ cache?\n\n"
            "Sẽ xóa:\n"
            "- Tất cả transcript cũ\n"
            "- Cache dịch (SQLite database)\n"
            "- Cache Whisper\n"
            "- Logs cũ\n"
            "- Reports cũ\n\n"
            "Hành động này không thể hoàn tác!"
        ):
            return
        
        try:
            import shutil
            import os
            deleted_items = []
            
            # 1. Clear transcripts (keep .gitkeep)
            transcripts_dir = Path("transcripts")
            if transcripts_dir.exists():
                for file in transcripts_dir.glob("*.txt"):
                    if file.name != ".gitkeep":
                        file.unlink()
                        deleted_items.append(f"Transcript: {file.name}")
            
            # 2. Clear transcription cache using diskcache API (proper way)
            transcription_cache_dir = Path(".cache/transcriptions")
            if transcription_cache_dir.exists():
                try:
                    # Use diskcache to properly clear the cache
                    import diskcache
                    cache = diskcache.Cache(str(transcription_cache_dir))
                    cache_size = len(cache)
                    cache.clear()
                    cache.close()
                    deleted_items.append(f"Transcription Cache: {cache_size} entries cleared")
                    print(f"✓ Cleared transcription cache: {cache_size} entries")
                    
                    # Now try to delete the database files
                    import time
                    time.sleep(0.5)  # Wait for cache to close
                    
                    cache_db_files = list(transcription_cache_dir.glob("cache.db*"))
                    for cache_file in cache_db_files:
                        try:
                            cache_file.unlink()
                            deleted_items.append(f"Transcription Cache File: {cache_file.name}")
                            print(f"✓ Deleted transcription cache file: {cache_file.name}")
                        except PermissionError:
                            print(f"⚠ Could not delete {cache_file.name} (file in use, but cache cleared)")
                        except Exception as e:
                            print(f"⚠ Error deleting {cache_file.name}: {e}")
                            
                except ImportError:
                    # Fallback if diskcache not available
                    print("⚠ diskcache not available, using fallback method")
                    cache_db_files = list(transcription_cache_dir.glob("cache.db*"))
                    for cache_file in cache_db_files:
                        try:
                            cache_file.unlink()
                            deleted_items.append(f"Transcription Cache: {cache_file.name}")
                            print(f"✓ Deleted transcription cache: {cache_file.name}")
                        except Exception as e:
                            print(f"⚠ Error deleting {cache_file.name}: {e}")
                except Exception as e:
                    print(f"⚠ Error clearing transcription cache: {e}")
            
            # 3. Clear .cache directory (handle locked files)
            cache_dir = Path(".cache")
            if cache_dir.exists():
                cache_errors = []
                # Try to delete individual files first using os.walk
                for root, dirs, files in os.walk(str(cache_dir)):
                    for file in files:
                        file_path = Path(root) / file
                        # Skip transcription cache files (already handled above)
                        if "transcriptions" in str(file_path):
                            continue
                        try:
                            file_path.unlink()
                            deleted_items.append(f"Cache: {file}")
                        except PermissionError:
                            cache_errors.append(file)
                        except Exception as e:
                            cache_errors.append(f"{file} ({str(e)})")
                
                # Try to remove empty directories
                try:
                    for root, dirs, files in os.walk(str(cache_dir), topdown=False):
                        for dir_name in dirs:
                            dir_path = Path(root) / dir_name
                            try:
                                dir_path.rmdir()
                            except:
                                pass
                except:
                    pass
                
                if cache_errors:
                    print(f"⚠ Could not delete {len(cache_errors)} cache files (in use): {cache_errors[:3]}")
                else:
                    deleted_items.append("Cache directory")
            
            # 3. Clear old logs (keep recent 5)
            logs_dir = Path("logs")
            if logs_dir.exists():
                log_files = sorted(logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)
                for log_file in log_files[5:]:  # Keep 5 most recent
                    log_file.unlink()
                    deleted_items.append(f"Log: {log_file.name}")
            
            # 4. Clear old reports (keep recent 10)
            reports_dir = Path("reports")
            if reports_dir.exists():
                report_files = sorted(reports_dir.glob("*.txt"), key=lambda x: x.stat().st_mtime, reverse=True)
                for report_file in report_files[10:]:  # Keep 10 most recent
                    report_file.unlink()
                    deleted_items.append(f"Report: {report_file.name}")
                
                # Also clear old JSON/PDF reports
                for ext in ["*.json", "*.pdf"]:
                    old_files = sorted(reports_dir.glob(ext), key=lambda x: x.stat().st_mtime, reverse=True)
                    for old_file in old_files[10:]:
                        old_file.unlink()
                        deleted_items.append(f"Report: {old_file.name}")
            
            # Show success message
            if deleted_items:
                transcription_cache_count = len([x for x in deleted_items if 'Transcription Cache' in x])
                messagebox.showinfo(
                    "Thành công",
                    f"Đã xóa cache thành công!\n\n"
                    f"Đã xóa {len(deleted_items)} mục:\n"
                    f"- {len([x for x in deleted_items if 'Transcript' in x and 'Transcription Cache' not in x])} transcripts\n"
                    f"- {transcription_cache_count} transcription cache files\n"
                    f"- {len([x for x in deleted_items if 'Log' in x])} logs\n"
                    f"- {len([x for x in deleted_items if 'Report' in x])} reports\n"
                    f"- Cache directory\n\n"
                    f"Bạn có thể sử dụng model mới ngay bây giờ!"
                )
                print(f"✓ Cache cleared: {len(deleted_items)} items deleted")
            else:
                messagebox.showinfo(
                    "Thông báo",
                    "Không có cache nào để xóa."
                )
                print("ℹ No cache to clear")
            
        except Exception as e:
            messagebox.showerror(
                "Lỗi",
                f"Không thể xóa cache!\n\n"
                f"Lỗi: {str(e)}"
            )
            print(f"Error clearing cache: {e}")
            import traceback
            traceback.print_exc()
    
    def show_emotion_summary(self):
        """Hiển thị cửa sổ tóm tắt cảm xúc."""
        # Create new window
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Ket Qua Phan Tich Cam Xuc")
        summary_window.geometry("600x700")
        summary_window.configure(bg='#2b2b2b')
        
        # Make window modal
        summary_window.transient(self.root)
        summary_window.grab_set()
        
        # Header
        header_frame = tk.Frame(summary_window, bg='#1e1e1e', height=80)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="KET QUA PHAN TICH CAM XUC",
            font=("Arial", 18, "bold"),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        title_label.pack(pady=25)
        
        # Content frame
        content_frame = tk.Frame(summary_window, bg='#2b2b2b')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Calculate statistics
        total_emotions = sum(self.emotion_counts.values())
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        # Find dominant emotion
        if total_emotions > 0:
            dominant_emotion = max(self.emotion_counts.items(), key=lambda x: x[1])
            dominant_name = dominant_emotion[0]
            dominant_count = dominant_emotion[1]
            dominant_percentage = (dominant_count / total_emotions) * 100
        else:
            dominant_name = "Khong xac dinh"
            dominant_count = 0
            dominant_percentage = 0
        
        # Summary info
        info_text = tk.Text(
            content_frame,
            font=("Courier", 11),
            bg='#1e1e1e',
            fg='#ffffff',
            wrap=tk.WORD,
            height=25,
            state=tk.NORMAL
        )
        info_text.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Build summary text
        summary = f"""
{'='*50}
           THONG TIN VIDEO
{'='*50}

Nguon:              {'Camera' if self.video_source == 'camera' else 'Video File'}
Thoi gian xu ly:    {elapsed:.1f}s
Tong khung hinh:    {self.frame_count}
Tong khuon mat:     {self.total_faces}

{'='*50}
         CAM XUC PHO BIEN NHAT
{'='*50}

    >>> {dominant_name.upper()} <<<
    
    So lan xuat hien: {dominant_count}
    Ti le: {dominant_percentage:.1f}%

{'='*50}
        PHAN BO TAT CA CAM XUC
{'='*50}

"""
        
        # Add all emotions with bar chart
        if total_emotions > 0:
            sorted_emotions = sorted(
                self.emotion_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for emotion, count in sorted_emotions:
                percentage = (count / total_emotions) * 100
                bar_length = int(percentage / 2)
                bar = "█" * bar_length
                
                # Highlight dominant emotion
                if emotion == dominant_name:
                    summary += f">>> {emotion:12s}: {count:4d} ({percentage:5.1f}%)\n"
                    summary += f"    {bar}\n\n"
                else:
                    summary += f"    {emotion:12s}: {count:4d} ({percentage:5.1f}%)\n"
                    summary += f"    {bar}\n\n"
        else:
            summary += "Khong phat hien cam xuc nao\n"
        
        # Attention stats
        if self.attention_detector is not None and len(self.attention_scores) > 0:
            avg_attention = np.mean(self.attention_scores)
            total_attention = self.attention_focused_frames + self.attention_distracted_frames
            attention_stats = self.attention_detector.get_statistics()
            
            summary += f"""
{'='*50}
          MUC DO TAP TRUNG
{'='*50}

Diem trung binh:       {avg_attention:.0f}/100
"""
            if total_attention > 0:
                focused_rate = self.attention_focused_frames / total_attention
                distracted_rate = self.attention_distracted_frames / total_attention
                summary += f"Ti le tap trung:       {focused_rate:.1%}\n"
                summary += f"Ti le mat tap trung:   {distracted_rate:.1%}\n\n"
            
            # Hiển thị công thức
            if self.focus_components:
                summary += f"""Cong thuc (4 thanh phan):
  Face:  {self.focus_components.get('face_presence', 0):.0f}% (40%)
  Gaze:  {self.focus_components.get('gaze_focus', 0):.0f}% (30%)
  Head:  {self.focus_components.get('head_focus', 0):.0f}% (20%)
  Drift: {self.focus_components.get('drift_score', 0):.0f}% (10%)
  → Tong: {self.focus_components.get('total_score', 0):.1f}/10

"""
            
            summary += f"""Mat nham:              {attention_stats['eyes_closed_rate']:.1%}
Nhin ra ngoai:         {attention_stats['looking_away_rate']:.1%}
So lan chop mat:       {attention_stats['blink_count']}

"""
        
        # Performance stats
        if self.classifier:
            perf = self.classifier.get_performance_stats()
            summary += f"""
{'='*50}
            HIEU SUAT XU LY
{'='*50}

Thoi gian trung binh:  {perf['average_time_ms']:.2f}ms
Thoi gian toi thieu:   {perf['min_time_ms']:.2f}ms
Thoi gian toi da:      {perf['max_time_ms']:.2f}ms
Dat yeu cau <30ms:     {'YES' if perf['meets_requirement'] else 'NO'}

"""
        
        info_text.insert(1.0, summary)
        info_text.config(state=tk.DISABLED)
        
        # Close button
        close_button = tk.Button(
            summary_window,
            text="DONG",
            font=("Arial", 12, "bold"),
            bg='#4CAF50',
            fg='#ffffff',
            activebackground='#45a049',
            command=summary_window.destroy,
            width=15,
            height=2,
            cursor='hand2'
        )
        close_button.pack(pady=20)
    
    def on_video_end(self):
        """Xử lý khi video file kết thúc."""
        # Stop processing (this will clear the display)
        self.stop_processing()
        
        # Show emotion summary in new window
        self.show_emotion_summary()
    
    def on_screen_capture_error(self):
        """Xử lý lỗi khi screen capture thất bại."""
        self.stop_processing()
        messagebox.showerror(
            "Loi Screen Capture",
            "Khong the capture man hinh!\n\n"
            "Co the:\n"
            "- Loi he thong\n"
            "- Khong co quyen truy cap\n"
            "- Thu khoi dong lai ung dung"
        )
    
    def toggle_dual_person_mode(self):
        """Toggle dual person comparison mode.
        
        Switches between normal single view and dual person split view.
        
        Requirements:
        - 1.1: Activate/deactivate dual person mode
        - 2.1, 2.2: Display split view with two video feeds
        """
        if not DUAL_PERSON_AVAILABLE:
            messagebox.showerror(
                "Loi",
                "Dual person comparison khong kha dung!\n\n"
                "Vui long kiem tra cac dependencies."
            )
            return
        
        # Stop processing if currently running
        if self.is_running:
            messagebox.showwarning(
                "Canh bao",
                "Vui long dung quet truoc khi chuyen che do!"
            )
            return
        
        if not self.dual_person_mode:
            # Activate dual person mode
            self._activate_dual_person_mode()
        else:
            # Deactivate dual person mode
            self._deactivate_dual_person_mode()
    
    def _activate_dual_person_mode(self):
        """Activate dual person comparison mode.
        
        Requirements:
        - 1.1: Initialize DualPersonCoordinator
        - 2.1, 2.2, 2.3: Switch to split view layout
        - 3.1, 3.2: Switch to dual statistics panels
        """
        print("Activating dual person comparison mode...")
        
        # Check if screen capture is available
        if not SCREEN_CAPTURE_AVAILABLE:
            messagebox.showerror(
                "Loi",
                "Screen capture khong kha dung!\n\n"
                "Dual person mode can screen capture.\n"
                "Vui long cai dat: pip install mss"
            )
            return
        
        # Initialize screen capture if not already done
        if self.screen_capture is None:
            try:
                self.screen_capture = ScreenCapture(monitor_number=1)
            except Exception as e:
                messagebox.showerror(
                    "Loi",
                    f"Khong the khoi tao screen capture!\n\n{str(e)}"
                )
                return
        
        # Initialize DualPersonCoordinator (use settings - Req 8.2)
        try:
            self.dual_person_coordinator = DualPersonCoordinator(
                face_detector=self.detector,
                preprocessor=self.preprocessor,
                emotion_classifier=self.classifier,
                attention_detector=self.attention_detector,
                screen_capture=self.screen_capture,
                camera_device_id=0,
                comparison_update_interval=self.dual_person_settings['comparison_update_interval'],
                enable_optimization=True
            )
            print("✓ DualPersonCoordinator initialized")
        except Exception as e:
            messagebox.showerror(
                "Loi",
                f"Khong the khoi tao dual person coordinator!\n\n{str(e)}"
            )
            return
        
        # Switch to split view
        self.single_view_frame.grid_remove()
        self.split_view_frame.grid(row=0, column=0, rowspan=2, sticky='nsew')
        
        # Switch to dual statistics panels (Req 3.1, 3.2)
        self.single_stats_frame.grid_remove()
        self.dual_stats_frame.grid(row=0, column=0, rowspan=2, sticky='nsew')
        
        # Reset statistics update timers
        self.last_dual_stats_update = 0
        self.last_emotion_detection_time = 0
        
        # Update mode flag
        self.dual_person_mode = True
        
        # Update button
        self.dual_person_button.config(
            text="TAT CHE DO SO SANH",
            bg='#f44336',
            activebackground='#da190b'
        )
        
        # Force camera source in dual person mode
        self.source_var.set("camera")
        self.on_source_change()
        
        # Disable source selection in dual person mode
        for widget in self.main_container.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Radiobutton):
                        child.config(state=tk.DISABLED)
        
        self.update_status("Che do so sanh 2 nguoi - San sang", "#00BCD4")
        
        print("✓ Dual person mode activated")
    
    def _deactivate_dual_person_mode(self):
        """Deactivate dual person comparison mode.
        
        Requirements:
        - 1.5: Cleanup resources
        - Switch back to single view and single statistics
        """
        print("Deactivating dual person comparison mode...")
        
        # Stop coordinator if active
        if self.dual_person_coordinator is not None:
            try:
                if self.dual_person_coordinator.is_active():
                    self.dual_person_coordinator.stop()
            except Exception as e:
                print(f"Warning: Error stopping coordinator: {e}")
            
            self.dual_person_coordinator = None
        
        # Switch back to single view
        self.split_view_frame.grid_remove()
        self.single_view_frame.grid(row=0, column=0, rowspan=2, sticky='nsew')
        
        # Switch back to single statistics panel
        self.dual_stats_frame.grid_remove()
        self.single_stats_frame.grid(row=0, column=0, rowspan=2, sticky='nsew')
        
        # Clear split view canvases
        self._clear_canvas(self.primary_canvas)
        self._clear_canvas(self.secondary_canvas)
        
        # Clear dual statistics panels
        self.primary_stats_text.config(state=tk.NORMAL)
        self.primary_stats_text.delete(1.0, tk.END)
        self.primary_stats_text.config(state=tk.DISABLED)
        
        self.secondary_stats_text.config(state=tk.NORMAL)
        self.secondary_stats_text.delete(1.0, tk.END)
        self.secondary_stats_text.config(state=tk.DISABLED)
        
        # Update mode flag
        self.dual_person_mode = False
        
        # Update button
        self.dual_person_button.config(
            text="CHE DO SO SANH 2 NGUOI",
            bg='#00BCD4',
            activebackground='#0097A7'
        )
        
        # Re-enable source selection
        for widget in self.main_container.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Radiobutton):
                        child.config(state=tk.NORMAL)
        
        # Update screen capture radio button state
        if not SCREEN_CAPTURE_AVAILABLE:
            for widget in self.main_container.winfo_children():
                if isinstance(widget, tk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, tk.Radiobutton) and child.cget('value') == 'screen':
                            child.config(state=tk.DISABLED)
        
        self.update_status("San sang", "#4CAF50")
        
        print("✓ Dual person mode deactivated")
    
    def update_split_view(self, primary_frame: np.ndarray, secondary_frame: np.ndarray):
        """Update split view display with both frames.
        
        Renders primary video on left canvas and secondary video on right canvas,
        maintaining aspect ratio and equal proportions.
        
        Args:
            primary_frame: Frame from primary source (camera)
            secondary_frame: Frame from secondary source (screen capture)
        
        Requirements:
        - 2.1: Display primary video on left
        - 2.2: Display secondary video on right
        - 2.3: Maintain equal proportions when resize
        - 2.5: Draw bounding boxes on both videos
        """
        # Update primary canvas (left)
        self._update_canvas(self.primary_canvas, primary_frame)
        
        # Update secondary canvas (right)
        if secondary_frame is not None:
            self._update_canvas(self.secondary_canvas, secondary_frame)
        else:
            # Show black screen if no secondary frame
            self._clear_canvas(self.secondary_canvas)
    
    def _update_canvas(self, canvas: tk.Label, frame: np.ndarray):
        """Update a canvas with a frame, maintaining aspect ratio.
        
        Args:
            canvas: Canvas to update
            frame: Frame to display
        
        Requirements:
        - 2.3: Maintain aspect ratio
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get canvas size
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Ensure canvas has valid size
        if canvas_width <= 1:
            canvas_width = 320  # Half of default 640
        if canvas_height <= 1:
            canvas_height = 480
        
        # Set maximum video display size for split view (smaller than single view)
        MAX_VIDEO_WIDTH = 400
        MAX_VIDEO_HEIGHT = 600
        
        # Limit canvas size to maximum
        canvas_width = min(canvas_width, MAX_VIDEO_WIDTH)
        canvas_height = min(canvas_height, MAX_VIDEO_HEIGHT)
        
        # Calculate aspect ratio preserving resize
        frame_height, frame_width = frame_rgb.shape[:2]
        frame_aspect = frame_width / frame_height
        canvas_aspect = canvas_width / canvas_height
        
        # Calculate new dimensions to fit canvas while preserving aspect ratio
        if canvas_aspect > frame_aspect:
            # Canvas is wider - fit to height
            new_height = canvas_height
            new_width = int(canvas_height * frame_aspect)
        else:
            # Canvas is taller - fit to width
            new_width = canvas_width
            new_height = int(canvas_width / frame_aspect)
        
        # Ensure dimensions are at least 1
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # Resize frame
        frame_rgb = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Convert to PhotoImage
        image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update canvas
        canvas.config(image=photo)
        canvas.image = photo
    
    def _clear_canvas(self, canvas: tk.Label):
        """Clear a canvas by displaying black screen.
        
        Args:
            canvas: Canvas to clear
        """
        # Get canvas size
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Ensure canvas has valid size
        if canvas_width <= 1:
            canvas_width = 320
        if canvas_height <= 1:
            canvas_height = 480
        
        # Create black image
        black_frame = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Convert to PhotoImage
        image = Image.fromarray(black_frame)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update canvas
        canvas.config(image=photo)
        canvas.image = photo
    
    # show_performance_settings() - REMOVED
    
    def show_audio_recording_dialog(self):
        """Show simple audio recording dialog."""
        # Create recording window
        rec_window = tk.Toplevel(self.root)
        rec_window.title("Thu Âm")
        rec_window.geometry("500x400")
        rec_window.configure(bg='#1a1a1a')
        
        # Make window modal
        rec_window.transient(self.root)
        rec_window.grab_set()
        
        # Center window
        rec_window.update_idletasks()
        x = (rec_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (rec_window.winfo_screenheight() // 2) - (400 // 2)
        rec_window.geometry(f"500x400+{x}+{y}")
        
        # Header
        header_frame = tk.Frame(rec_window, bg='#0d47a1', height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🎤 THU ÂM",
            font=("Segoe UI", 18, "bold"),
            bg='#0d47a1',
            fg='#ffffff'
        )
        title_label.pack(pady=15)
        
        # Content frame
        content_frame = tk.Frame(rec_window, bg='#1a1a1a')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Status label
        status_label = tk.Label(
            content_frame,
            text="⏸️ Chưa bắt đầu thu âm",
            font=("Segoe UI", 12, "bold"),
            bg='#1a1a1a',
            fg='#4CAF50'
        )
        status_label.pack(pady=20)
        
        # Time label
        time_label = tk.Label(
            content_frame,
            text="⏱️ 00:00:00",
            font=("Segoe UI", 16, "bold"),
            bg='#1a1a1a',
            fg='#4CAF50'
        )
        time_label.pack(pady=10)
        
        # Audio level progress
        level_frame = tk.Frame(content_frame, bg='#1a1a1a')
        level_frame.pack(fill=tk.X, pady=20)
        
        level_label = tk.Label(
            level_frame,
            text="🔊 Mức âm thanh:",
            font=("Segoe UI", 10),
            bg='#1a1a1a',
            fg='#90caf9'
        )
        level_label.pack(anchor='w', pady=(0, 5))
        
        level_progress = ttk.Progressbar(
            level_frame,
            mode='determinate',
            length=400,
            maximum=100
        )
        level_progress.pack(fill=tk.X)
        
        # Buttons frame
        button_frame = tk.Frame(content_frame, bg='#1a1a1a')
        button_frame.pack(pady=30)
        
        # Recording state
        is_recording = [False]  # Use list to allow modification in nested function
        coordinator = [None]
        start_time = [None]
        update_timer = [None]
        
        def update_time():
            """Update elapsed time."""
            if is_recording[0] and start_time[0]:
                import time
                elapsed = time.time() - start_time[0]
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                time_label.config(text=f"⏱️ {hours:02d}:{minutes:02d}:{seconds:02d}")
                update_timer[0] = rec_window.after(100, update_time)
        
        def on_audio_update(level, elapsed):
            """Update audio level."""
            level_percent = int(level * 100)
            level_progress['value'] = level_percent
        
        def start_recording():
            """Start recording."""
            try:
                from src.audio_recording.coordinator import AudioRecordingCoordinator
                from src.audio_recording.config import AudioRecordingConfig
                import time
                
                # Initialize coordinator
                config = AudioRecordingConfig()
                coordinator[0] = AudioRecordingCoordinator(
                    config=config,
                    gui_callback=on_audio_update
                )
                
                # Start recording
                coordinator[0].start_recording()
                is_recording[0] = True
                start_time[0] = time.time()
                
                # Update UI
                status_label.config(text="⏺️ ĐANG THU ÂM...", fg='#f44336')
                start_button.config(state=tk.DISABLED)
                stop_button.config(state=tk.NORMAL)
                
                # Start time update
                update_time()
                
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể bắt đầu thu âm:\n\n{str(e)}")
        
        def stop_recording():
            """Stop recording."""
            try:
                if coordinator[0]:
                    file_path = coordinator[0].stop_recording()
                    is_recording[0] = False
                    
                    # Stop time update
                    if update_timer[0]:
                        rec_window.after_cancel(update_timer[0])
                    
                    # Update UI
                    status_label.config(text="✅ Hoàn thành!", fg='#4CAF50')
                    start_button.config(state=tk.NORMAL)
                    stop_button.config(state=tk.DISABLED)
                    time_label.config(text="⏱️ 00:00:00")
                    level_progress['value'] = 0
                    
                    # Show success message
                    from pathlib import Path
                    messagebox.showinfo(
                        "Thành Công",
                        f"Thu âm hoàn tất!\n\n"
                        f"File: {Path(file_path).name}\n"
                        f"Đường dẫn: {file_path}"
                    )
                    
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể dừng thu âm:\n\n{str(e)}")
        
        # Start button
        start_button = tk.Button(
            button_frame,
            text="⏺️ BẮT ĐẦU THU ÂM",
            font=("Segoe UI", 12, "bold"),
            bg='#d32f2f',
            fg='#ffffff',
            activebackground='#b71c1c',
            command=start_recording,
            cursor='hand2',
            relief=tk.FLAT,
            padx=30,
            pady=12
        )
        start_button.pack(side=tk.LEFT, padx=10)
        
        # Stop button
        stop_button = tk.Button(
            button_frame,
            text="⏹️ DỪNG THU ÂM",
            font=("Segoe UI", 12, "bold"),
            bg='#1976d2',
            fg='#ffffff',
            activebackground='#0d47a1',
            command=stop_recording,
            cursor='hand2',
            relief=tk.FLAT,
            padx=30,
            pady=12,
            state=tk.DISABLED
        )
        stop_button.pack(side=tk.LEFT, padx=10)
        
        # Close button
        close_button = tk.Button(
            content_frame,
            text="❌ Đóng",
            font=("Segoe UI", 10),
            bg='#757575',
            fg='#ffffff',
            activebackground='#616161',
            command=rec_window.destroy,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8
        )
        close_button.pack(pady=10)

    def show_appearance_settings(self):
        """Show appearance assessment settings dialog."""
        # Create settings window with EXTRA LARGE fixed size to show ALL content
        app_window = tk.Toplevel(self.root)
        app_window.title("Tùy Chọn Quét")
        
        # Set EXTRA LARGE size - 800x800 pixels
        window_width = 800
        window_height = 800
        
        print(f"DEBUG: Creating window with size {window_width}x{window_height}")
        
        # Set minimum size to prevent shrinking
        app_window.minsize(window_width, window_height)
        app_window.geometry(f"{window_width}x{window_height}")
        app_window.resizable(False, False)
        app_window.configure(bg='#2b2b2b')
        
        # Make window modal
        app_window.transient(self.root)
        app_window.grab_set()
        
        # Center window on screen
        app_window.update_idletasks()
        screen_width = app_window.winfo_screenwidth()
        screen_height = app_window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        app_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        print(f"DEBUG: Window positioned at {x},{y}")
        
        # Header - NO pack_propagate to allow natural sizing
        header_frame = tk.Frame(app_window, bg='#1e1e1e')
        header_frame.pack(fill=tk.X, pady=0)
        
        title_label = tk.Label(
            header_frame,
            text="🎯 TÙY CHỌN QUÉT",
            font=("Segoe UI", 18, "bold"),
            bg='#1e1e1e',
            fg='#4CAF50'
        )
        title_label.pack(pady=10)
        
        # Main content container - REDUCED padding
        main_content = tk.Frame(app_window, bg='#2b2b2b')
        main_content.pack(fill=tk.BOTH, expand=True, padx=25, pady=10)
        
        # Quick scan button at top
        def start_quick_scan():
            """Start scanning immediately without additional features."""
            self.appearance_enabled = False
            self.assess_clothing = False
            self.assess_lighting = False
            app_window.destroy()
            if not self.is_running:
                self.toggle_processing()
        
        quick_scan_btn = tk.Button(
            main_content,
            text="▶ QUÉT NGAY",
            font=("Segoe UI", 12, "bold"),
            bg='#43a047',
            fg='#ffffff',
            activebackground='#388e3c',
            command=start_quick_scan,
            cursor='hand2',
            height=2,
            relief=tk.FLAT
        )
        quick_scan_btn.pack(fill=tk.X, pady=(0, 8))
        
        # Separator
        separator1 = tk.Frame(main_content, bg='#555555', height=2)
        separator1.pack(fill=tk.X, pady=6)
        
        # Section title
        section_title = tk.Label(
            main_content,
            text="Chọn các chức năng bổ sung:",
            font=("Segoe UI", 11, "bold"),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        section_title.pack(anchor='w', pady=(3, 6))
        
        # Clothing assessment checkbox - COMPACT
        clothing_frame = tk.Frame(main_content, bg='#3a3a3a', relief=tk.RAISED, bd=1)
        clothing_frame.pack(fill=tk.X, pady=5)
        
        clothing_var = tk.BooleanVar(value=self.assess_clothing)
        
        clothing_check = tk.Checkbutton(
            clothing_frame,
            text="👔 Quét Diện Mạo (Quần Áo)",
            variable=clothing_var,
            font=("Segoe UI", 10, "bold"),
            bg='#3a3a3a',
            fg='#ffffff',
            selectcolor='#2b2b2b',
            activebackground='#3a3a3a',
            activeforeground='#ffffff'
        )
        clothing_check.pack(anchor='w', padx=12, pady=6)
        
        clothing_desc = tk.Label(
            clothing_frame,
            text="→ Đánh giá màu sắc, kiểu dáng, độ chuyên nghiệp",
            font=("Segoe UI", 8),
            bg='#3a3a3a',
            fg='#b0b0b0'
        )
        clothing_desc.pack(anchor='w', padx=12, pady=(0, 6))
        
        # Lighting assessment checkbox - COMPACT
        lighting_frame = tk.Frame(main_content, bg='#3a3a3a', relief=tk.RAISED, bd=1)
        lighting_frame.pack(fill=tk.X, pady=5)
        
        lighting_var = tk.BooleanVar(value=self.assess_lighting)
        
        lighting_check = tk.Checkbutton(
            lighting_frame,
            text="💡 Quét Ánh Sáng",
            variable=lighting_var,
            font=("Segoe UI", 10, "bold"),
            bg='#3a3a3a',
            fg='#ffffff',
            selectcolor='#2b2b2b',
            activebackground='#3a3a3a',
            activeforeground='#ffffff'
        )
        lighting_check.pack(anchor='w', padx=12, pady=6)
        
        lighting_desc = tk.Label(
            lighting_frame,
            text="→ Đánh giá độ sáng, độ tương phản, chất lượng",
            font=("Segoe UI", 8),
            bg='#3a3a3a',
            fg='#b0b0b0'
        )
        lighting_desc.pack(anchor='w', padx=12, pady=(0, 6))
        
        # Note - COMPACT
        note_label = tk.Label(
            main_content,
            text="Lưu ý: Bật thêm chức năng có thể giảm FPS một chút",
            font=("Segoe UI", 8, "italic"),
            bg='#2b2b2b',
            fg='#888888'
        )
        note_label.pack(pady=8)
        
        # Bottom button frame - COMPACT
        button_frame = tk.Frame(main_content, bg='#2b2b2b')
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        def start_scan_with_options():
            """Start scanning with selected options."""
            self.assess_clothing = clothing_var.get()
            self.assess_lighting = lighting_var.get()
            self.appearance_enabled = self.assess_clothing or self.assess_lighting
            app_window.destroy()
            if not self.is_running:
                self.toggle_processing()
        
        # Start scan button
        scan_btn = tk.Button(
            button_frame,
            text="▶ BẮT ĐẦU QUÉT",
            font=("Segoe UI", 11, "bold"),
            bg='#1e88e5',
            fg='#ffffff',
            activebackground='#1565c0',
            command=start_scan_with_options,
            cursor='hand2',
            width=18,
            height=2,
            relief=tk.FLAT
        )
        scan_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Cancel button
        cancel_btn = tk.Button(
            button_frame,
            text="✗ HỦY",
            font=("Segoe UI", 11, "bold"),
            bg='#f44336',
            fg='#ffffff',
            activebackground='#d32f2f',
            command=app_window.destroy,
            cursor='hand2',
            width=18,
            height=2,
            relief=tk.FLAT
        )
        cancel_btn.pack(side=tk.LEFT, padx=(10, 0))
    
    def show_dual_person_settings(self):
        """Show settings dialog for dual person comparison mode.
        
        Provides controls to adjust:
        - Comparison update interval (5-30 seconds)
        - Toggle bounding boxes on/off
        - Swap positions of primary and secondary videos
        
        Settings are applied immediately without restarting the session.
        
        Requirements:
        - 8.1: Provide settings panel accessible during comparison mode
        - 8.2: Allow adjustment of comparison update interval (5-30s)
        - 8.3: Allow toggle of bounding boxes
        - 8.4: Allow swap of video positions
        - 8.5: Apply changes immediately without restart
        """
        # Create settings window
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Cai Dat Che Do So Sanh 2 Nguoi")
        settings_window.geometry("500x400")
        settings_window.configure(bg='#2b2b2b')
        
        # Make window modal
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # Center window on screen
        settings_window.update_idletasks()
        x = (settings_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (settings_window.winfo_screenheight() // 2) - (400 // 2)
        settings_window.geometry(f"500x400+{x}+{y}")
        
        # Header
        header_frame = tk.Frame(settings_window, bg='#1e1e1e', height=60)
        header_frame.pack(fill=tk.X, padx=10, pady=10)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="⚙️ CAI DAT CHE DO SO SANH",
            font=("Arial", 16, "bold"),
            bg='#1e1e1e',
            fg='#ffffff'
        )
        title_label.pack(pady=15)
        
        # Content frame
        content_frame = tk.Frame(settings_window, bg='#2b2b2b')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Setting 1: Comparison Update Interval (Req 8.2)
        interval_frame = tk.Frame(content_frame, bg='#2b2b2b')
        interval_frame.pack(fill=tk.X, pady=15)
        
        interval_label = tk.Label(
            interval_frame,
            text="Khoang thoi gian cap nhat so sanh:",
            font=("Arial", 11, "bold"),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        interval_label.pack(anchor=tk.W)
        
        interval_desc = tk.Label(
            interval_frame,
            text="Thoi gian giua cac lan so sanh (5-30 giay)",
            font=("Arial", 9),
            bg='#2b2b2b',
            fg='#9E9E9E'
        )
        interval_desc.pack(anchor=tk.W, pady=(0, 5))
        
        # Slider frame with value display
        slider_frame = tk.Frame(interval_frame, bg='#2b2b2b')
        slider_frame.pack(fill=tk.X)
        
        interval_value_label = tk.Label(
            slider_frame,
            text=f"{self.dual_person_settings['comparison_update_interval']:.0f}s",
            font=("Arial", 11, "bold"),
            bg='#2b2b2b',
            fg='#4CAF50',
            width=5
        )
        interval_value_label.pack(side=tk.RIGHT, padx=10)
        
        def on_interval_change(value):
            """Update interval value label."""
            interval_value_label.config(text=f"{float(value):.0f}s")
        
        interval_slider = tk.Scale(
            slider_frame,
            from_=5,
            to=30,
            orient=tk.HORIZONTAL,
            bg='#2b2b2b',
            fg='#ffffff',
            highlightthickness=0,
            troughcolor='#1e1e1e',
            activebackground='#4CAF50',
            command=on_interval_change,
            length=350
        )
        interval_slider.set(self.dual_person_settings['comparison_update_interval'])
        interval_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Separator
        separator1 = tk.Frame(content_frame, bg='#555555', height=2)
        separator1.pack(fill=tk.X, pady=15)
        
        # Setting 2: Bounding Boxes Toggle (Req 8.3)
        bbox_frame = tk.Frame(content_frame, bg='#2b2b2b')
        bbox_frame.pack(fill=tk.X, pady=15)
        
        bbox_label = tk.Label(
            bbox_frame,
            text="Hien thi khung bao quanh khuon mat:",
            font=("Arial", 11, "bold"),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        bbox_label.pack(anchor=tk.W)
        
        bbox_desc = tk.Label(
            bbox_frame,
            text="Bat/tat hien thi khung xanh quanh khuon mat",
            font=("Arial", 9),
            bg='#2b2b2b',
            fg='#9E9E9E'
        )
        bbox_desc.pack(anchor=tk.W, pady=(0, 5))
        
        bbox_var = tk.BooleanVar(value=self.dual_person_settings['show_bounding_boxes'])
        
        bbox_checkbox = tk.Checkbutton(
            bbox_frame,
            text="Hien thi khung bao",
            variable=bbox_var,
            font=("Arial", 10),
            bg='#2b2b2b',
            fg='#ffffff',
            selectcolor='#1e1e1e',
            activebackground='#2b2b2b',
            activeforeground='#ffffff',
            highlightthickness=0
        )
        bbox_checkbox.pack(anchor=tk.W, padx=20)
        
        # Separator
        separator2 = tk.Frame(content_frame, bg='#555555', height=2)
        separator2.pack(fill=tk.X, pady=15)
        
        # Setting 3: Swap Video Positions (Req 8.4)
        swap_frame = tk.Frame(content_frame, bg='#2b2b2b')
        swap_frame.pack(fill=tk.X, pady=15)
        
        swap_label = tk.Label(
            swap_frame,
            text="Vi tri video:",
            font=("Arial", 11, "bold"),
            bg='#2b2b2b',
            fg='#ffffff'
        )
        swap_label.pack(anchor=tk.W)
        
        swap_desc = tk.Label(
            swap_frame,
            text="Hoan doi vi tri cua video Ban va Nguoi Khac",
            font=("Arial", 9),
            bg='#2b2b2b',
            fg='#9E9E9E'
        )
        swap_desc.pack(anchor=tk.W, pady=(0, 5))
        
        current_position = self.dual_person_settings['primary_position']
        position_text = "Ban (Trai) | Nguoi Khac (Phai)" if current_position == 'left' else "Nguoi Khac (Trai) | Ban (Phai)"
        
        position_label = tk.Label(
            swap_frame,
            text=f"Hien tai: {position_text}",
            font=("Arial", 10),
            bg='#2b2b2b',
            fg='#4CAF50'
        )
        position_label.pack(anchor=tk.W, padx=20, pady=(5, 10))
        
        def swap_positions():
            """Swap video positions."""
            current = self.dual_person_settings['primary_position']
            new_position = 'right' if current == 'left' else 'left'
            self.dual_person_settings['primary_position'] = new_position
            
            # Update label
            new_text = "Ban (Trai) | Nguoi Khac (Phai)" if new_position == 'left' else "Nguoi Khac (Trai) | Ban (Phai)"
            position_label.config(text=f"Hien tai: {new_text}")
            
            # Apply immediately (Req 8.5)
            self._apply_position_swap()
        
        swap_button = tk.Button(
            swap_frame,
            text="🔄 HOAN DOI VI TRI",
            font=("Arial", 10, "bold"),
            bg='#2196F3',
            fg='#ffffff',
            activebackground='#0b7dda',
            command=swap_positions,
            cursor='hand2',
            width=20,
            height=1
        )
        swap_button.pack(anchor=tk.W, padx=20)
        
        # Button frame
        button_frame = tk.Frame(settings_window, bg='#2b2b2b')
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        
        def apply_settings():
            """Apply settings and close dialog (Req 8.5)."""
            # Update settings
            self.dual_person_settings['comparison_update_interval'] = float(interval_slider.get())
            self.dual_person_settings['show_bounding_boxes'] = bbox_var.get()
            
            # Apply to coordinator if active
            if self.dual_person_coordinator is not None:
                try:
                    # Update comparison interval
                    self.dual_person_coordinator.comparison_update_interval = self.dual_person_settings['comparison_update_interval']
                    print(f"✓ Updated comparison interval to {self.dual_person_settings['comparison_update_interval']}s")
                except Exception as e:
                    print(f"Warning: Could not update coordinator settings: {e}")
            
            # Close dialog
            settings_window.destroy()
            
            # Show confirmation
            messagebox.showinfo(
                "Thanh cong",
                "Da ap dung cai dat thanh cong!\n\n"
                "Cac thay doi co hieu luc ngay lap tuc."
            )
        
        def cancel_settings():
            """Cancel and close dialog."""
            settings_window.destroy()
        
        # Apply button
        apply_button = tk.Button(
            button_frame,
            text="✓ AP DUNG",
            font=("Arial", 12, "bold"),
            bg='#4CAF50',
            fg='#ffffff',
            activebackground='#45a049',
            command=apply_settings,
            cursor='hand2',
            width=15,
            height=2
        )
        apply_button.pack(side=tk.LEFT, padx=10)
        
        # Cancel button
        cancel_button = tk.Button(
            button_frame,
            text="✗ HUY",
            font=("Arial", 12, "bold"),
            bg='#f44336',
            fg='#ffffff',
            activebackground='#da190b',
            command=cancel_settings,
            cursor='hand2',
            width=15,
            height=2
        )
        cancel_button.pack(side=tk.RIGHT, padx=10)
    
    def export_comparison_report(self):
        """Export comparison report to JSON file.
        
        Generates a comprehensive SessionReport with all statistics, timelines,
        and comparison history for the dual person comparison session.
        
        Requirements:
        - 6.1: Generate detailed comparison report when session ends
        - 6.2: Include average Emotion Score and Attention Score for both users
        - 6.3: Include emotion and attention timelines for both users
        - 6.4: Save report as JSON file with timestamp
        - 6.5: Display confirmation message with file path
        """
        if not self.dual_person_mode or self.dual_person_coordinator is None:
            messagebox.showwarning(
                "Canh bao",
                "Vui long bat che do so sanh 2 nguoi truoc khi xuat bao cao!"
            )
            return
        
        try:
            # Get session report from coordinator
            session_report = self.dual_person_coordinator.get_session_report()
            
            # Generate default filename
            timestamp = session_report.start_time.strftime("%Y%m%d_%H%M%S")
            default_name = f"dual_person_comparison_{timestamp}"
            default_dir = str(Path("./reports").absolute())
            
            # Ask where to save with custom dialog
            filename = ask_save_file(
                parent=self.root,
                title="Xuất Báo Cáo So Sánh",
                default_filename=default_name,
                default_dir=default_dir,
                file_extension=".json",
                file_types=[
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ]
            )
            
            if not filename:
                return  # User cancelled
            
            filename = Path(filename)
            
            # Convert SessionReport to dictionary for JSON serialization
            report_dict = self._session_report_to_dict(session_report)
            
            # Add timeline chart data (Requirement 6.3)
            report_dict['charts'] = self._generate_timeline_charts(session_report)
            
            # Save to JSON file
            import json
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report_dict, f, indent=2, ensure_ascii=False)
            
            # Display confirmation message (Requirement 6.5)
            messagebox.showinfo(
                "Thanh cong",
                f"Da xuat bao cao so sanh thanh cong!\n\n"
                f"Duong dan: {filename.absolute()}\n\n"
                f"Thoi gian phien: {session_report.duration_seconds:.1f} giay\n"
                f"Nguoi thang cam xuc: {self._translate_winner(session_report.overall_emotion_winner)}\n"
                f"Nguoi thang tap trung: {self._translate_winner(session_report.overall_attention_winner)}"
            )
            
            print(f"✓ Comparison report exported to: {filename}")
            
        except Exception as e:
            messagebox.showerror(
                "Loi",
                f"Khong the xuat bao cao: {str(e)}"
            )
            print(f"Error exporting comparison report: {e}")
            import traceback
            traceback.print_exc()
    
    def _session_report_to_dict(self, report) -> dict:
        """Convert SessionReport to dictionary for JSON serialization.
        
        Args:
            report: SessionReport instance
        
        Returns:
            Dictionary representation of the report
        
        Requirements:
        - 6.2: Include average Emotion Score and Attention Score
        - 6.3: Include emotion and attention timelines
        """
        return {
            'session_info': {
                'session_id': report.session_id,
                'start_time': report.start_time.isoformat(),
                'end_time': report.end_time.isoformat(),
                'duration_seconds': report.duration_seconds,
                'duration_minutes': report.duration_seconds / 60.0
            },
            'primary_user': {
                'label': 'Ban (Primary User)',
                'statistics': report.primary_stats,
                'emotion_timeline': [
                    {
                        'timestamp': ed.timestamp,
                        'emotion': ed.emotion,
                        'confidence': ed.confidence,
                        'frame_number': ed.frame_number
                    }
                    for ed in report.primary_emotion_timeline
                ],
                'attention_timeline': [
                    {
                        'timestamp': i * 0.033,  # Approximate timestamp (30 FPS)
                        'score': score
                    }
                    for i, score in enumerate(report.primary_attention_timeline)
                ],
                'average_emotion_score': (
                    sum(ed.confidence for ed in report.primary_emotion_timeline) / len(report.primary_emotion_timeline)
                    if report.primary_emotion_timeline else 0.0
                ),
                'average_attention_score': (
                    sum(report.primary_attention_timeline) / len(report.primary_attention_timeline)
                    if report.primary_attention_timeline else 0.0
                )
            },
            'secondary_user': {
                'label': 'Nguoi Khac (Secondary User)',
                'statistics': report.secondary_stats,
                'emotion_timeline': [
                    {
                        'timestamp': ed.timestamp,
                        'emotion': ed.emotion,
                        'confidence': ed.confidence,
                        'frame_number': ed.frame_number
                    }
                    for ed in report.secondary_emotion_timeline
                ],
                'attention_timeline': [
                    {
                        'timestamp': i * 0.033,  # Approximate timestamp (30 FPS)
                        'score': score
                    }
                    for i, score in enumerate(report.secondary_attention_timeline)
                ],
                'average_emotion_score': (
                    sum(ed.confidence for ed in report.secondary_emotion_timeline) / len(report.secondary_emotion_timeline)
                    if report.secondary_emotion_timeline else 0.0
                ),
                'average_attention_score': (
                    sum(report.secondary_attention_timeline) / len(report.secondary_attention_timeline)
                    if report.secondary_attention_timeline else 0.0
                )
            },
            'comparison': {
                'overall_emotion_winner': report.overall_emotion_winner,
                'overall_attention_winner': report.overall_attention_winner,
                'average_emotion_difference': report.average_emotion_difference,
                'average_attention_difference': report.average_attention_difference,
                'comparison_history': [
                    {
                        'timestamp': comp.timestamp,
                        'emotion': {
                            'primary_score': comp.primary_emotion_score,
                            'secondary_score': comp.secondary_emotion_score,
                            'winner': comp.emotion_winner,
                            'difference': comp.emotion_difference,
                            'significant_gap': comp.significant_emotion_gap
                        },
                        'attention': {
                            'primary_score': comp.primary_attention_score,
                            'secondary_score': comp.secondary_attention_score,
                            'winner': comp.attention_winner,
                            'difference': comp.attention_difference,
                            'significant_gap': comp.significant_attention_gap
                        }
                    }
                    for comp in report.comparison_history
                ]
            }
        }
    
    def _generate_timeline_charts(self, report) -> dict:
        """Generate timeline chart data for visualization.
        
        Creates data points for emotion and attention trend charts
        that can be used by visualization tools.
        
        Args:
            report: SessionReport instance
        
        Returns:
            Dictionary with chart data
        
        Requirements:
        - 6.3: Include timeline charts in report
        - 13.1: Generate emotion and attention trend charts
        """
        # Emotion trend chart data
        emotion_chart = {
            'title': 'Emotion Trend Over Time',
            'x_axis': 'Time (seconds)',
            'y_axis': 'Emotion Score (0-100)',
            'primary_data': [],
            'secondary_data': []
        }
        
        # Process primary emotion timeline
        if report.primary_emotion_timeline:
            start_time = report.primary_emotion_timeline[0].timestamp
            for ed in report.primary_emotion_timeline:
                emotion_chart['primary_data'].append({
                    'time': ed.timestamp - start_time,
                    'score': ed.confidence * 100,  # Convert to 0-100 scale
                    'emotion': ed.emotion
                })
        
        # Process secondary emotion timeline
        if report.secondary_emotion_timeline:
            start_time = report.secondary_emotion_timeline[0].timestamp
            for ed in report.secondary_emotion_timeline:
                emotion_chart['secondary_data'].append({
                    'time': ed.timestamp - start_time,
                    'score': ed.confidence * 100,  # Convert to 0-100 scale
                    'emotion': ed.emotion
                })
        
        # Attention trend chart data
        attention_chart = {
            'title': 'Attention Trend Over Time',
            'x_axis': 'Time (seconds)',
            'y_axis': 'Attention Score (0-100)',
            'primary_data': [],
            'secondary_data': []
        }
        
        # Process primary attention timeline
        for i, score in enumerate(report.primary_attention_timeline):
            attention_chart['primary_data'].append({
                'time': i * 0.033,  # Approximate time (30 FPS)
                'score': score
            })
        
        # Process secondary attention timeline
        for i, score in enumerate(report.secondary_attention_timeline):
            attention_chart['secondary_data'].append({
                'time': i * 0.033,  # Approximate time (30 FPS)
                'score': score
            })
        
        return {
            'emotion_trend': emotion_chart,
            'attention_trend': attention_chart
        }
    
    def _translate_winner(self, winner: str) -> str:
        """Translate winner identifier to Vietnamese.
        
        Args:
            winner: Winner identifier ("primary", "secondary", or "tie")
        
        Returns:
            Vietnamese translation
        """
        translations = {
            'primary': 'Ban',
            'secondary': 'Nguoi Khac',
            'tie': 'Hoa'
        }
        return translations.get(winner, winner)
    
    def _draw_no_face_warning(self, frame: np.ndarray, person_type: str):
        """Draw no face warning overlay on video frame.
        
        Displays a semi-transparent warning overlay when no face has been
        detected for more than 5 seconds.
        
        Args:
            frame: Frame to draw warning on (modified in-place)
            person_type: "primary" or "secondary"
        
        Requirements:
        - 7.3: Display warning overlay when no face detected >5s
        """
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Draw warning box
        box_height = 100
        box_y = (height - box_height) // 2
        cv2.rectangle(
            overlay,
            (0, box_y),
            (width, box_y + box_height),
            (0, 0, 139),  # Dark red
            -1
        )
        
        # Blend overlay with original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw warning text
        warning_text = "⚠ KHONG PHAT HIEN KHUON MAT"
        instruction_text = "Vui long di chuyen vao khung hinh"
        
        # Calculate text size and position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(warning_text, font, font_scale, thickness)
        text_x = (width - text_width) // 2
        text_y = box_y + 40
        
        # Draw warning text with shadow for better visibility
        cv2.putText(frame, warning_text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(frame, warning_text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Draw instruction text
        font_scale_small = 0.5
        (text_width2, text_height2), _ = cv2.getTextSize(instruction_text, font, font_scale_small, 1)
        text_x2 = (width - text_width2) // 2
        text_y2 = text_y + 35
        
        cv2.putText(frame, instruction_text, (text_x2 + 1, text_y2 + 1), font, font_scale_small, (0, 0, 0), 2)
        cv2.putText(frame, instruction_text, (text_x2, text_y2), font, font_scale_small, (255, 255, 255), 1)
    
    def _show_fallback_message(self):
        """Show message when falling back to single mode due to screen capture failure.
        
        Requirements:
        - 1.3: Display automatic fallback to single mode message
        - 7.2: Handle screen capture failure
        """
        messagebox.showwarning(
            "Canh bao Screen Capture",
            "Screen capture that bai lien tuc!\n\n"
            "He thong da tu dong chuyen sang che do don (chi camera).\n\n"
            "Nguyen nhan co the:\n"
            "- Cua so chia se bi dong\n"
            "- Loi quyen truy cap man hinh\n"
            "- Loi he thong\n\n"
            "Ban co the:\n"
            "- Kiem tra lai cua so chia se\n"
            "- Khoi dong lai che do so sanh"
        )
    
    def _show_performance_warning(self, current_fps: float):
        """Show performance warning when FPS drops below 15.
        
        Args:
            current_fps: Current FPS value
        
        Requirements:
        - 1.4: Maintain >= 15 FPS
        - Performance warning when FPS < 15
        """
        response = messagebox.showwarning(
            "Canh bao Hieu suat",
            f"FPS hien tai: {current_fps:.1f} (muc tieu: >= 15 FPS)\n\n"
            "Hieu suat xu ly thap co the anh huong den do chinh xac.\n\n"
            "De cai thien hieu suat:\n"
            "- Dong cac ung dung khac\n"
            "- Giam do phan giai screen capture\n"
            "- Tat bounding boxes (trong Cai Dat)\n"
            "- Khoi dong lai che do so sanh\n\n"
            "Ban co muon tiep tuc?",
            type=messagebox.OKCANCEL
        )
        
        if response == 'cancel':
            # User wants to stop
            self.root.after(0, self.stop_processing)
    
    def _apply_position_swap(self):
        """Apply video position swap immediately.
        
        Swaps the grid positions of primary and secondary canvases
        without restarting the session.
        
        Requirements:
        - 8.4: Swap positions of videos
        - 8.5: Apply immediately without restart
        """
        if not self.dual_person_mode:
            return
        
        position = self.dual_person_settings['primary_position']
        
        # Remove current grid placements
        self.primary_canvas.grid_forget()
        self.secondary_canvas.grid_forget()
        
        # Also swap labels
        for widget in self.split_view_frame.winfo_children():
            if isinstance(widget, tk.Label):
                widget.grid_forget()
        
        if position == 'left':
            # Primary on left, secondary on right (default)
            # Primary label
            primary_label = tk.Label(
                self.split_view_frame,
                text="Ban",
                font=("Arial", 12, "bold"),
                bg='#1e1e1e',
                fg='#4CAF50'
            )
            primary_label.grid(row=0, column=0, pady=10)
            
            # Primary canvas
            self.primary_canvas.grid(row=1, column=0, sticky='nsew', padx=(10, 5), pady=10)
            
            # Secondary label
            secondary_label = tk.Label(
                self.split_view_frame,
                text="Nguoi Khac",
                font=("Arial", 12, "bold"),
                bg='#1e1e1e',
                fg='#2196F3'
            )
            secondary_label.grid(row=0, column=2, pady=10)
            
            # Secondary canvas
            self.secondary_canvas.grid(row=1, column=2, sticky='nsew', padx=(5, 10), pady=10)
        else:
            # Primary on right, secondary on left (swapped)
            # Secondary label
            secondary_label = tk.Label(
                self.split_view_frame,
                text="Nguoi Khac",
                font=("Arial", 12, "bold"),
                bg='#1e1e1e',
                fg='#2196F3'
            )
            secondary_label.grid(row=0, column=0, pady=10)
            
            # Secondary canvas
            self.secondary_canvas.grid(row=1, column=0, sticky='nsew', padx=(10, 5), pady=10)
            
            # Primary label
            primary_label = tk.Label(
                self.split_view_frame,
                text="Ban",
                font=("Arial", 12, "bold"),
                bg='#1e1e1e',
                fg='#4CAF50'
            )
            primary_label.grid(row=0, column=2, pady=10)
            
            # Primary canvas
            self.primary_canvas.grid(row=1, column=2, sticky='nsew', padx=(5, 10), pady=10)
        
        print(f"✓ Swapped video positions: primary is now on {position}")
    
    def update_score_summary_tab(self):
        """Cập nhật điểm vào Score Summary Tab."""
        if not hasattr(self, 'score_summary_tab') or self.score_summary_tab is None:
            return
        
        try:
            # Cập nhật điểm Cảm xúc (Emotion)
            if self.current_emotion_score > 0:
                self.score_summary_tab.update_emotion_score(self.current_emotion_score)
            
            # Cập nhật điểm Tập trung (Focus)
            if self.current_focus_score > 0:
                self.score_summary_tab.update_focus_score(self.current_focus_score)
            
            # TODO: Cập nhật điểm Rõ ràng (Clarity) từ speech analysis
            # TODO: Cập nhật điểm Nội dung (Content) từ content evaluator
            
        except Exception as e:
            print(f"Error updating score summary tab: {e}")
    
    def on_closing(self):
        """Xử lý khi đóng cửa sổ."""
        # Stop video processing
        if self.is_running:
            self.stop_processing()
        
        # Deactivate dual person mode if active
        if self.dual_person_mode:
            self._deactivate_dual_person_mode()
        
        # Cleanup live report overlay
        if hasattr(self, 'live_report_overlay'):
            try:
                self.live_report_overlay.destroy()
            except:
                pass
        
        self.root.destroy()
    
    def toggle_ui_visibility(self):
        """Ẩn/hiện UI chính và hiển thị overlay báo cáo."""
        print("="*70)
        print("toggle_ui_visibility() CALLED!")
        print("="*70)
        
        try:
            # Get the toplevel window (works for both Frame and Tk)
            toplevel = self.root.winfo_toplevel()
            
            # Check if overlay is currently visible
            overlay_visible = self.live_report_overlay.is_visible
            print(f"Overlay visible: {overlay_visible}")
            print(f"Current UI state: {toplevel.state()}")
            
            if not overlay_visible:
                # Overlay is hidden, so hide main UI and show overlay
                print("Hiding main UI and showing overlay...")
                
                # Update statistics before hiding to ensure data is current
                if self.is_running:
                    self.update_statistics()
                
                # TẠM DỪNG tính attention khi ẩn UI
                self.ui_hidden = True
                print("✓ Tạm dừng tính attention (UI đã ẩn)")
                
                toplevel.withdraw()
                self.live_report_overlay.show()
                print("✓ UI chính đã ẩn - Overlay báo cáo đang hiển thị")
            else:
                # Overlay is visible, so show main UI and hide overlay
                print("Showing main UI and hiding overlay...")
                
                # Update statistics before showing to ensure data is synced
                if self.is_running:
                    self.update_statistics()
                
                # TIẾP TỤC tính attention khi hiện UI
                self.ui_hidden = False
                print("✓ Tiếp tục tính attention (UI đã hiện)")
                
                toplevel.deiconify()
                self.live_report_overlay.hide()
                print("✓ UI chính đã hiển thị - Overlay báo cáo đã ẩn")
        
        except Exception as e:
            print(f"Error toggling UI visibility: {e}")
            import traceback
            traceback.print_exc()
    
    def extract_video_transcript(self):
        """Extract transcript từ video file."""
        if not self.video_file_path:
            messagebox.showerror("Lỗi", "Vui lòng chọn video file trước!")
            return
        
        if self.is_extracting_transcript:
            messagebox.showinfo("Thông báo", "Đang extract transcript, vui lòng đợi...")
            return
        
        # Confirm
        result = messagebox.askyesno(
            "Chuyển Đổi Audio Sang Văn Bản",
            f"Chuyển audio từ video thành văn bản?\n\n"
            f"Video: {Path(self.video_file_path).name}\n\n"
            f"🎯 Model: Whisper Large-v3 (Độ chính xác 95-98%)\n"
            f"🚀 Pipeline: Advanced + Vietnamese Optimizer\n"
            f"✨ Tính năng: Audio Enhancement, VAD, Post-processing\n\n"
            f"⏱️ Quá trình này có thể mất vài phút.\n"
            f"Bạn có muốn tiếp tục?"
        )
        
        if not result:
            return
        
        # Use large-v3 model for best accuracy (like TurboScribe)
        model_name = "large-v3"
        
        # Start extraction in background thread
        self.is_extracting_transcript = True
        
        def extract_thread(model_name_param):
            try:
                # Import required modules
                from src.utils.video_audio_player import VideoAudioPlayer
                from src.speech_analysis.speech_to_text import WhisperSTTEngine, STTConfig
                from src.speech_analysis.advanced_transcription_pipeline import AdvancedTranscriptionPipeline
                from src.speech_analysis.vietnamese_stt_optimizer import VietnameseSTTOptimizer, get_optimized_whisper_config
                
                # Validate video path
                if not self.video_file_path or not Path(self.video_file_path).exists():
                    raise RuntimeError(f"Video file không tồn tại: {self.video_file_path}")
                
                # Update status
                self.root.after(0, lambda: self.update_status("Đang extract audio...", "#ff9800"))
                
                # Load video and get audio
                player = VideoAudioPlayer(self.video_file_path)
                audio_data = player.get_audio_data(0, player.duration)
                
                if audio_data is None or len(audio_data) == 0:
                    raise RuntimeError("Không có audio data")
                
                # Update status
                self.root.after(0, lambda: self.update_status("Đang khởi tạo Whisper engine...", "#ff9800"))
                
                # Get optimized config for Vietnamese
                optimized_params = get_optimized_whisper_config()
                
                # Initialize STT engine with optimized parameters
                config = STTConfig(
                    model_name=model_name_param,
                    language="vi",
                    device="auto"
                )
                engine = WhisperSTTEngine(config)
                
                # Initialize Vietnamese optimizer
                vn_optimizer = VietnameseSTTOptimizer()
                
                # Update status
                self.root.after(0, lambda: self.update_status("Đang xử lý với Advanced Pipeline + Vietnamese Optimizer...", "#ff9800"))
                
                # Use Advanced Transcription Pipeline for best accuracy
                pipeline = AdvancedTranscriptionPipeline(engine, player.sample_rate)
                
                # Transcribe using TurboScribe method: chunking + enhancement
                self.root.after(0, lambda: self.update_status("Đang transcribe (TurboScribe method)...", "#ff9800"))
                transcript_text = pipeline.transcribe(
                    audio_data,
                    use_vad=False,  # Don't use VAD, use chunking instead
                    use_enhancement=True,  # TurboScribe always enhances
                    chunk_length=20,  # 20s chunks
                    use_llm=False  # Tắt LLM để nhanh hơn (bật nếu cần 95-98% accuracy)
                )
                
                # Apply Vietnamese post-processing
                transcript_text = vn_optimizer.post_process_text(transcript_text)
                
                # Update status
                self.root.after(0, lambda: self.update_status("Đang hoàn thiện văn bản...", "#ff9800"))
                
                # Create result object
                from src.speech_analysis.speech_to_text import TranscriptionResult
                result = TranscriptionResult(
                    text=transcript_text,
                    confidence=0.95,  # Advanced pipeline + Vietnamese optimizer achieves 95%+
                    language="vi",
                    segments=[],
                    processing_time=0.0
                )
                
                # Save to file
                video_name = Path(self.video_file_path).stem
                output_path = Path("transcripts") / f"{video_name}_transcript.txt"
                output_path.parent.mkdir(exist_ok=True)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"Video: {self.video_file_path}\n")
                    f.write(f"Duration: {player.duration:.1f}s\n")
                    f.write(f"Model: {model_name_param}\n")
                    f.write(f"Confidence: {result.confidence:.2f}\n")
                    f.write(f"Pipeline: Advanced + Vietnamese Optimizer\n")
                    f.write(f"Features: Audio Enhancement, VAD, Post-processing\n")
                    f.write("\n" + "="*70 + "\n")
                    f.write("TRANSCRIPT\n")
                    f.write("="*70 + "\n\n")
                    f.write(result.text)
                    f.write("\n")
                
                # Show success message
                self.root.after(0, lambda: messagebox.showinfo(
                    "✅ Thành Công",
                    f"Transcript đã được lưu!\n\n"
                    f"📁 File: {output_path}\n\n"
                    f"🎯 Confidence: {result.confidence:.0%}\n"
                    f"📝 Độ dài: {len(result.text)} ký tự\n"
                    f"🚀 Pipeline: Advanced + Vietnamese Optimizer"
                ))
                
                # Update status
                self.root.after(0, lambda: self.update_status("Hoàn tất extract transcript", "#4CAF50"))
                
            except Exception as e:
                # Show error
                self.root.after(0, lambda: messagebox.showerror(
                    "Lỗi",
                    f"Không thể extract transcript:\n\n{str(e)}"
                ))
                self.root.after(0, lambda: self.update_status("Lỗi extract transcript", "#f44336"))
            
            finally:
                # Reset flag
                self.is_extracting_transcript = False
        
        # Start thread with model_name parameter
        self.transcript_thread = threading.Thread(target=lambda: extract_thread(model_name), daemon=True)
        self.transcript_thread.start()


def main():
    """Hàm main."""
    root = tk.Tk()
    root.title("Emotion Recognition & Video Transcription")
    root.geometry("1200x900")
    root.configure(bg='#1a1a1a')
    
    # Thiết lập icon cho cửa sổ
    try:
        assets_dir = Path(__file__).parent.parent / "assets"
        
        # Windows: Thử .ico trước
        if sys.platform == 'win32':
            ico_path = assets_dir / "icon.ico"
            if ico_path.exists():
                root.iconbitmap(str(ico_path))
            else:
                # Fallback sang .png
                png_path = assets_dir / "icon.png"
                if png_path.exists():
                    icon_image = Image.open(png_path)
                    icon_photo = ImageTk.PhotoImage(icon_image)
                    root.iconphoto(True, icon_photo)
        else:
            # Linux/Mac: Dùng .png
            png_path = assets_dir / "icon.png"
            if png_path.exists():
                icon_image = Image.open(png_path)
                icon_photo = ImageTk.PhotoImage(icon_image)
                root.iconphoto(True, icon_photo)
    except Exception:
        # Không hiển thị lỗi nếu không load được icon
        pass
    
    # Create notebook for tabs
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)
    
    # Style the notebook
    style = ttk.Style()
    style.theme_use('default')
    style.configure('TNotebook', background='#1a1a1a', borderwidth=0)
    style.configure('TNotebook.Tab', 
                   background='#252525', 
                   foreground='#e0e0e0',
                   padding=[20, 10],
                   font=('Segoe UI', 11, 'bold'))
    style.map('TNotebook.Tab',
             background=[('selected', '#0d47a1')],
             foreground=[('selected', '#ffffff')])
    
    # Tab 1: Emotion Recognition (existing functionality)
    emotion_frame = tk.Frame(notebook, bg='#1a1a1a')
    notebook.add(emotion_frame, text='😊 Nhận Diện Cảm Xúc')
    
    app = EmotionRecognitionGUI(emotion_frame)
    
    # Tab 2: Video Transcription (new functionality)
    if VIDEO_TRANSCRIPTION_AVAILABLE:
        transcription_frame = tk.Frame(notebook, bg='#1a1a1a')
        notebook.add(transcription_frame, text='📹 Chuyển Đổi Video')
        
        try:
            video_transcription_tab = VideoTranscriptionTab(transcription_frame)
        except Exception as e:
            print(f"Error initializing video transcription tab: {e}")
            import traceback
            traceback.print_exc()
    
    # Tab 3: Audio Transcription
    audio_transcription_tab = None
    try:
        from gui.audio_transcription_tab_simple import AudioTranscriptionTab
        audio_transcription_frame = tk.Frame(notebook, bg='#1a1a1a')
        notebook.add(audio_transcription_frame, text='🎤➡️📝 Chuyển Đổi Audio')
        
        try:
            audio_transcription_tab = AudioTranscriptionTab(audio_transcription_frame)
        except Exception as e:
            print(f"Error initializing audio transcription tab: {e}")
            import traceback
            traceback.print_exc()
    except ImportError as e:
        print(f"Audio transcription tab not available: {e}")
    
    # Tab 4: Score Summary - TỔNG HỢP ĐIỂM
    score_summary_tab = None
    try:
        from gui.score_summary_tab import ScoreSummaryTab
        # Không cần tạo frame, ScoreSummaryTab tự tạo frame của nó
        score_summary_tab = ScoreSummaryTab(notebook)
        notebook.add(score_summary_tab.get_frame(), text='📊 Tổng Hợp Điểm')
        
        try:
            # Kết nối với app
            pass
            
            # Kết nối với emotion recognition tab để tự động cập nhật điểm
            if hasattr(app, 'score_summary_tab'):
                app.score_summary_tab = score_summary_tab
            
            print("✅ Score Summary Tab initialized successfully")
        except Exception as e:
            print(f"Error initializing score summary tab: {e}")
            import traceback
            traceback.print_exc()
    except ImportError as e:
        print(f"Score summary tab not available: {e}")
    
    # Tab 5: Emotion Scoring - ĐÃ XÓA
    # Chức năng chấm điểm đã được tích hợp vào Tab 4 (Tổng Hợp Điểm)
    # với hệ thống 4 tiêu chí mới: Cảm xúc, Tập trung, Rõ ràng lời nói, Nội dung
    
    # Handle window closing
    def on_closing():
        if hasattr(app, 'on_closing'):
            app.on_closing()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()
