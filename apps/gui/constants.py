# -*- coding: utf-8 -*-
"""
GUI Constants and Configuration

Contains color schemes, button styles, and other UI constants.
"""

# Color Scheme
COLORS = {
    'bg_dark': '#1a1a1a',
    'bg_medium': '#252525',
    'bg_light': '#2b2b2b',
    'primary': '#0d47a1',
    'success': '#4CAF50',
    'warning': '#ff9800',
    'error': '#f44336',
    'info': '#2196F3',
    'text_primary': '#ffffff',
    'text_secondary': '#e0e0e0',
    'text_muted': '#9e9e9e',
    'accent_blue': '#64b5f6',
    'accent_green': '#81c784',
    'accent_orange': '#ffa726',
    'accent_purple': '#7e57c2',
    'accent_cyan': '#00acc1',
    'accent_teal': '#00897b',
    'accent_grey': '#546e7a',
    'separator': '#3a3a3a',
}

# Button Styles
BUTTON_STYLE = {
    'font': ("Segoe UI", 11, "bold"),
    'cursor': 'hand2',
    'relief': 'flat',
    'padx': 20,
    'pady': 10
}

BUTTON_STYLE_SMALL = {
    'font': ("Arial", 9, "bold"),
    'cursor': 'hand2',
    'width': 12,
    'height': 1
}

# Radio Button Style
RADIO_STYLE = {
    'font': ("Segoe UI", 10),
    'bg': COLORS['bg_medium'],
    'fg': COLORS['text_secondary'],
    'selectcolor': COLORS['bg_dark'],
    'activebackground': COLORS['bg_medium'],
    'activeforeground': COLORS['text_primary'],
    'highlightthickness': 0
}

# Label Styles
LABEL_STYLE_HEADER = {
    'font': ("Segoe UI", 13, "bold"),
    'bg': COLORS['bg_medium'],
}

LABEL_STYLE_STATUS = {
    'font': ("Segoe UI", 11, "bold"),
    'bg': COLORS['bg_medium'],
}

# Text Widget Styles
TEXT_STYLE_STATS = {
    'font': ("Consolas", 9),
    'bg': '#1e1e1e',
    'fg': COLORS['text_secondary'],
    'wrap': 'word',
    'relief': 'flat',
    'padx': 10,
    'pady': 10
}

TEXT_STYLE_DUAL = {
    'font': ("Courier", 9),
    'bg': COLORS['bg_light'],
    'fg': COLORS['text_primary'],
    'wrap': 'word',
    'state': 'disabled',
    'height': 15
}

# Emotion Icons
EMOTION_ICONS = {
    'happy': '😊',
    'sad': '😢',
    'angry': '😠',
    'surprise': '😲',
    'fear': '😨',
    'disgust': '🤢',
    'neutral': '😐'
}

# Window Sizes
WINDOW_SIZE = {
    'width': 1400,
    'height': 900,
    'min_width': 1000,
    'min_height': 700
}

# Video Display Limits
VIDEO_DISPLAY = {
    'max_width': 800,
    'max_height': 600,
    'default_width': 640,
    'default_height': 480
}

# Performance Settings - BALANCED (Fast Detection + Smooth Display)
PERFORMANCE = {
    'default_skip_frames': 1,  # GIẢM: Skip 1 frame = process every 2nd frame (nhanh hơn)
    'default_resize_factor': 0.7,  # 70% để giảm processing
    'stats_update_interval': 30,  # GIẢM: Update stats thường xuyên hơn
    'max_consecutive_failures': 30,
    'frame_buffer_size': 3,
    'use_fast_interpolation': True,  # Use INTER_NEAREST (fastest)
    'face_detection_skip': 2,  # GIẢM: Skip 2 frames (process every 3rd frame) - nhanh hơn
    'emotion_cache_frames': 5,  # GIẢM: Cache ít hơn để update nhanh hơn
    'max_faces_to_process': 1,
    'disable_attention_in_video': True,
    'disable_appearance_in_video': True,
    'use_emotion_cache': True,
    'async_processing': True,
    'display_update_throttle': 0.033,  # 30 FPS - mượt mà
    'processing_thread_priority': 'normal',
    'ui_update_interval': 10,  # GIẢM: Update UI mỗi 10 frames (~0.3s)
    'disable_overlays_in_video': True,
}
