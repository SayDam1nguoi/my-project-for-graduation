# -*- coding: utf-8 -*-
"""
Audio Recording Tab for GUI

Provides a user interface for audio recording functionality.
Implements Requirements 1.1, 1.2, 1.3, 2.1, 2.2, 3.1, 3.2, 5.1, 5.2, 7.3
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import sys
import os
import subprocess
import platform
import shutil
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.audio_recording.coordinator import AudioRecordingCoordinator
from src.audio_recording.config import AudioRecordingConfig
from src.audio_recording.models import RecordingState, AudioDevice
from src.audio_recording.exceptions import AudioRecordingError, StateError, DeviceError, FileSystemError
from src.audio_recording.manager import RecordingManager
from src.audio_recording.settings import AudioRecordingSettings
from apps.gui.file_save_dialog import ask_save_file
from src.audio_recording.utils import (
    check_disk_space, get_error_message_with_solution, 
    format_file_size, get_permission_instructions
)
from src.audio_recording.transcriber import get_transcriber
from apps.gui.audio_settings_dialog import AudioSettingsDialog
from src.database import get_mongodb_manager

# Try to import audio player
try:
    from src.utils.video_audio_player import VideoAudioPlayer
    AUDIO_PLAYER_AVAILABLE = True
except ImportError:
    AUDIO_PLAYER_AVAILABLE = False


class AudioRecordingTab:
    """
    GUI tab for audio recording functionality.
    
    Implements Requirements:
    - 1.1: Start recording from microphone
    - 1.2: Display recording status indicator
    - 1.3: Stop recording and save file
    - 2.1: Display elapsed time
    - 2.2: Display audio level indicator
    - 3.1: Display available microphone devices
    - 3.2: Allow device selection
    - 5.1: Display sample rate options
    - 5.2: Display bit depth options
    - 7.3: Display recording button in UI
    """
    
    def __init__(self, parent_frame: tk.Frame):
        """
        Initialize audio recording tab.
        
        Args:
            parent_frame: Parent frame to contain this tab
        """
        self.parent = parent_frame
        self.coordinator: Optional[AudioRecordingCoordinator] = None
        self.config: Optional[AudioRecordingConfig] = None
        self.recording_manager: Optional[RecordingManager] = None
        self.settings_manager: Optional[AudioRecordingSettings] = None
        
        # State variables
        self.is_recording = False
        self.is_paused = False
        self.current_file_path: Optional[str] = None
        self.available_devices = []
        self.audio_player: Optional[VideoAudioPlayer] = None
        self.transcriber = None
        self.is_transcribing = False
        self.mongodb_manager = None
        self.current_recording_id: Optional[str] = None
        
        # Create UI
        self.create_widgets()
        
        # Initialize components
        self.initialize_components()
    
    def create_widgets(self):
        """Create all UI widgets for the tab."""
        # Main container with padding
        main_container = tk.Frame(self.parent, bg='#1a1a1a')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Configure grid
        main_container.grid_rowconfigure(0, weight=0)  # Header
        main_container.grid_rowconfigure(1, weight=0)  # Device selection
        main_container.grid_rowconfigure(2, weight=0)  # Settings
        main_container.grid_rowconfigure(3, weight=0)  # Recording controls
        main_container.grid_rowconfigure(4, weight=0)  # Audio level
        main_container.grid_rowconfigure(5, weight=0)  # Status/Info
        main_container.grid_rowconfigure(6, weight=1)  # Recordings list
        main_container.grid_columnconfigure(0, weight=1)
        
        # Header
        header_frame = tk.Frame(main_container, bg='#0d47a1', height=60)
        header_frame.grid(row=0, column=0, sticky='ew', pady=(0, 20))
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame,
            text="🎙️ THU ÂM",
            font=("Segoe UI", 18, "bold"),
            bg='#0d47a1',
            fg='#ffffff'
        )
        header_label.pack(pady=15)
        
        # Device selection section (Requirements 3.1, 3.2)
        device_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        device_frame.grid(row=1, column=0, sticky='ew', pady=(0, 15))
        
        device_inner = tk.Frame(device_frame, bg='#252525')
        device_inner.pack(fill=tk.X, padx=20, pady=15)
        
        device_label = tk.Label(
            device_inner,
            text="🎤 Thiết bị microphone:",
            font=("Segoe UI", 11, "bold"),
            bg='#252525',
            fg='#90caf9'
        )
        device_label.pack(side=tk.LEFT, padx=(0, 15))
        
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            device_inner,
            textvariable=self.device_var,
            state='readonly',
            width=40,
            font=("Segoe UI", 10)
        )
        self.device_combo.pack(side=tk.LEFT, padx=10)
        self.device_combo.bind('<<ComboboxSelected>>', self.on_device_selected)
        
        # Refresh devices button
        self.refresh_devices_button = tk.Button(
            device_inner,
            text="🔄",
            font=("Segoe UI", 10, "bold"),
            bg='#424242',
            fg='#ffffff',
            activebackground='#616161',
            command=self.refresh_devices,
            cursor='hand2',
            relief=tk.FLAT,
            padx=10,
            pady=8
        )
        self.refresh_devices_button.pack(side=tk.LEFT, padx=5)
        
        # Settings button
        self.settings_button = tk.Button(
            device_inner,
            text="⚙️ Cài đặt",
            font=("Segoe UI", 10, "bold"),
            bg='#424242',
            fg='#ffffff',
            activebackground='#616161',
            command=self.open_settings_dialog,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8
        )
        self.settings_button.pack(side=tk.LEFT, padx=10)
        
        # Settings section (Requirements 5.1, 5.2)
        settings_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        settings_frame.grid(row=2, column=0, sticky='ew', pady=(0, 15))
        
        settings_inner = tk.Frame(settings_frame, bg='#252525')
        settings_inner.pack(fill=tk.X, padx=20, pady=15)
        
        # Sample rate selection
        sample_rate_frame = tk.Frame(settings_inner, bg='#252525')
        sample_rate_frame.pack(side=tk.LEFT, padx=(0, 30))
        
        sample_rate_label = tk.Label(
            sample_rate_frame,
            text="📊 Sample Rate:",
            font=("Segoe UI", 10, "bold"),
            bg='#252525',
            fg='#81c784'
        )
        sample_rate_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.sample_rate_var = tk.StringVar(value="16000 Hz")
        sample_rate_options = ["16000 Hz", "44100 Hz", "48000 Hz"]
        
        self.sample_rate_combo = ttk.Combobox(
            sample_rate_frame,
            textvariable=self.sample_rate_var,
            values=sample_rate_options,
            state='readonly',
            width=15,
            font=("Segoe UI", 10)
        )
        self.sample_rate_combo.pack(side=tk.LEFT)
        
        # Bit depth selection
        bit_depth_frame = tk.Frame(settings_inner, bg='#252525')
        bit_depth_frame.pack(side=tk.LEFT, padx=30)
        
        bit_depth_label = tk.Label(
            bit_depth_frame,
            text="🎵 Bit Depth:",
            font=("Segoe UI", 10, "bold"),
            bg='#252525',
            fg='#81c784'
        )
        bit_depth_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.bit_depth_var = tk.StringVar(value="16-bit")
        bit_depth_options = ["16-bit", "24-bit"]
        
        self.bit_depth_combo = ttk.Combobox(
            bit_depth_frame,
            textvariable=self.bit_depth_var,
            values=bit_depth_options,
            state='readonly',
            width=10,
            font=("Segoe UI", 10)
        )
        self.bit_depth_combo.pack(side=tk.LEFT)
        
        # Recording controls section (Requirements 1.1, 1.3, 7.3)
        controls_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        controls_frame.grid(row=3, column=0, sticky='ew', pady=(0, 15))
        
        controls_inner = tk.Frame(controls_frame, bg='#252525')
        controls_inner.pack(fill=tk.X, padx=20, pady=15)
        
        # Start/Stop button
        self.record_button = tk.Button(
            controls_inner,
            text="⏺️ BẮT ĐẦU THU ÂM",
            font=("Segoe UI", 12, "bold"),
            bg='#d32f2f',
            fg='#ffffff',
            activebackground='#b71c1c',
            command=self.toggle_recording,
            cursor='hand2',
            relief=tk.FLAT,
            padx=30,
            pady=12
        )
        self.record_button.pack(side=tk.LEFT, padx=10)
        
        # Pause/Resume button
        self.pause_button = tk.Button(
            controls_inner,
            text="⏸️ TẠM DỪNG",
            font=("Segoe UI", 11, "bold"),
            bg='#ff9800',
            fg='#ffffff',
            activebackground='#f57c00',
            command=self.toggle_pause,
            cursor='hand2',
            relief=tk.FLAT,
            padx=25,
            pady=10,
            state=tk.DISABLED
        )
        self.pause_button.pack(side=tk.LEFT, padx=10)
        
        # Elapsed time display (Requirement 2.1)
        self.time_label = tk.Label(
            controls_inner,
            text="⏱️ 00:00:00",
            font=("Segoe UI", 14, "bold"),
            bg='#252525',
            fg='#4CAF50'
        )
        self.time_label.pack(side=tk.RIGHT, padx=20)
        
        # Audio level section (Requirement 2.2)
        level_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        level_frame.grid(row=4, column=0, sticky='ew', pady=(0, 15))
        
        level_inner = tk.Frame(level_frame, bg='#252525')
        level_inner.pack(fill=tk.X, padx=20, pady=15)
        
        level_label = tk.Label(
            level_inner,
            text="🔊 Mức âm thanh:",
            font=("Segoe UI", 10, "bold"),
            bg='#252525',
            fg='#90caf9'
        )
        level_label.pack(anchor='w', pady=(0, 10))
        
        # Audio level progress bar
        self.level_progress = ttk.Progressbar(
            level_inner,
            mode='determinate',
            length=400,
            maximum=100
        )
        self.level_progress.pack(fill=tk.X, pady=(0, 5))
        
        # Level value label
        self.level_value_label = tk.Label(
            level_inner,
            text="0%",
            font=("Segoe UI", 9),
            bg='#252525',
            fg='#9e9e9e'
        )
        self.level_value_label.pack(anchor='w')
        
        # Status/Info section (Requirement 1.2)
        status_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        status_frame.grid(row=5, column=0, sticky='ew', pady=(0, 15))
        
        status_label = tk.Label(
            status_frame,
            text="📋 Trạng thái:",
            font=("Segoe UI", 11, "bold"),
            bg='#252525',
            fg='#81c784'
        )
        status_label.pack(anchor='w', padx=20, pady=(15, 10))
        
        # Status text area
        text_container = tk.Frame(status_frame, bg='#1a1a1a', relief=tk.SOLID, bd=1)
        text_container.pack(fill=tk.X, padx=20, pady=(0, 15))
        
        self.status_text = tk.Text(
            text_container,
            font=("Consolas", 10),
            bg='#1e1e1e',
            fg='#e0e0e0',
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            padx=15,
            pady=15,
            height=6
        )
        self.status_text.pack(fill=tk.X)
        
        # Initial status message
        self.update_status("⏸️ Chưa bắt đầu thu âm\n\nNhấn 'BẮT ĐẦU THU ÂM' để bắt đầu ghi âm.")
        
        # Recordings list section (Requirements 8.4, 8.5, 9.1, 9.2, 9.3, 9.4, 9.5)
        recordings_frame = tk.Frame(main_container, bg='#252525', relief=tk.FLAT, bd=0)
        recordings_frame.grid(row=6, column=0, sticky='nsew', pady=(0, 0))
        
        # Header with refresh button
        recordings_header = tk.Frame(recordings_frame, bg='#252525')
        recordings_header.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        recordings_label = tk.Label(
            recordings_header,
            text="📁 Danh sách bản thu âm:",
            font=("Segoe UI", 11, "bold"),
            bg='#252525',
            fg='#81c784'
        )
        recordings_label.pack(side=tk.LEFT)
        
        refresh_button = tk.Button(
            recordings_header,
            text="🔄 Làm mới",
            font=("Segoe UI", 9),
            bg='#424242',
            fg='#ffffff',
            activebackground='#616161',
            command=self.refresh_recordings_list,
            cursor='hand2',
            relief=tk.FLAT,
            padx=15,
            pady=5
        )
        refresh_button.pack(side=tk.RIGHT)
        
        # Treeview for recordings list
        tree_container = tk.Frame(recordings_frame, bg='#1a1a1a', relief=tk.SOLID, bd=1)
        tree_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))
        
        # Create treeview with scrollbar
        tree_scroll = ttk.Scrollbar(tree_container)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.recordings_tree = ttk.Treeview(
            tree_container,
            columns=("name", "date", "duration", "size"),
            show="headings",
            yscrollcommand=tree_scroll.set,
            height=5
        )
        tree_scroll.config(command=self.recordings_tree.yview)
        
        # Define columns
        self.recordings_tree.heading("name", text="Tên file")
        self.recordings_tree.heading("date", text="Ngày tạo")
        self.recordings_tree.heading("duration", text="Thời lượng")
        self.recordings_tree.heading("size", text="Kích thước")
        
        self.recordings_tree.column("name", width=250, anchor='w')
        self.recordings_tree.column("date", width=150, anchor='center')
        self.recordings_tree.column("duration", width=100, anchor='center')
        self.recordings_tree.column("size", width=100, anchor='center')
        
        self.recordings_tree.pack(fill=tk.BOTH, expand=True)
        
        # Bind selection event
        self.recordings_tree.bind('<<TreeviewSelect>>', self.on_recording_selected)
        
        # Instruction label
        instruction_label = tk.Label(
            recordings_frame,
            text="💡 Chọn một file trong danh sách để xem các tùy chọn",
            font=("Segoe UI", 9, "italic"),
            bg='#252525',
            fg='#9e9e9e'
        )
        instruction_label.pack(pady=(5, 5))
        
        # Action buttons frame - Row 1
        actions_frame_row1 = tk.Frame(recordings_frame, bg='#252525')
        actions_frame_row1.pack(fill=tk.X, padx=20, pady=(5, 5))
        
        # Play button
        self.play_button = tk.Button(
            actions_frame_row1,
            text="▶️ Phát",
            font=("Segoe UI", 10, "bold"),
            bg='#4CAF50',
            fg='#ffffff',
            activebackground='#388e3c',
            command=self.play_recording,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.play_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Delete button
        self.delete_button = tk.Button(
            actions_frame_row1,
            text="🗑️ Xóa",
            font=("Segoe UI", 10, "bold"),
            bg='#f44336',
            fg='#ffffff',
            activebackground='#d32f2f',
            command=self.delete_recording,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.delete_button.pack(side=tk.LEFT, padx=10)
        
        # Export button
        self.export_button = tk.Button(
            actions_frame_row1,
            text="💾 Xuất",
            font=("Segoe UI", 10, "bold"),
            bg='#9C27B0',
            fg='#ffffff',
            activebackground='#7B1FA2',
            command=self.export_recording,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.export_button.pack(side=tk.LEFT, padx=10)
        
        # Action buttons frame - Row 2
        actions_frame_row2 = tk.Frame(recordings_frame, bg='#252525')
        actions_frame_row2.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        # Transcribe button (HIGHLIGHTED)
        self.transcribe_button = tk.Button(
            actions_frame_row2,
            text="📝 Chuyển sang text",
            font=("Segoe UI", 11, "bold"),
            bg='#FF9800',
            fg='#ffffff',
            activebackground='#F57C00',
            command=self.transcribe_recording,
            cursor='hand2',
            relief=tk.FLAT,
            padx=25,
            pady=10,
            state=tk.DISABLED
        )
        self.transcribe_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Open folder button
        self.open_folder_button = tk.Button(
            actions_frame_row2,
            text="📂 Mở thư mục",
            font=("Segoe UI", 10, "bold"),
            bg='#2196F3',
            fg='#ffffff',
            activebackground='#1976d2',
            command=self.open_recordings_folder,
            cursor='hand2',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            state=tk.DISABLED
        )
        self.open_folder_button.pack(side=tk.LEFT, padx=10)
        
        # File path display (in row 2, right side)
        self.file_path_label = tk.Label(
            actions_frame_row2,
            text="",
            font=("Segoe UI", 9),
            bg='#252525',
            fg='#9e9e9e',
            anchor='w'
        )
        self.file_path_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
    
    def initialize_components(self):
        """Initialize audio recording components.
        
        Implements Requirements 10.1, 10.2 - Error handling and user notifications
        """
        try:
            # Initialize settings manager
            self.settings_manager = AudioRecordingSettings()
            
            # Load configuration from settings file
            self.config = self.settings_manager.load_settings()
            
            # Check disk space
            has_space, available_mb = check_disk_space(self.config.output_dir, required_mb=100)
            if not has_space:
                self.show_warning(
                    "Cảnh Báo Dung Lượng",
                    f"Dung lượng ổ đĩa thấp: {available_mb} MB còn lại.\n\n"
                    "Bạn có thể gặp lỗi khi thu âm.\n"
                    "Khuyến nghị giải phóng ít nhất 100 MB."
                )
            
            # Initialize coordinator with GUI callback
            self.coordinator = AudioRecordingCoordinator(
                config=self.config,
                gui_callback=self.on_audio_update
            )
            
            # Initialize recording manager
            self.recording_manager = RecordingManager(self.config.output_dir)
            
            # Initialize transcriber
            try:
                self.transcriber = get_transcriber()
                if self.transcriber.is_available():
                    logger.info("Audio transcriber initialized successfully")
                else:
                    logger.warning("Audio transcriber not available")
            except Exception as e:
                logger.error(f"Failed to initialize transcriber: {e}")
                self.transcriber = None
            
            # Initialize MongoDB
            try:
                self.mongodb_manager = get_mongodb_manager()
                if self.mongodb_manager.is_connected():
                    logger.info("MongoDB connected successfully")
                    stats = self.mongodb_manager.get_database_stats()
                    logger.info(f"MongoDB stats: {stats}")
                else:
                    logger.warning("MongoDB not connected - will save to disk only")
            except Exception as e:
                logger.error(f"Failed to initialize MongoDB: {e}")
                self.mongodb_manager = None
            
            # Load available devices
            self.load_devices()
            
            # Load recordings list
            self.refresh_recordings_list()
            
            # Update UI with loaded settings
            self.update_settings_ui()
            
            self.update_status("✓ Sẵn sàng thu âm\n\nChọn thiết bị microphone và nhấn 'BẮT ĐẦU THU ÂM'.")
            
        except DeviceError as e:
            # Handle device errors with specific instructions
            title, message = get_error_message_with_solution(e)
            self.show_error(title, message)
            self.update_status(f"✗ {title}: Không thể khởi tạo")
            
        except Exception as e:
            # Generic error handling
            title, message = get_error_message_with_solution(e)
            self.show_error(title, message)
            self.update_status(f"✗ Lỗi khởi tạo: {str(e)}")
    
    def load_devices(self):
        """Load available audio devices.
        
        Implements Requirements 10.1, 10.2 - Graceful degradation for missing devices
        """
        try:
            self.available_devices = self.coordinator.get_available_devices()
            
            if not self.available_devices:
                # Graceful degradation - show helpful error with instructions
                title, message = get_error_message_with_solution(
                    DeviceError("No audio input devices found")
                )
                self.show_error(title, message)
                
                self.device_combo['values'] = ["❌ Không có thiết bị"]
                self.device_combo.current(0)
                self.record_button.config(state=tk.DISABLED)
                self.update_status(
                    "✗ Không tìm thấy microphone\n\n"
                    "Vui lòng:\n"
                    "• Kết nối microphone\n"
                    "• Cấp quyền truy cập\n"
                    "• Nhấn 'Làm mới' để thử lại"
                )
                return
            
            # Populate device combo box
            device_names = [f"{dev.name} (ID: {dev.id})" for dev in self.available_devices]
            self.device_combo['values'] = device_names
            
            # Select default device
            default_idx = 0
            for i, dev in enumerate(self.available_devices):
                if dev.is_default:
                    default_idx = i
                    break
            
            self.device_combo.current(default_idx)
            
            # Set device in coordinator
            selected_device = self.available_devices[default_idx]
            self.coordinator.set_device(selected_device.id)
            
            # Enable recording button
            self.record_button.config(state=tk.NORMAL)
            
        except DeviceError as e:
            # Handle device errors with specific instructions
            title, message = get_error_message_with_solution(e)
            self.show_error(title, message)
            
            self.device_combo['values'] = ["❌ Lỗi tải thiết bị"]
            self.device_combo.current(0)
            self.record_button.config(state=tk.DISABLED)
            self.update_status(f"✗ {title}\n\nVui lòng kiểm tra và thử lại.")
    
    def refresh_devices(self):
        """Refresh device list.
        
        Implements graceful degradation - allows user to retry after fixing device issues.
        """
        if self.is_recording:
            self.show_warning(
                "Không thể làm mới",
                "Không thể làm mới danh sách thiết bị trong khi đang thu âm."
            )
            return
        
        try:
            # Reinitialize coordinator to refresh device list
            self.coordinator = AudioRecordingCoordinator(
                config=self.config,
                gui_callback=self.on_audio_update
            )
            
            # Reload devices
            self.load_devices()
            
            if self.available_devices:
                self.show_info(
                    "Làm Mới Thành Công",
                    f"Đã tìm thấy {len(self.available_devices)} thiết bị microphone."
                )
            
        except Exception as e:
            title, message = get_error_message_with_solution(e)
            self.show_error(title, message)
    
    def on_device_selected(self, event):
        """Handle device selection change."""
        if not self.coordinator or self.is_recording:
            return
        
        try:
            selected_idx = self.device_combo.current()
            if 0 <= selected_idx < len(self.available_devices):
                device = self.available_devices[selected_idx]
                self.coordinator.set_device(device.id)
                self.update_status(
                    f"✓ Đã chọn thiết bị: {device.name}\n\n"
                    f"ID: {device.id}\n"
                    f"Channels: {device.channels}\n"
                    f"Sample Rate: {device.default_sample_rate} Hz"
                )
        except (StateError, DeviceError) as e:
            title, message = get_error_message_with_solution(e)
            self.show_error(title, message)
        except Exception as e:
            title, message = get_error_message_with_solution(e)
            self.show_error(title, message)
    
    def toggle_recording(self):
        """Toggle recording on/off."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def start_recording(self):
        """Start audio recording.
        
        Implements Requirements 1.5, 10.1, 10.2, 10.3 - Error handling with user notifications
        """
        if not self.coordinator:
            self.show_error("Lỗi", "Coordinator chưa được khởi tạo!")
            return
        
        try:
            # Check disk space before starting
            has_space, available_mb = check_disk_space(self.config.output_dir, required_mb=100)
            if not has_space:
                self.show_error(
                    "Ổ Đĩa Đầy",
                    f"Không đủ dung lượng để thu âm.\n\n"
                    f"Dung lượng còn lại: {available_mb} MB\n"
                    f"Cần ít nhất: 100 MB\n\n"
                    "Vui lòng giải phóng dung lượng hoặc chọn thư mục khác."
                )
                return
            
            # Update config with current settings
            sample_rate_str = self.sample_rate_var.get().replace(" Hz", "")
            self.config.sample_rate = int(sample_rate_str)
            
            bit_depth_str = self.bit_depth_var.get().replace("-bit", "")
            self.config.bit_depth = int(bit_depth_str)
            
            # Start recording
            self.coordinator.start_recording()
            
            # Update UI state
            self.is_recording = True
            self.record_button.config(
                text="⏹️ DỪNG THU ÂM",
                bg='#1976d2',
                activebackground='#0d47a1'
            )
            self.pause_button.config(state=tk.NORMAL)
            
            # Disable settings during recording
            self.device_combo.config(state=tk.DISABLED)
            self.sample_rate_combo.config(state=tk.DISABLED)
            self.bit_depth_combo.config(state=tk.DISABLED)
            self.settings_button.config(state=tk.DISABLED)
            
            self.update_status(
                "⏺️ ĐANG THU ÂM...\n\n"
                f"Sample Rate: {self.config.sample_rate} Hz\n"
                f"Bit Depth: {self.config.bit_depth}-bit\n"
                f"Channels: {self.config.channels} (Mono)\n"
                f"Thư mục: {self.config.output_dir}\n"
                f"Dung lượng còn lại: {available_mb} MB"
            )
            
        except (StateError, DeviceError, FileSystemError, AudioRecordingError) as e:
            # Use enhanced error handling with solutions
            title, message = get_error_message_with_solution(e)
            self.show_error(title, message)
            self.update_status(f"✗ {title}\n\nVui lòng thử lại.")
            
        except Exception as e:
            # Fallback for unexpected errors
            title, message = get_error_message_with_solution(e)
            self.show_error(title, message)
            self.update_status(f"✗ Lỗi: {str(e)}")
    
    def stop_recording(self):
        """Stop audio recording.
        
        Implements Requirement 1.5 - Success notifications for completed recordings
        """
        if not self.coordinator:
            return
        
        try:
            # Stop recording and get file path
            file_path = self.coordinator.stop_recording()
            self.current_file_path = file_path
            
            # Get file size for success message
            file_size = Path(file_path).stat().st_size
            file_size_str = format_file_size(file_size)
            
            # Update UI state
            self.is_recording = False
            self.is_paused = False
            self.record_button.config(
                text="⏺️ BẮT ĐẦU THU ÂM",
                bg='#d32f2f',
                activebackground='#b71c1c'
            )
            self.pause_button.config(
                text="⏸️ TẠM DỪNG",
                state=tk.DISABLED
            )
            
            # Re-enable settings
            self.device_combo.config(state='readonly')
            self.sample_rate_combo.config(state='readonly')
            self.bit_depth_combo.config(state='readonly')
            self.settings_button.config(state=tk.NORMAL)
            
            # Reset audio level
            self.level_progress['value'] = 0
            self.level_value_label.config(text="0%")
            
            # Reset time
            self.time_label.config(text="⏱️ 00:00:00")
            
            # Show success message with file details
            file_name = Path(file_path).name
            self.update_status(
                f"✓ THU ÂM HOÀN TẤT!\n\n"
                f"File đã lưu: {file_name}\n"
                f"Kích thước: {file_size_str}\n"
                f"Đường dẫn: {file_path}\n\n"
                f"Nhấn 'BẮT ĐẦU THU ÂM' để thu âm mới."
            )
            
            # Show success notification dialog
            messagebox.showinfo(
                "✓ Hoàn Thành",
                f"Thu âm hoàn tất!\n\n"
                f"📁 File: {file_name}\n"
                f"📊 Kích thước: {file_size_str}\n"
                f"📂 Đường dẫn: {file_path}\n\n"
                f"File đã được lưu thành công!"
            )
            
            # Refresh recordings list
            self.refresh_recordings_list()
            
            # Calculate duration from file
            duration_seconds = self._get_audio_duration(file_path)
            
            # Save to MongoDB
            self.save_recording_to_mongodb(file_path, duration_seconds=duration_seconds)
            
            # Auto-transcribe if transcriber is available
            if self.transcriber and self.transcriber.is_available():
                self.auto_transcribe_recording(file_path)
            
        except (StateError, FileSystemError, AudioRecordingError) as e:
            # Use enhanced error handling
            title, message = get_error_message_with_solution(e)
            self.show_error(title, message)
            self.update_status(f"✗ {title}\n\nVui lòng thử lại.")
            
        except Exception as e:
            # Fallback for unexpected errors
            title, message = get_error_message_with_solution(e)
            self.show_error(title, message)
            self.update_status(f"✗ Lỗi: {str(e)}")
    
    def toggle_pause(self):
        """Toggle pause/resume."""
        if not self.coordinator or not self.is_recording:
            return
        
        try:
            if self.is_paused:
                # Resume
                self.coordinator.resume_recording()
                self.is_paused = False
                self.pause_button.config(
                    text="⏸️ TẠM DỪNG",
                    bg='#ff9800',
                    activebackground='#f57c00'
                )
                self.update_status("▶️ Đã tiếp tục thu âm...")
            else:
                # Pause
                self.coordinator.pause_recording()
                self.is_paused = True
                self.pause_button.config(
                    text="▶️ TIẾP TỤC",
                    bg='#4CAF50',
                    activebackground='#388e3c'
                )
                self.update_status("⏸️ Đã tạm dừng thu âm\n\nNhấn 'TIẾP TỤC' để tiếp tục.")
                
        except (StateError, AudioRecordingError) as e:
            title, message = get_error_message_with_solution(e)
            self.show_error(title, message)
        except Exception as e:
            title, message = get_error_message_with_solution(e)
            self.show_error(title, message)
    
    def on_audio_update(self, level: float, elapsed_time: float):
        """Handle audio level and time updates from coordinator.
        
        Args:
            level: Audio level (0.0 to 1.0)
            elapsed_time: Elapsed time in seconds
        """
        # Update on main thread
        self.parent.after(0, lambda: self._update_audio_display(level, elapsed_time))
    
    def _update_audio_display(self, level: float, elapsed_time: float):
        """Update audio level and time display (runs on main thread).
        
        Args:
            level: Audio level (0.0 to 1.0)
            elapsed_time: Elapsed time in seconds
        """
        # Update audio level progress bar
        level_percent = int(level * 100)
        self.level_progress['value'] = level_percent
        self.level_value_label.config(text=f"{level_percent}%")
        
        # Update elapsed time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        time_str = f"⏱️ {hours:02d}:{minutes:02d}:{seconds:02d}"
        self.time_label.config(text=time_str)
    
    def update_status(self, text: str):
        """Update status text display.
        
        Args:
            text: Status text to display
        """
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, text)
        self.status_text.config(state=tk.DISABLED)
    
    def show_error(self, title: str, message: str):
        """Show error message dialog.
        
        Args:
            title: Error dialog title
            message: Error message
        """
        messagebox.showerror(title, message)
    
    def show_warning(self, title: str, message: str):
        """Show warning message dialog.
        
        Args:
            title: Warning dialog title
            message: Warning message
        """
        messagebox.showwarning(title, message)
    
    def show_info(self, title: str, message: str):
        """Show info message dialog.
        
        Args:
            title: Info dialog title
            message: Info message
        """
        messagebox.showinfo(title, message)
    
    def refresh_recordings_list(self):
        """Refresh the recordings list from disk.
        
        Implements Requirements 9.1, 9.2
        """
        if not self.recording_manager:
            return
        
        try:
            # Clear existing items
            for item in self.recordings_tree.get_children():
                self.recordings_tree.delete(item)
            
            # Get recordings
            recordings = self.recording_manager.list_recordings()
            
            # Populate tree
            for recording in recordings:
                # Format date
                date_str = recording.created_at.strftime("%Y-%m-%d %H:%M:%S")
                
                # Format duration
                duration_mins = int(recording.duration // 60)
                duration_secs = int(recording.duration % 60)
                duration_str = f"{duration_mins:02d}:{duration_secs:02d}"
                
                # Format file size
                size_kb = recording.file_size / 1024
                if size_kb < 1024:
                    size_str = f"{size_kb:.1f} KB"
                else:
                    size_mb = size_kb / 1024
                    size_str = f"{size_mb:.1f} MB"
                
                # Insert into tree
                self.recordings_tree.insert(
                    "",
                    tk.END,
                    values=(recording.file_name, date_str, duration_str, size_str),
                    tags=(str(recording.file_path),)
                )
            
            # Enable open folder button if recordings exist
            if recordings:
                self.open_folder_button.config(state=tk.NORMAL)
            else:
                self.open_folder_button.config(state=tk.DISABLED)
                
        except Exception as e:
            self.show_error("Lỗi", f"Không thể tải danh sách bản thu:\n\n{str(e)}")
    
    def on_recording_selected(self, event):
        """Handle recording selection in tree.
        
        Implements Requirements 8.4, 9.3
        """
        selection = self.recordings_tree.selection()
        
        if selection:
            # Enable action buttons
            self.play_button.config(state=tk.NORMAL)
            self.delete_button.config(state=tk.NORMAL)
            self.export_button.config(state=tk.NORMAL)
            
            # Enable transcribe button if transcriber is available
            if self.transcriber and self.transcriber.is_available() and not self.is_transcribing:
                self.transcribe_button.config(state=tk.NORMAL)
            
            # Get file path from tags
            item = selection[0]
            tags = self.recordings_tree.item(item, 'tags')
            if tags:
                file_path = tags[0]
                self.file_path_label.config(text=f"📄 {file_path}")
        else:
            # Disable action buttons
            self.play_button.config(state=tk.DISABLED)
            self.delete_button.config(state=tk.DISABLED)
            self.export_button.config(state=tk.DISABLED)
            self.transcribe_button.config(state=tk.DISABLED)
            self.file_path_label.config(text="")
    
    def play_recording(self):
        """Play selected recording.
        
        Implements Requirement 9.4
        """
        selection = self.recordings_tree.selection()
        if not selection:
            return
        
        # Get file path
        item = selection[0]
        tags = self.recordings_tree.item(item, 'tags')
        if not tags:
            return
        
        file_path = tags[0]
        
        try:
            # Stop any currently playing audio
            if self.audio_player:
                self.audio_player.stop()
                self.audio_player = None
            
            if AUDIO_PLAYER_AVAILABLE:
                # Use VideoAudioPlayer to play WAV file
                self.audio_player = VideoAudioPlayer(file_path)
                self.audio_player.start()
                
                messagebox.showinfo(
                    "Đang phát",
                    f"Đang phát: {Path(file_path).name}\n\n"
                    f"Thời lượng: {self.audio_player.duration:.1f}s"
                )
            else:
                # Fallback: open with system default player
                self.open_with_default_player(file_path)
                
        except Exception as e:
            self.show_error("Lỗi Phát", f"Không thể phát file:\n\n{str(e)}")
    
    def open_with_default_player(self, file_path: str):
        """Open audio file with system default player.
        
        Args:
            file_path: Path to audio file
        """
        try:
            system = platform.system()
            
            if system == "Windows":
                os.startfile(file_path)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", file_path])
            else:  # Linux
                subprocess.run(["xdg-open", file_path])
                
        except Exception as e:
            raise Exception(f"Không thể mở file: {e}")
    
    def delete_recording(self):
        """Delete selected recording.
        
        Implements Requirement 9.5
        """
        selection = self.recordings_tree.selection()
        if not selection:
            return
        
        # Get file info
        item = selection[0]
        values = self.recordings_tree.item(item, 'values')
        tags = self.recordings_tree.item(item, 'tags')
        
        if not tags:
            return
        
        file_name = values[0]
        file_path = Path(tags[0])
        
        # Confirm deletion
        result = messagebox.askyesno(
            "Xác nhận xóa",
            f"Bạn có chắc muốn xóa bản thu này?\n\n"
            f"File: {file_name}\n\n"
            f"Hành động này không thể hoàn tác!"
        )
        
        if not result:
            return
        
        try:
            # Stop audio if playing this file
            if self.audio_player:
                self.audio_player.stop()
                self.audio_player = None
            
            # Delete file
            self.recording_manager.delete_recording(file_path)
            
            # Refresh list
            self.refresh_recordings_list()
            
            messagebox.showinfo("Thành công", f"Đã xóa: {file_name}")
            
        except Exception as e:
            self.show_error("Lỗi Xóa", f"Không thể xóa file:\n\n{str(e)}")
    
    def export_recording(self):
        """Export selected recording to a user-chosen location.
        
        Implements Requirement 9.3 (Export button)
        """
        selection = self.recordings_tree.selection()
        if not selection:
            return
        
        # Get file info
        item = selection[0]
        values = self.recordings_tree.item(item, 'values')
        tags = self.recordings_tree.item(item, 'tags')
        
        if not tags:
            return
        
        file_name = values[0]
        source_path = Path(tags[0])
        
        try:
            # Get default filename (without extension)
            default_name = Path(file_name).stem
            default_dir = str(Path("./recordings").absolute())
            
            # Ask user where to save with custom dialog
            destination = ask_save_file(
                parent=self.parent,
                title="Xuất Bản Thu",
                default_filename=default_name,
                default_dir=default_dir,
                file_extension=".wav",
                file_types=[
                    ("WAV files", "*.wav"),
                    ("All files", "*.*")
                ]
            )
            
            if not destination:
                # User cancelled
                return
            
            # Copy file to destination
            shutil.copy2(source_path, destination)
            
            messagebox.showinfo(
                "Thành công",
                f"Đã xuất bản thu!\n\n"
                f"Từ: {file_name}\n"
                f"Đến: {destination}"
            )
            
        except Exception as e:
            self.show_error("Lỗi Xuất", f"Không thể xuất file:\n\n{str(e)}")
    
    def open_recordings_folder(self):
        """Open recordings folder in file explorer.
        
        Implements Requirement 8.5
        """
        if not self.config:
            return
        
        try:
            recordings_dir = self.config.output_dir
            
            # Create directory if it doesn't exist
            recordings_dir.mkdir(parents=True, exist_ok=True)
            
            # Open in file explorer
            system = platform.system()
            
            if system == "Windows":
                os.startfile(str(recordings_dir))
            elif system == "Darwin":  # macOS
                subprocess.run(["open", str(recordings_dir)])
            else:  # Linux
                subprocess.run(["xdg-open", str(recordings_dir)])
                
        except Exception as e:
            self.show_error("Lỗi", f"Không thể mở thư mục:\n\n{str(e)}")
    
    def open_settings_dialog(self):
        """Open settings dialog.
        
        Implements Requirements 3.1, 3.2, 5.1, 5.2, 5.3
        """
        if self.is_recording:
            messagebox.showwarning(
                "Không thể thay đổi",
                "Không thể thay đổi cài đặt trong khi đang thu âm.\n\n"
                "Vui lòng dừng thu âm trước."
            )
            return
        
        try:
            # Open settings dialog
            dialog = AudioSettingsDialog(
                parent=self.parent.winfo_toplevel(),
                current_config=self.config,
                available_devices=self.available_devices,
                on_save_callback=self.on_settings_saved
            )
            
            result = dialog.show()
            
            if result:
                # Settings were saved
                self.config = result
                self.update_settings_ui()
                
                # Reinitialize coordinator with new config
                self.coordinator = AudioRecordingCoordinator(
                    config=self.config,
                    gui_callback=self.on_audio_update
                )
                
                # Reinitialize recording manager
                self.recording_manager = RecordingManager(self.config.output_dir)
                
                self.update_status(
                    "✓ Cài đặt đã được cập nhật\n\n"
                    f"Sample Rate: {self.config.sample_rate} Hz\n"
                    f"Bit Depth: {self.config.bit_depth}-bit\n"
                    f"Thư mục: {self.config.output_dir}"
                )
        
        except Exception as e:
            self.show_error("Lỗi", f"Không thể mở cài đặt:\n\n{str(e)}")
    
    def on_settings_saved(self, new_config: AudioRecordingConfig):
        """Callback when settings are saved.
        
        Args:
            new_config: New configuration
        """
        # This is called from the dialog before it closes
        pass
    
    def update_settings_ui(self):
        """Update UI to reflect current settings."""
        if not self.config:
            return
        
        # Update sample rate combo
        sample_rate_str = f"{self.config.sample_rate} Hz"
        self.sample_rate_var.set(sample_rate_str)
        
        # Update bit depth combo
        bit_depth_str = f"{self.config.bit_depth}-bit"
        self.bit_depth_var.set(bit_depth_str)
        
        # Update device selection if device_id is set
        if self.config.device_id is not None:
            for i, dev in enumerate(self.available_devices):
                if dev.id == self.config.device_id:
                    self.device_combo.current(i)
                    break
    
    def auto_transcribe_recording(self, file_path: str):
        """
        Automatically transcribe recording after it's saved.
        
        Args:
            file_path: Path to audio file
        """
        if not self.transcriber or self.is_transcribing:
            return
        
        try:
            self.is_transcribing = True
            
            # Update status
            self.update_status(
                "🔄 Đang tự động chuyển đổi sang text...\n\n"
                "Vui lòng đợi, quá trình này có thể mất vài phút."
            )
            
            # Disable buttons during transcription
            self.record_button.config(state=tk.DISABLED)
            
            # Transcribe in background (using after to not block UI)
            def do_transcribe():
                def progress_update(message):
                    # Update status on main thread
                    self.parent.after(0, lambda: self.update_status(message))
                
                result = self.transcriber.transcribe_file(
                    file_path,
                    progress_callback=progress_update
                )
                
                # Re-enable buttons on main thread
                self.parent.after(0, lambda: self._transcribe_complete(result, file_path))
            
            # Run in separate thread to not block UI
            import threading
            thread = threading.Thread(target=do_transcribe, daemon=True)
            thread.start()
            
        except Exception as e:
            logger.error(f"Auto-transcribe failed: {e}")
            self.is_transcribing = False
            self.record_button.config(state=tk.NORMAL)
    
    def transcribe_recording(self):
        """
        Manually transcribe selected recording.
        """
        selection = self.recordings_tree.selection()
        if not selection:
            return
        
        # Get file path
        item = selection[0]
        tags = self.recordings_tree.item(item, 'tags')
        if not tags:
            return
        
        file_path = tags[0]
        
        if not self.transcriber or self.is_transcribing:
            messagebox.showwarning(
                "Không thể chuyển đổi",
                "Transcriber chưa sẵn sàng hoặc đang xử lý file khác."
            )
            return
        
        # Check if transcription already exists
        text_path = Path(file_path).with_suffix('.txt')
        if text_path.exists():
            result = messagebox.askyesno(
                "File text đã tồn tại",
                f"File text đã tồn tại:\n{text_path.name}\n\n"
                "Bạn có muốn chuyển đổi lại không?"
            )
            if not result:
                return
        
        try:
            self.is_transcribing = True
            
            # Disable buttons
            self.transcribe_button.config(state=tk.DISABLED)
            self.record_button.config(state=tk.DISABLED)
            
            # Update status
            self.update_status(
                f"🔄 Đang chuyển đổi: {Path(file_path).name}\n\n"
                "Vui lòng đợi, quá trình này có thể mất vài phút..."
            )
            
            # Transcribe in background
            def do_transcribe():
                def progress_update(message):
                    self.parent.after(0, lambda: self.update_status(message))
                
                result = self.transcriber.transcribe_file(
                    file_path,
                    progress_callback=progress_update
                )
                
                self.parent.after(0, lambda: self._transcribe_complete(result, file_path))
            
            import threading
            thread = threading.Thread(target=do_transcribe, daemon=True)
            thread.start()
            
        except Exception as e:
            logger.error(f"Manual transcribe failed: {e}")
            self.show_error("Lỗi", f"Không thể chuyển đổi:\n\n{str(e)}")
            self.is_transcribing = False
            self.transcribe_button.config(state=tk.NORMAL)
            self.record_button.config(state=tk.NORMAL)
    
    def _transcribe_complete(self, result: Optional[str], file_path: str):
        """
        Handle transcription completion.
        
        Args:
            result: Transcribed text or None if failed
            file_path: Path to audio file
        """
        self.is_transcribing = False
        
        # Re-enable buttons
        self.record_button.config(state=tk.NORMAL)
        self.transcribe_button.config(state=tk.NORMAL)
        
        if result is not None:
            # Success
            text_path = Path(file_path).with_suffix('.txt')
            
            # Save transcription to MongoDB if we have a recording ID
            if self.current_recording_id:
                self.save_transcription_to_mongodb(self.current_recording_id, result)
            
            # Show result in status
            preview = result[:200] + "..." if len(result) > 200 else result
            mongodb_status = "✅ Đã lưu vào MongoDB" if self.current_recording_id else "⚠️ Chưa lưu vào MongoDB"
            self.update_status(
                f"✅ Chuyển đổi hoàn tất!\n\n"
                f"📝 Nội dung:\n{preview}\n\n"
                f"💾 Đã lưu vào: {text_path.name}\n"
                f"🗄️ {mongodb_status}"
            )
            
            # Show success dialog
            messagebox.showinfo(
                "✓ Hoàn Thành",
                f"Chuyển đổi thành công!\n\n"
                f"📁 File audio: {Path(file_path).name}\n"
                f"📝 File text: {text_path.name}\n\n"
                f"Nội dung: {len(result)} ký tự\n"
                f"{mongodb_status}"
            )
        else:
            # Failed
            self.update_status(
                "❌ Chuyển đổi thất bại\n\n"
                "Vui lòng kiểm tra:\n"
                "• File audio có giọng nói rõ ràng\n"
                "• Whisper engine đã được cài đặt\n"
                "• Xem log để biết chi tiết"
            )
    
    def _get_audio_duration(self, file_path: str) -> float:
        """
        Get audio duration from WAV file.
        
        Args:
            file_path: Path to WAV file
            
        Returns:
            Duration in seconds
        """
        try:
            import wave
            with wave.open(str(file_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate)
                return duration
        except Exception as e:
            logger.error(f"Error getting audio duration: {e}")
            return 0.0
    
    def save_recording_to_mongodb(self, file_path: str, duration_seconds: float):
        """
        Save recording information to MongoDB.
        
        Args:
            file_path: Path to audio file
            duration_seconds: Duration in seconds
        """
        if not self.mongodb_manager or not self.mongodb_manager.is_connected():
            logger.warning("MongoDB not connected - skipping database save")
            return
        
        try:
            file_path_obj = Path(file_path)
            file_size = file_path_obj.stat().st_size
            
            # Save to MongoDB
            recording_id = self.mongodb_manager.save_audio_recording(
                file_path=str(file_path_obj.absolute()),
                duration_seconds=duration_seconds,
                sample_rate=self.config.sample_rate,
                bit_depth=self.config.bit_depth,
                channels=self.config.channels,
                file_size_bytes=file_size,
                metadata={
                    "device_name": self.device_var.get() if self.device_var else "Unknown",
                    "app_version": "1.0.0"
                }
            )
            
            if recording_id:
                self.current_recording_id = recording_id
                logger.info(f"✅ Saved recording to MongoDB: {recording_id}")
            else:
                logger.warning("Failed to save recording to MongoDB")
                
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {e}")
    
    def save_transcription_to_mongodb(self, recording_id: str, transcription_text: str):
        """
        Save transcription to MongoDB.
        
        Args:
            recording_id: MongoDB recording ID
            transcription_text: Transcribed text
        """
        if not self.mongodb_manager or not self.mongodb_manager.is_connected():
            logger.warning("MongoDB not connected - skipping transcription save")
            return
        
        try:
            transcription_id = self.mongodb_manager.save_transcription(
                recording_id=recording_id,
                transcription_text=transcription_text,
                language="vi",
                metadata={
                    "model": "whisper-base",
                    "app_version": "1.0.0"
                }
            )
            
            if transcription_id:
                logger.info(f"✅ Saved transcription to MongoDB: {transcription_id}")
            else:
                logger.warning("Failed to save transcription to MongoDB")
                
        except Exception as e:
            logger.error(f"Error saving transcription to MongoDB: {e}")
    
    def cleanup(self):
        """Clean up resources when tab is closed."""
        # Stop audio player if playing
        if self.audio_player:
            try:
                self.audio_player.stop()
                self.audio_player = None
            except Exception as e:
                print(f"Error stopping audio player: {e}")
        
        # Stop recording if active
        if self.coordinator:
            try:
                if self.is_recording:
                    self.coordinator.stop_recording()
                self.coordinator.cleanup()
            except Exception as e:
                print(f"Error during cleanup: {e}")
        
        # Disconnect MongoDB
        if self.mongodb_manager:
            try:
                self.mongodb_manager.disconnect()
            except Exception as e:
                print(f"Error disconnecting MongoDB: {e}")


# Example usage for testing
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Audio Recording Test")
    root.geometry("900x700")
    root.configure(bg='#1a1a1a')
    
    # Create tab
    tab = AudioRecordingTab(root)
    
    # Handle window close
    def on_closing():
        tab.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.mainloop()
