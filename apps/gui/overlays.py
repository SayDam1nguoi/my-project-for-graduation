# -*- coding: utf-8 -*-
"""
GUI Overlay Components

Contains overlay classes for live reports and appearance warnings.
"""

import tkinter as tk
import time
import numpy as np
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.video_analysis.appearance.models import AppearanceWarning


class LiveReportOverlay:
    """Cửa sổ overlay nhỏ hiển thị báo cáo trực tiếp khi ẩn UI chính."""
    
    def __init__(self, parent_gui):
        """
        Khởi tạo overlay window.
        
        Args:
            parent_gui: Reference to main EmotionRecognitionGUI instance
        """
        self.parent_gui = parent_gui
        self.window = None
        self.is_visible = False
        
        # Update interval (ms)
        self.update_interval = 500
        self.update_job = None
        
    def create_window(self):
        """Tạo cửa sổ overlay."""
        if self.window is not None:
            return
        
        self.window = tk.Toplevel()
        self.window.title("Báo Cáo Trực Tiếp")
        self.window.geometry("350x500")
        self.window.configure(bg='#1a1a1a')
        
        # Always on top
        self.window.attributes('-topmost', True)
        
        # Handle close event - show main UI when clicking X
        self.window.protocol("WM_DELETE_WINDOW", self.show_main_ui)
        
        # Main frame
        main_frame = tk.Frame(self.window, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        header_frame = tk.Frame(main_frame, bg='#0d47a1', height=50)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame,
            text="📊 BÁO CÁO TRỰC TIẾP",
            font=("Segoe UI", 14, "bold"),
            bg='#0d47a1',
            fg='#ffffff'
        )
        header_label.pack(pady=12)
        
        # Content frame with scrollbar
        content_frame = tk.Frame(main_frame, bg='#252525')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Text widget for report
        self.report_text = tk.Text(
            content_frame,
            font=("Consolas", 10),
            bg='#252525',
            fg='#ffffff',
            wrap=tk.WORD,
            relief=tk.FLAT,
            padx=10,
            pady=10,
            state=tk.DISABLED
        )
        self.report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(content_frame, command=self.report_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.report_text.config(yscrollcommand=scrollbar.set)
        
        # Button frame
        button_frame = tk.Frame(main_frame, bg='#1a1a1a')
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Show main UI button
        show_ui_btn = tk.Button(
            button_frame,
            text="🔄 Hiện UI Chính",
            command=self.show_main_ui,
            font=("Segoe UI", 10, "bold"),
            bg='#4CAF50',
            fg='#ffffff',
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor='hand2'
        )
        show_ui_btn.pack(fill=tk.X)
        
    def show(self):
        """Hiển thị overlay window."""
        if self.window is None:
            self.create_window()
        
        self.window.deiconify()
        self.is_visible = True
        self.start_updates()
        
    def hide(self):
        """Ẩn overlay window."""
        if self.window is not None:
            self.window.withdraw()
        self.is_visible = False
        self.stop_updates()
        
    def show_main_ui(self):
        """Hiện lại UI chính và ẩn overlay."""
        if self.parent_gui.is_running:
            self.parent_gui.update_statistics()
        
        self.hide()
        self.parent_gui.root.winfo_toplevel().deiconify()
        
    def start_updates(self):
        """Bắt đầu cập nhật báo cáo."""
        if self.is_visible:
            self.update_report()
            self.update_job = self.window.after(self.update_interval, self.start_updates)
    
    def stop_updates(self):
        """Dừng cập nhật báo cáo."""
        if self.update_job is not None:
            try:
                self.window.after_cancel(self.update_job)
            except:
                pass
            self.update_job = None
    
    def update_report(self):
        """Cập nhật nội dung báo cáo."""
        if not self.is_visible or self.window is None:
            return
        
        try:
            elapsed = time.time() - self.parent_gui.start_time if self.parent_gui.start_time else 0
            
            report = f"""
⏱️  THỜI GIAN: {elapsed:.0f}s
📹 KHUNG HÌNH: {self.parent_gui.frame_count}
👤 KHUÔN MẶT: {self.parent_gui.total_faces}

{'='*35}
😊 CẢM XÚC HIỆN TẠI
{'='*35}

"""
            
            if hasattr(self.parent_gui, 'current_emotion') and self.parent_gui.current_emotion:
                emotion_icons = {
                    'happy': '😊', 'sad': '😢', 'angry': '😠',
                    'surprise': '😲', 'fear': '😨', 'disgust': '🤢', 'neutral': '😐'
                }
                icon = emotion_icons.get(self.parent_gui.current_emotion, '😐')
                report += f"{icon} {self.parent_gui.current_emotion.upper()}\n\n"
            else:
                report += "Chưa phát hiện\n\n"
            
            report += f"""{'='*35}
📊 PHÂN BỐ CẢM XÚC
{'='*35}

"""
            
            total_emotions = sum(self.parent_gui.emotion_counts.values())
            if total_emotions > 0:
                for emotion in sorted(self.parent_gui.emotion_counts.keys()):
                    count = self.parent_gui.emotion_counts[emotion]
                    percentage = (count / total_emotions) * 100
                    bar_length = int(percentage / 5)
                    bar = "█" * bar_length
                    report += f"{emotion:10s}: {percentage:5.1f}%\n{bar}\n\n"
            else:
                report += "Chưa có dữ liệu\n\n"
            
            # Save scroll position and update
            current_yview = self.report_text.yview()
            self.report_text.config(state=tk.NORMAL)
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(1.0, report)
            
            if current_yview[0] > 0.01:
                self.report_text.yview_moveto(current_yview[0])
            
            self.report_text.config(state=tk.DISABLED)
            
        except Exception as e:
            print(f"Error updating overlay report: {e}")
    
    def destroy(self):
        """Hủy overlay window."""
        self.stop_updates()
        if self.window is not None:
            try:
                self.window.destroy()
            except:
                pass
            self.window = None


class AppearanceWarningOverlay:
    """
    Non-intrusive warning overlay for appearance assessment issues.
    
    Displays warnings for lighting and clothing issues with color-coded severity.
    Requirements: 7.1, 7.2, 7.3, 7.5
    """
    
    # Severity colors
    SEVERITY_COLORS = {
        'low': '#FFC107',      # Yellow
        'medium': '#FF9800',   # Orange
        'high': '#F44336'      # Red
    }
    
    # Warning icons
    WARNING_ICONS = {
        'lighting': '💡',
        'clothing': '👔'
    }
    
    def __init__(self, parent_canvas: tk.Label):
        """
        Initialize warning overlay.
        
        Args:
            parent_canvas: The video canvas to overlay warnings on
        """
        self.parent_canvas = parent_canvas
        self.warning_labels: Dict[str, tk.Label] = {}
        self.active_warnings: List['AppearanceWarning'] = []
        self.last_update_time = 0
        self.update_interval = 5.0  # Update every 5 seconds (Req 7.1)
    
    def update_warnings(self, warnings: List['AppearanceWarning']):
        """
        Update displayed warnings.
        
        Args:
            warnings: List of AppearanceWarning objects to display
            
        Requirements: 7.1, 7.2, 7.3
        """
        current_time = time.time()
        
        # Check if enough time has passed since last update (5 second interval)
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.last_update_time = current_time
        self.active_warnings = warnings
        
        # Clear existing warning labels
        self._clear_warnings()
        
        # Create new warning labels
        if warnings:
            self._display_warnings(warnings)
    
    def _clear_warnings(self):
        """Clear all warning labels."""
        for label in self.warning_labels.values():
            try:
                label.destroy()
            except:
                pass
        self.warning_labels.clear()
    
    def _display_warnings(self, warnings: List['AppearanceWarning']):
        """
        Display warnings as overlay labels.
        
        Args:
            warnings: List of warnings to display
            
        Requirements: 7.2, 7.3, 7.5
        """
        # Position warnings vertically with spacing
        y_offset = 10
        spacing = 80
        
        for warning in warnings:
            # Get severity color (Req 7.5: color coding)
            bg_color = self.SEVERITY_COLORS.get(warning.severity, '#FFC107')
            
            # Get warning icon
            icon = self.WARNING_ICONS.get(warning.warning_type, '⚠️')
            
            # Create warning frame
            warning_frame = tk.Frame(
                self.parent_canvas,
                bg=bg_color,
                relief=tk.RAISED,
                borderwidth=2
            )
            warning_frame.place(x=10, y=y_offset, width=300)
            
            # Warning header with icon and type
            header_text = f"{icon} {warning.warning_type.upper()} ISSUE"
            header_label = tk.Label(
                warning_frame,
                text=header_text,
                font=("Segoe UI", 10, "bold"),
                bg=bg_color,
                fg='#000000'
            )
            header_label.pack(pady=(5, 2))
            
            # Warning message
            message_label = tk.Label(
                warning_frame,
                text=warning.message,
                font=("Segoe UI", 8),
                bg=bg_color,
                fg='#000000',
                wraplength=280,
                justify=tk.LEFT
            )
            message_label.pack(pady=(0, 5), padx=5)
            
            # Recommendations (show first 2)
            if warning.recommendations:
                rec_text = "💡 " + "\n💡 ".join(warning.recommendations[:2])
                rec_label = tk.Label(
                    warning_frame,
                    text=rec_text,
                    font=("Segoe UI", 7),
                    bg=bg_color,
                    fg='#000000',
                    wraplength=280,
                    justify=tk.LEFT
                )
                rec_label.pack(pady=(0, 5), padx=5)
            
            # Store reference
            self.warning_labels[warning.warning_type] = warning_frame
            
            # Update y offset for next warning
            y_offset += spacing
    
    def clear(self):
        """Clear all warnings (Req 7.4: remove warnings when resolved)."""
        self._clear_warnings()
        self.active_warnings.clear()
    
    def destroy(self):
        """Destroy the overlay and clean up resources."""
        self._clear_warnings()
