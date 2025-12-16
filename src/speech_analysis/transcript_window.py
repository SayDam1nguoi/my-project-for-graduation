# -*- coding: utf-8 -*-
"""
Transcript Window - Cửa sổ hiển thị văn bản chuyển đổi từ giọng nói.

Module này cung cấp giao diện để hiển thị transcript real-time,
điểm chất lượng giọng nói, và các controls để lưu/xóa transcript.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
from typing import Optional, Callable
import re


class TranscriptWindow:
    """Cửa sổ hiển thị transcript với quality scores và controls."""
    
    # Styling constants
    STYLE = {
        'bg_color': '#1e1e1e',
        'text_color': '#ffffff',
        'timestamp_color': '#9E9E9E',
        'unclear_color': '#ff9800',
        'new_text_color': '#4CAF50',
        'font_family': 'Consolas',
        'font_size': 10,
        'line_spacing': 1.5,
        'header_bg': '#2d2d2d',
        'footer_bg': '#2d2d2d',
        'button_bg': '#424242',
        'button_fg': '#ffffff',
        'button_hover': '#616161'
    }
    
    def __init__(self, parent: tk.Tk, on_save_callback: Optional[Callable] = None):
        """
        Khởi tạo Transcript Window.
        
        Args:
            parent: Cửa sổ cha (Tkinter root)
            on_save_callback: Callback khi user click nút Lưu
        """
        self.parent = parent
        self.on_save_callback = on_save_callback
        
        # Create toplevel window
        self.window = tk.Toplevel(parent)
        self.window.title("Transcript - Phiên Ghi Âm")
        self.window.geometry("700x600")
        self.window.minsize(600, 400)
        self.window.configure(bg=self.STYLE['bg_color'])
        
        # Session data
        self.session_start = datetime.now()
        self.word_count = 0
        self.clarity_score = 0.0
        self.fluency_score = 0.0
        
        # Auto-scroll control
        self.auto_scroll_enabled = True
        self.user_scrolled = False
        
        # Track if content has been modified
        self.has_content = False
        
        # Build UI
        self._create_widgets()
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Initially hide the window
        self.window.withdraw()
    
    def _create_widgets(self):
        """Tạo các widgets cho giao diện."""
        # Header frame
        self._create_header()
        
        # Quality bars frame
        self._create_quality_bars()
        
        # Text area frame
        self._create_text_area()
        
        # Footer frame
        self._create_footer()
    
    def _create_header(self):
        """Tạo header với timestamp."""
        header_frame = tk.Frame(
            self.window,
            bg=self.STYLE['header_bg'],
            height=40
        )
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        # Session timestamp label
        self.timestamp_label = tk.Label(
            header_frame,
            text=f"Phiên Ghi Âm - {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}",
            font=(self.STYLE['font_family'], 11, 'bold'),
            bg=self.STYLE['header_bg'],
            fg=self.STYLE['text_color']
        )
        self.timestamp_label.pack(pady=10)
    
    def _create_quality_bars(self):
        """Tạo progress bars cho Clarity và Fluency scores."""
        quality_frame = tk.Frame(
            self.window,
            bg=self.STYLE['header_bg'],
            height=60
        )
        quality_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        quality_frame.pack_propagate(False)
        
        # Clarity score
        clarity_container = tk.Frame(quality_frame, bg=self.STYLE['header_bg'])
        clarity_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.clarity_label = tk.Label(
            clarity_container,
            text="Clarity: 0/100",
            font=(self.STYLE['font_family'], 9),
            bg=self.STYLE['header_bg'],
            fg=self.STYLE['text_color'],
            anchor='w'
        )
        self.clarity_label.pack(fill=tk.X)
        
        self.clarity_bar = ttk.Progressbar(
            clarity_container,
            orient='horizontal',
            length=200,
            mode='determinate',
            maximum=100
        )
        self.clarity_bar.pack(fill=tk.X, pady=(2, 0))
        
        # Fluency score
        fluency_container = tk.Frame(quality_frame, bg=self.STYLE['header_bg'])
        fluency_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.fluency_label = tk.Label(
            fluency_container,
            text="Fluency: 0/100",
            font=(self.STYLE['font_family'], 9),
            bg=self.STYLE['header_bg'],
            fg=self.STYLE['text_color'],
            anchor='w'
        )
        self.fluency_label.pack(fill=tk.X)
        
        self.fluency_bar = ttk.Progressbar(
            fluency_container,
            orient='horizontal',
            length=200,
            mode='determinate',
            maximum=100
        )
        self.fluency_bar.pack(fill=tk.X, pady=(2, 0))
        
        # Configure progress bar styles
        self._configure_progressbar_styles()
    
    def _configure_progressbar_styles(self):
        """Cấu hình styles cho progress bars với màu động."""
        style = ttk.Style()
        
        # Default style
        style.theme_use('default')
        
        # Green style (>80)
        style.configure(
            "green.Horizontal.TProgressbar",
            troughcolor=self.STYLE['header_bg'],
            bordercolor=self.STYLE['header_bg'],
            background='#4CAF50',
            lightcolor='#4CAF50',
            darkcolor='#4CAF50'
        )
        
        # Yellow style (60-80)
        style.configure(
            "yellow.Horizontal.TProgressbar",
            troughcolor=self.STYLE['header_bg'],
            bordercolor=self.STYLE['header_bg'],
            background='#FFC107',
            lightcolor='#FFC107',
            darkcolor='#FFC107'
        )
        
        # Red style (<60)
        style.configure(
            "red.Horizontal.TProgressbar",
            troughcolor=self.STYLE['header_bg'],
            bordercolor=self.STYLE['header_bg'],
            background='#f44336',
            lightcolor='#f44336',
            darkcolor='#f44336'
        )
    
    def _create_text_area(self):
        """Tạo text area với scrollbar."""
        text_frame = tk.Frame(self.window, bg=self.STYLE['bg_color'])
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text widget
        self.text_area = tk.Text(
            text_frame,
            wrap=tk.WORD,
            font=(self.STYLE['font_family'], self.STYLE['font_size']),
            bg=self.STYLE['bg_color'],
            fg=self.STYLE['text_color'],
            insertbackground=self.STYLE['text_color'],
            selectbackground='#424242',
            selectforeground=self.STYLE['text_color'],
            yscrollcommand=scrollbar.set,
            spacing1=2,
            spacing3=2,
            padx=10,
            pady=10
        )
        self.text_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.text_area.yview)
        
        # Configure text tags for styling
        self.text_area.tag_configure(
            'timestamp',
            foreground=self.STYLE['timestamp_color']
        )
        self.text_area.tag_configure(
            'new_text',
            foreground=self.STYLE['new_text_color']
        )
        self.text_area.tag_configure(
            'unclear',
            foreground=self.STYLE['unclear_color'],
            font=(self.STYLE['font_family'], self.STYLE['font_size'], 'italic')
        )
        
        # Bind scroll events to detect user scrolling
        self.text_area.bind('<MouseWheel>', self._on_user_scroll)
        self.text_area.bind('<Button-4>', self._on_user_scroll)  # Linux scroll up
        self.text_area.bind('<Button-5>', self._on_user_scroll)  # Linux scroll down
        self.text_area.bind('<Key>', self._on_user_scroll)
        
        # Make text area read-only
        self.text_area.config(state=tk.DISABLED)
    
    def _create_footer(self):
        """Tạo footer với stats và buttons."""
        footer_frame = tk.Frame(
            self.window,
            bg=self.STYLE['footer_bg'],
            height=50
        )
        footer_frame.pack(fill=tk.X, padx=0, pady=0)
        footer_frame.pack_propagate(False)
        
        # Stats container (left side)
        stats_frame = tk.Frame(footer_frame, bg=self.STYLE['footer_bg'])
        stats_frame.pack(side=tk.LEFT, padx=15, pady=10)
        
        self.duration_label = tk.Label(
            stats_frame,
            text="Thời lượng: 00:00:00",
            font=(self.STYLE['font_family'], 9),
            bg=self.STYLE['footer_bg'],
            fg=self.STYLE['text_color']
        )
        self.duration_label.pack(side=tk.LEFT, padx=(0, 15))
        
        self.word_count_label = tk.Label(
            stats_frame,
            text="Số từ: 0",
            font=(self.STYLE['font_family'], 9),
            bg=self.STYLE['footer_bg'],
            fg=self.STYLE['text_color']
        )
        self.word_count_label.pack(side=tk.LEFT)
        
        # Buttons container (right side)
        buttons_frame = tk.Frame(footer_frame, bg=self.STYLE['footer_bg'])
        buttons_frame.pack(side=tk.RIGHT, padx=15, pady=10)
        
        # Save button
        self.save_button = tk.Button(
            buttons_frame,
            text="Lưu",
            font=(self.STYLE['font_family'], 9, 'bold'),
            bg='#4CAF50',
            fg=self.STYLE['button_fg'],
            activebackground='#45a049',
            command=self._on_save,
            width=8,
            cursor='hand2',
            relief=tk.FLAT
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Clear button
        self.clear_button = tk.Button(
            buttons_frame,
            text="Xóa",
            font=(self.STYLE['font_family'], 9, 'bold'),
            bg='#f44336',
            fg=self.STYLE['button_fg'],
            activebackground='#da190b',
            command=self._on_clear,
            width=8,
            cursor='hand2',
            relief=tk.FLAT
        )
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # Close button
        self.close_button = tk.Button(
            buttons_frame,
            text="Đóng",
            font=(self.STYLE['font_family'], 9, 'bold'),
            bg=self.STYLE['button_bg'],
            fg=self.STYLE['button_fg'],
            activebackground=self.STYLE['button_hover'],
            command=self._on_close,
            width=8,
            cursor='hand2',
            relief=tk.FLAT
        )
        self.close_button.pack(side=tk.LEFT, padx=5)
        
        # Start duration update timer
        self._update_duration()
    
    def _on_user_scroll(self, event):
        """Xử lý khi user scroll."""
        # Check if user scrolled up (away from bottom)
        self.user_scrolled = True
        
        # Schedule check if user is at bottom
        self.window.after(100, self._check_scroll_position)
    
    def _check_scroll_position(self):
        """Kiểm tra vị trí scroll và enable/disable auto-scroll."""
        # Get current scroll position
        yview = self.text_area.yview()
        
        # If at bottom (within 5% threshold), re-enable auto-scroll
        if yview[1] >= 0.95:
            self.auto_scroll_enabled = True
            self.user_scrolled = False
        else:
            self.auto_scroll_enabled = False
    
    def _update_duration(self):
        """Cập nhật thời lượng phiên."""
        if self.window.winfo_exists():
            duration = datetime.now() - self.session_start
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            seconds = duration.seconds % 60
            
            self.duration_label.config(
                text=f"Thời lượng: {hours:02d}:{minutes:02d}:{seconds:02d}"
            )
            
            # Update every second
            self.window.after(1000, self._update_duration)
    
    def show(self):
        """Hiển thị cửa sổ."""
        self.window.deiconify()
        self.window.lift()
        self.window.focus_force()
    
    def hide(self):
        """Ẩn cửa sổ."""
        self.window.withdraw()
    
    def destroy(self):
        """Đóng và hủy cửa sổ."""
        if self.window.winfo_exists():
            self.window.destroy()
    
    def append_text(self, text: str, timestamp: Optional[datetime] = None):
        """
        Thêm văn bản mới với timestamp.
        
        Args:
            text: Văn bản cần thêm
            timestamp: Thời điểm (mặc định là hiện tại)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Format timestamp
        time_str = timestamp.strftime("%H:%M:%S")
        
        # Enable editing temporarily
        self.text_area.config(state=tk.NORMAL)
        
        # Insert timestamp
        self.text_area.insert(tk.END, f"[{time_str}] ", 'timestamp')
        
        # Check for unclear parts and apply appropriate tags
        if '[không rõ]' in text:
            # Split text by unclear markers
            parts = re.split(r'(\[không rõ\])', text)
            for part in parts:
                if part == '[không rõ]':
                    self.text_area.insert(tk.END, part, ('unclear', 'new_text'))
                elif part:
                    self.text_area.insert(tk.END, part, 'new_text')
        else:
            # Insert normal text with new_text tag
            self.text_area.insert(tk.END, text, 'new_text')
        
        # Add newline
        self.text_area.insert(tk.END, '\n\n')
        
        # Update word count
        self._update_word_count(text)
        
        # Mark as having content
        self.has_content = True
        
        # Auto-scroll if enabled
        if self.auto_scroll_enabled and not self.user_scrolled:
            self.text_area.see(tk.END)
        
        # Disable editing
        self.text_area.config(state=tk.DISABLED)
        
        # Remove new_text tag after a delay to return to normal color
        self.window.after(2000, self._remove_new_text_tags)
    
    def _remove_new_text_tags(self):
        """Xóa tag new_text để văn bản trở về màu bình thường."""
        self.text_area.config(state=tk.NORMAL)
        
        # Get all text with new_text tag
        ranges = self.text_area.tag_ranges('new_text')
        
        # Remove the tag
        if ranges:
            self.text_area.tag_remove('new_text', ranges[0], ranges[-1])
        
        self.text_area.config(state=tk.DISABLED)
    
    def _update_word_count(self, text: str):
        """Cập nhật số từ."""
        # Count words (split by whitespace)
        words = text.split()
        self.word_count += len(words)
        
        self.word_count_label.config(text=f"Số từ: {self.word_count}")
    
    def update_quality_scores(self, clarity: float, fluency: float):
        """
        Cập nhật điểm chất lượng.
        
        Args:
            clarity: Điểm Clarity (0-100)
            fluency: Điểm Fluency (0-100)
        """
        self.clarity_score = clarity
        self.fluency_score = fluency
        
        # Update labels
        self.clarity_label.config(text=f"Clarity: {int(clarity)}/100")
        self.fluency_label.config(text=f"Fluency: {int(fluency)}/100")
        
        # Update progress bars
        self.clarity_bar['value'] = clarity
        self.fluency_bar['value'] = fluency
        
        # Update bar colors based on score
        clarity_style = self._get_score_style(clarity)
        fluency_style = self._get_score_style(fluency)
        
        self.clarity_bar.config(style=clarity_style)
        self.fluency_bar.config(style=fluency_style)
    
    def _get_score_style(self, score: float) -> str:
        """
        Lấy style cho progress bar dựa trên điểm.
        
        Args:
            score: Điểm (0-100)
            
        Returns:
            Tên style
        """
        if score >= 80:
            return "green.Horizontal.TProgressbar"
        elif score >= 60:
            return "yellow.Horizontal.TProgressbar"
        else:
            return "red.Horizontal.TProgressbar"
    
    def get_full_text(self) -> str:
        """
        Lấy toàn bộ văn bản.
        
        Returns:
            Toàn bộ nội dung transcript
        """
        return self.text_area.get('1.0', tk.END).strip()
    
    def clear(self):
        """Xóa toàn bộ nội dung."""
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete('1.0', tk.END)
        self.text_area.config(state=tk.DISABLED)
        
        # Reset counters
        self.word_count = 0
        self.word_count_label.config(text="Số từ: 0")
        self.has_content = False
    
    def _on_save(self):
        """Xử lý khi user click nút Lưu."""
        if not self.has_content:
            messagebox.showinfo(
                "Thông báo",
                "Không có nội dung để lưu.",
                parent=self.window
            )
            return
        
        if self.on_save_callback:
            self.on_save_callback()
        else:
            messagebox.showinfo(
                "Thông báo",
                "Chức năng lưu chưa được cấu hình.",
                parent=self.window
            )
    
    def _on_clear(self):
        """Xử lý khi user click nút Xóa."""
        if not self.has_content:
            return
        
        # Confirm before clearing
        result = messagebox.askyesno(
            "Xác nhận",
            "Bạn có chắc muốn xóa toàn bộ nội dung?",
            parent=self.window
        )
        
        if result:
            self.clear()
    
    def _on_close(self):
        """Xử lý khi user đóng cửa sổ."""
        # If has content, ask to save
        if self.has_content:
            result = messagebox.askyesnocancel(
                "Lưu transcript",
                "Bạn có muốn lưu transcript trước khi đóng?",
                parent=self.window
            )
            
            if result is None:  # Cancel
                return
            elif result:  # Yes - save
                self._on_save()
        
        # Hide window instead of destroying (can be shown again)
        self.hide()
