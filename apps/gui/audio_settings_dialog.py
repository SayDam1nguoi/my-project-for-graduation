# -*- coding: utf-8 -*-
"""
Audio Settings Dialog

Provides a dialog for configuring audio recording settings.
Implements Requirements 3.1, 3.2, 5.1, 5.2, 5.3
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import sys
from typing import Optional, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.audio_recording.config import AudioRecordingConfig
from src.audio_recording.models import AudioDevice
from src.audio_recording.settings import AudioRecordingSettings


class AudioSettingsDialog:
    """
    Dialog for audio recording settings.
    
    Implements Requirements:
    - 3.1: Display available microphone devices
    - 3.2: Allow device selection and save
    - 5.1: Display sample rate options
    - 5.2: Display bit depth options and save
    - 5.3: Save all settings to config file
    """
    
    def __init__(
        self,
        parent: tk.Tk,
        current_config: AudioRecordingConfig,
        available_devices: List[AudioDevice],
        on_save_callback: Optional[callable] = None
    ):
        """Initialize settings dialog.
        
        Args:
            parent: Parent window
            current_config: Current audio recording configuration
            available_devices: List of available audio devices
            on_save_callback: Optional callback when settings are saved
        """
        self.parent = parent
        self.current_config = current_config
        self.available_devices = available_devices
        self.on_save_callback = on_save_callback
        
        # Settings manager
        self.settings_manager = AudioRecordingSettings()
        
        # Result
        self.result: Optional[AudioRecordingConfig] = None
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("⚙️ Cài đặt Thu Âm")
        self.dialog.geometry("600x500")
        self.dialog.configure(bg='#1a1a1a')
        self.dialog.resizable(False, False)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog on parent
        self.center_dialog()
        
        # Create UI
        self.create_widgets()
        
        # Load current settings
        self.load_current_settings()
    
    def center_dialog(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        
        # Get parent position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Get dialog size
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        # Calculate center position
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def create_widgets(self):
        """Create all dialog widgets."""
        # Main container
        main_container = tk.Frame(self.dialog, bg='#1a1a1a')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(main_container, bg='#0d47a1', height=50)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame,
            text="⚙️ CÀI ĐẶT THU ÂM",
            font=("Segoe UI", 16, "bold"),
            bg='#0d47a1',
            fg='#ffffff'
        )
        header_label.pack(pady=12)
        
        # Settings sections container
        settings_container = tk.Frame(main_container, bg='#252525')
        settings_container.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Device selection section (Requirements 3.1, 3.2)
        device_section = self.create_section(
            settings_container,
            "🎤 Thiết bị Microphone",
            0
        )
        
        device_label = tk.Label(
            device_section,
            text="Chọn thiết bị:",
            font=("Segoe UI", 10),
            bg='#252525',
            fg='#e0e0e0'
        )
        device_label.pack(anchor='w', pady=(0, 8))
        
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(
            device_section,
            textvariable=self.device_var,
            state='readonly',
            width=50,
            font=("Segoe UI", 10)
        )
        self.device_combo.pack(fill=tk.X, pady=(0, 5))
        
        # Populate device list
        if self.available_devices:
            device_names = [
                f"{dev.name} (ID: {dev.id})" + (" [Mặc định]" if dev.is_default else "")
                for dev in self.available_devices
            ]
            self.device_combo['values'] = device_names
        else:
            self.device_combo['values'] = ["Không có thiết bị"]
        
        # Sample rate section (Requirements 5.1, 5.2)
        sample_rate_section = self.create_section(
            settings_container,
            "📊 Sample Rate (Tần số lấy mẫu)",
            1
        )
        
        sample_rate_label = tk.Label(
            sample_rate_section,
            text="Chọn sample rate:",
            font=("Segoe UI", 10),
            bg='#252525',
            fg='#e0e0e0'
        )
        sample_rate_label.pack(anchor='w', pady=(0, 8))
        
        # Sample rate radio buttons
        self.sample_rate_var = tk.IntVar(value=16000)
        
        sample_rates = [
            (16000, "16 kHz (Giọng nói - Tiết kiệm dung lượng)"),
            (44100, "44.1 kHz (CD Quality - Cân bằng)"),
            (48000, "48 kHz (Professional - Chất lượng cao)")
        ]
        
        for rate, description in sample_rates:
            rb = tk.Radiobutton(
                sample_rate_section,
                text=description,
                variable=self.sample_rate_var,
                value=rate,
                font=("Segoe UI", 10),
                bg='#252525',
                fg='#e0e0e0',
                selectcolor='#424242',
                activebackground='#252525',
                activeforeground='#90caf9'
            )
            rb.pack(anchor='w', pady=3)
        
        # Bit depth section (Requirements 5.1, 5.2)
        bit_depth_section = self.create_section(
            settings_container,
            "🎵 Bit Depth (Độ sâu bit)",
            2
        )
        
        bit_depth_label = tk.Label(
            bit_depth_section,
            text="Chọn bit depth:",
            font=("Segoe UI", 10),
            bg='#252525',
            fg='#e0e0e0'
        )
        bit_depth_label.pack(anchor='w', pady=(0, 8))
        
        # Bit depth radio buttons
        self.bit_depth_var = tk.IntVar(value=16)
        
        bit_depths = [
            (16, "16-bit (Tiêu chuẩn - Đủ cho hầu hết mục đích)"),
            (24, "24-bit (Professional - Chất lượng cao nhất)")
        ]
        
        for depth, description in bit_depths:
            rb = tk.Radiobutton(
                bit_depth_section,
                text=description,
                variable=self.bit_depth_var,
                value=depth,
                font=("Segoe UI", 10),
                bg='#252525',
                fg='#e0e0e0',
                selectcolor='#424242',
                activebackground='#252525',
                activeforeground='#90caf9'
            )
            rb.pack(anchor='w', pady=3)
        
        # Info section
        info_frame = tk.Frame(main_container, bg='#1e3a5f', relief=tk.FLAT, bd=1)
        info_frame.pack(fill=tk.X, pady=(0, 20))
        
        info_label = tk.Label(
            info_frame,
            text="💡 Lưu ý: Cài đặt chỉ có thể thay đổi khi không đang thu âm.",
            font=("Segoe UI", 9),
            bg='#1e3a5f',
            fg='#90caf9',
            wraplength=500,
            justify=tk.LEFT
        )
        info_label.pack(padx=15, pady=12)
        
        # Buttons
        button_frame = tk.Frame(main_container, bg='#1a1a1a')
        button_frame.pack(fill=tk.X)
        
        # Save button
        save_button = tk.Button(
            button_frame,
            text="💾 Lưu",
            font=("Segoe UI", 11, "bold"),
            bg='#4CAF50',
            fg='#ffffff',
            activebackground='#388e3c',
            command=self.save_settings,
            cursor='hand2',
            relief=tk.FLAT,
            padx=30,
            pady=10
        )
        save_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Reset button
        reset_button = tk.Button(
            button_frame,
            text="🔄 Đặt lại mặc định",
            font=("Segoe UI", 11, "bold"),
            bg='#ff9800',
            fg='#ffffff',
            activebackground='#f57c00',
            command=self.reset_to_defaults,
            cursor='hand2',
            relief=tk.FLAT,
            padx=30,
            pady=10
        )
        reset_button.pack(side=tk.LEFT, padx=10)
        
        # Cancel button
        cancel_button = tk.Button(
            button_frame,
            text="❌ Hủy",
            font=("Segoe UI", 11, "bold"),
            bg='#757575',
            fg='#ffffff',
            activebackground='#616161',
            command=self.cancel,
            cursor='hand2',
            relief=tk.FLAT,
            padx=30,
            pady=10
        )
        cancel_button.pack(side=tk.RIGHT)
    
    def create_section(self, parent: tk.Frame, title: str, row: int) -> tk.Frame:
        """Create a settings section.
        
        Args:
            parent: Parent frame
            title: Section title
            row: Grid row number
            
        Returns:
            Section frame
        """
        section_frame = tk.Frame(parent, bg='#252525')
        section_frame.pack(fill=tk.X, padx=15, pady=10)
        
        title_label = tk.Label(
            section_frame,
            text=title,
            font=("Segoe UI", 11, "bold"),
            bg='#252525',
            fg='#81c784'
        )
        title_label.pack(anchor='w', pady=(0, 10))
        
        content_frame = tk.Frame(section_frame, bg='#252525')
        content_frame.pack(fill=tk.X, padx=10)
        
        return content_frame
    
    def load_current_settings(self):
        """Load current settings into dialog."""
        # Set device
        if self.current_config.device_id is not None:
            # Find device in list
            for i, dev in enumerate(self.available_devices):
                if dev.id == self.current_config.device_id:
                    self.device_combo.current(i)
                    break
        else:
            # Select default device
            for i, dev in enumerate(self.available_devices):
                if dev.is_default:
                    self.device_combo.current(i)
                    break
        
        # Set sample rate
        self.sample_rate_var.set(self.current_config.sample_rate)
        
        # Set bit depth
        self.bit_depth_var.set(self.current_config.bit_depth)
    
    def save_settings(self):
        """Save settings and close dialog.
        
        Implements Requirement 5.3
        """
        try:
            # Get selected device
            device_id = None
            selected_idx = self.device_combo.current()
            if 0 <= selected_idx < len(self.available_devices):
                device_id = self.available_devices[selected_idx].id
            
            # Create new config
            new_config = AudioRecordingConfig(
                sample_rate=self.sample_rate_var.get(),
                channels=self.current_config.channels,
                bit_depth=self.bit_depth_var.get(),
                chunk_size=self.current_config.chunk_size,
                output_dir=self.current_config.output_dir,
                format=self.current_config.format,
                device_id=device_id
            )
            
            # Save to file
            if self.settings_manager.save_settings(new_config):
                self.result = new_config
                
                # Call callback if provided
                if self.on_save_callback:
                    self.on_save_callback(new_config)
                
                messagebox.showinfo(
                    "Thành công",
                    "Cài đặt đã được lưu!\n\n"
                    "Cài đặt mới sẽ được áp dụng cho lần thu âm tiếp theo."
                )
                
                self.dialog.destroy()
            else:
                messagebox.showerror(
                    "Lỗi",
                    "Không thể lưu cài đặt.\n\n"
                    "Vui lòng kiểm tra quyền ghi file."
                )
        
        except ValueError as e:
            messagebox.showerror(
                "Lỗi Cài Đặt",
                f"Cài đặt không hợp lệ:\n\n{str(e)}"
            )
        except Exception as e:
            messagebox.showerror(
                "Lỗi",
                f"Không thể lưu cài đặt:\n\n{str(e)}"
            )
    
    def reset_to_defaults(self):
        """Reset settings to defaults."""
        result = messagebox.askyesno(
            "Xác nhận",
            "Bạn có chắc muốn đặt lại tất cả cài đặt về mặc định?\n\n"
            "Hành động này sẽ:\n"
            "- Đặt sample rate về 16 kHz\n"
            "- Đặt bit depth về 16-bit\n"
            "- Chọn thiết bị microphone mặc định"
        )
        
        if result:
            # Reset to defaults
            self.sample_rate_var.set(16000)
            self.bit_depth_var.set(16)
            
            # Select default device
            for i, dev in enumerate(self.available_devices):
                if dev.is_default:
                    self.device_combo.current(i)
                    break
            
            messagebox.showinfo(
                "Thành công",
                "Đã đặt lại cài đặt về mặc định.\n\n"
                "Nhấn 'Lưu' để áp dụng thay đổi."
            )
    
    def cancel(self):
        """Cancel and close dialog."""
        self.dialog.destroy()
    
    def show(self) -> Optional[AudioRecordingConfig]:
        """Show dialog and wait for result.
        
        Returns:
            New AudioRecordingConfig if saved, None if cancelled
        """
        self.dialog.wait_window()
        return self.result


# Example usage for testing
if __name__ == "__main__":
    from src.audio_recording.models import AudioDevice
    
    root = tk.Tk()
    root.title("Settings Dialog Test")
    root.geometry("400x300")
    root.configure(bg='#1a1a1a')
    
    # Mock devices
    devices = [
        AudioDevice(id=0, name="Built-in Microphone", channels=1, default_sample_rate=44100, is_default=True),
        AudioDevice(id=1, name="USB Microphone", channels=2, default_sample_rate=48000, is_default=False),
        AudioDevice(id=2, name="External Audio Interface", channels=2, default_sample_rate=48000, is_default=False),
    ]
    
    # Current config
    config = AudioRecordingConfig()
    
    def show_settings():
        dialog = AudioSettingsDialog(root, config, devices)
        result = dialog.show()
        if result:
            print(f"New config: {result}")
    
    button = tk.Button(
        root,
        text="Open Settings",
        command=show_settings,
        font=("Segoe UI", 12),
        bg='#4CAF50',
        fg='#ffffff',
        padx=20,
        pady=10
    )
    button.pack(expand=True)
    
    root.mainloop()
