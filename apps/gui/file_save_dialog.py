# -*- coding: utf-8 -*-
"""
File Save Dialog with Custom Filename

Provides a dialog for users to rename files before saving.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, Tuple
import re


class FileSaveDialog:
    """
    Custom dialog for saving files with filename editing capability.
    
    Features:
    - Edit filename before saving
    - Choose save directory
    - Preview full path
    - Validate filename
    """
    
    def __init__(
        self,
        parent: tk.Widget,
        title: str = "Lưu File",
        default_filename: str = "untitled",
        default_dir: Optional[str] = None,
        file_extension: str = ".txt",
        file_types: Optional[list] = None
    ):
        """
        Initialize file save dialog.
        
        Args:
            parent: Parent widget
            title: Dialog title
            default_filename: Default filename (without extension)
            default_dir: Default directory (None = current directory)
            file_extension: File extension (e.g., ".txt", ".json")
            file_types: List of file type tuples for filedialog
        """
        self.parent = parent
        self.title = title
        self.default_filename = default_filename
        self.default_dir = Path(default_dir) if default_dir else Path.cwd()
        self.file_extension = file_extension if file_extension.startswith('.') else f'.{file_extension}'
        self.file_types = file_types or [
            (f"{file_extension.upper()[1:]} files", f"*{file_extension}"),
            ("All files", "*.*")
        ]
        
        # Result
        self.result_path: Optional[Path] = None
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("600x300")
        self.dialog.configure(bg='#1a1a1a')
        self.dialog.resizable(False, False)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center dialog
        self.center_dialog()
        
        # Create widgets
        self.create_widgets()
        
        # Focus on filename entry
        self.filename_entry.focus_set()
        self.filename_entry.select_range(0, tk.END)
    
    def center_dialog(self):
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        
        # Get parent position and size
        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()
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
        """Create dialog widgets."""
        # Header
        header_frame = tk.Frame(self.dialog, bg='#0d47a1', height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_label = tk.Label(
            header_frame,
            text=f"💾 {self.title.upper()}",
            font=("Segoe UI", 14, "bold"),
            bg='#0d47a1',
            fg='#ffffff'
        )
        header_label.pack(pady=15)
        
        # Content frame
        content_frame = tk.Frame(self.dialog, bg='#1a1a1a')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Filename section
        filename_label = tk.Label(
            content_frame,
            text="📝 Tên file:",
            font=("Segoe UI", 10, "bold"),
            bg='#1a1a1a',
            fg='#90caf9'
        )
        filename_label.pack(anchor='w', pady=(0, 5))
        
        filename_frame = tk.Frame(content_frame, bg='#1a1a1a')
        filename_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.filename_var = tk.StringVar(value=self.default_filename)
        self.filename_entry = tk.Entry(
            filename_frame,
            textvariable=self.filename_var,
            font=("Segoe UI", 11),
            bg='#2b2b2b',
            fg='#ffffff',
            insertbackground='#ffffff',
            relief=tk.FLAT,
            bd=0
        )
        self.filename_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        self.filename_entry.bind('<KeyRelease>', self.on_filename_change)
        
        # Extension label
        ext_label = tk.Label(
            filename_frame,
            text=self.file_extension,
            font=("Segoe UI", 11, "bold"),
            bg='#1a1a1a',
            fg='#81c784'
        )
        ext_label.pack(side=tk.LEFT)
        
        # Directory section
        dir_label = tk.Label(
            content_frame,
            text="📁 Thư mục:",
            font=("Segoe UI", 10, "bold"),
            bg='#1a1a1a',
            fg='#90caf9'
        )
        dir_label.pack(anchor='w', pady=(0, 5))
        
        dir_frame = tk.Frame(content_frame, bg='#1a1a1a')
        dir_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.dir_var = tk.StringVar(value=str(self.default_dir))
        dir_entry = tk.Entry(
            dir_frame,
            textvariable=self.dir_var,
            font=("Segoe UI", 9),
            bg='#2b2b2b',
            fg='#9e9e9e',
            state='readonly',
            relief=tk.FLAT,
            bd=0
        )
        dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        
        browse_button = tk.Button(
            dir_frame,
            text="📂",
            font=("Segoe UI", 12),
            bg='#455a64',
            fg='#ffffff',
            activebackground='#37474f',
            command=self.browse_directory,
            cursor='hand2',
            relief=tk.FLAT,
            width=3
        )
        browse_button.pack(side=tk.LEFT)
        
        # Preview section
        preview_label = tk.Label(
            content_frame,
            text="ℹ️ Preview:",
            font=("Segoe UI", 9),
            bg='#1a1a1a',
            fg='#9e9e9e'
        )
        preview_label.pack(anchor='w', pady=(0, 5))
        
        self.preview_var = tk.StringVar()
        self.update_preview()
        
        preview_text = tk.Label(
            content_frame,
            textvariable=self.preview_var,
            font=("Consolas", 9),
            bg='#2b2b2b',
            fg='#64b5f6',
            anchor='w',
            justify=tk.LEFT,
            wraplength=550,
            relief=tk.FLAT,
            padx=10,
            pady=8
        )
        preview_text.pack(fill=tk.X, pady=(0, 15))
        
        # Validation message
        self.validation_var = tk.StringVar()
        self.validation_label = tk.Label(
            content_frame,
            textvariable=self.validation_var,
            font=("Segoe UI", 9),
            bg='#1a1a1a',
            fg='#f44336',
            anchor='w'
        )
        self.validation_label.pack(fill=tk.X, pady=(0, 10))
        
        # Buttons
        button_frame = tk.Frame(content_frame, bg='#1a1a1a')
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        save_button = tk.Button(
            button_frame,
            text="💾 Lưu",
            font=("Segoe UI", 11, "bold"),
            bg='#43a047',
            fg='#ffffff',
            activebackground='#388e3c',
            command=self.save,
            cursor='hand2',
            relief=tk.FLAT,
            padx=30,
            pady=10
        )
        save_button.pack(side=tk.LEFT, padx=(0, 10))
        
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
        cancel_button.pack(side=tk.LEFT)
        
        # Bind Enter key to save
        self.dialog.bind('<Return>', lambda e: self.save())
        self.dialog.bind('<Escape>', lambda e: self.cancel())
    
    def on_filename_change(self, event=None):
        """Handle filename change."""
        self.update_preview()
        self.validate_filename()
    
    def update_preview(self):
        """Update preview path."""
        filename = self.filename_var.get().strip()
        if not filename:
            filename = "untitled"
        
        full_path = Path(self.dir_var.get()) / f"{filename}{self.file_extension}"
        self.preview_var.set(str(full_path))
    
    def validate_filename(self) -> bool:
        """
        Validate filename.
        
        Returns:
            True if valid, False otherwise
        """
        filename = self.filename_var.get().strip()
        
        if not filename:
            self.validation_var.set("⚠️ Tên file không được để trống!")
            return False
        
        # Check for invalid characters (Windows)
        invalid_chars = r'[<>:"/\\|?*]'
        if re.search(invalid_chars, filename):
            self.validation_var.set("⚠️ Tên file chứa ký tự không hợp lệ: < > : \" / \\ | ? *")
            return False
        
        # Check for reserved names (Windows)
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL', 'COM1', 'COM2', 'COM3', 'COM4',
                         'COM5', 'COM6', 'COM7', 'COM8', 'COM9', 'LPT1', 'LPT2',
                         'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9']
        if filename.upper() in reserved_names:
            self.validation_var.set(f"⚠️ '{filename}' là tên file dành riêng của hệ thống!")
            return False
        
        # Check if file already exists
        full_path = Path(self.dir_var.get()) / f"{filename}{self.file_extension}"
        if full_path.exists():
            self.validation_var.set("ℹ️ File đã tồn tại - sẽ ghi đè nếu lưu")
            # Not an error, just a warning
        else:
            self.validation_var.set("")
        
        return True
    
    def browse_directory(self):
        """Browse for save directory."""
        directory = filedialog.askdirectory(
            title="Chọn thư mục lưu file",
            initialdir=self.dir_var.get()
        )
        
        if directory:
            self.dir_var.set(directory)
            self.update_preview()
    
    def save(self):
        """Save file."""
        if not self.validate_filename():
            return
        
        filename = self.filename_var.get().strip()
        directory = Path(self.dir_var.get())
        full_path = directory / f"{filename}{self.file_extension}"
        
        # Check if file exists and confirm overwrite
        if full_path.exists():
            response = messagebox.askyesno(
                "Xác nhận ghi đè",
                f"File '{full_path.name}' đã tồn tại.\n\n"
                f"Bạn có muốn ghi đè không?",
                parent=self.dialog
            )
            if not response:
                return
        
        # Set result and close
        self.result_path = full_path
        self.dialog.destroy()
    
    def cancel(self):
        """Cancel and close dialog."""
        self.result_path = None
        self.dialog.destroy()
    
    def show(self) -> Optional[Path]:
        """
        Show dialog and wait for result.
        
        Returns:
            Path object if saved, None if cancelled
        """
        # Wait for dialog to close
        self.dialog.wait_window()
        return self.result_path


def ask_save_file(
    parent: tk.Widget,
    title: str = "Lưu File",
    default_filename: str = "untitled",
    default_dir: Optional[str] = None,
    file_extension: str = ".txt",
    file_types: Optional[list] = None
) -> Optional[str]:
    """
    Show file save dialog and return selected path.
    
    Args:
        parent: Parent widget
        title: Dialog title
        default_filename: Default filename (without extension)
        default_dir: Default directory
        file_extension: File extension
        file_types: List of file type tuples
    
    Returns:
        File path as string if saved, None if cancelled
    """
    dialog = FileSaveDialog(
        parent=parent,
        title=title,
        default_filename=default_filename,
        default_dir=default_dir,
        file_extension=file_extension,
        file_types=file_types
    )
    
    result = dialog.show()
    return str(result) if result else None


# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    root.title("File Save Dialog Test")
    root.geometry("400x300")
    root.configure(bg='#1a1a1a')
    
    def test_dialog():
        from datetime import datetime
        
        default_name = f"interview_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        file_path = ask_save_file(
            parent=root,
            title="Lưu Kết Quả Phỏng Vấn",
            default_filename=default_name,
            default_dir="./reports",
            file_extension=".txt",
            file_types=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            print(f"Selected: {file_path}")
            messagebox.showinfo("Result", f"File path:\n{file_path}")
        else:
            print("Cancelled")
    
    test_button = tk.Button(
        root,
        text="Test Save Dialog",
        command=test_dialog,
        font=("Segoe UI", 12),
        bg='#43a047',
        fg='#ffffff',
        padx=20,
        pady=10
    )
    test_button.pack(expand=True)
    
    root.mainloop()
