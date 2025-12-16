"""Utility functions for audio recording system."""

import shutil
import platform
from pathlib import Path
from typing import Tuple


def check_disk_space(path: Path, required_mb: int = 100) -> Tuple[bool, int]:
    """Check if sufficient disk space is available.
    
    Args:
        path: Path to check disk space for
        required_mb: Required space in megabytes (default: 100 MB)
        
    Returns:
        Tuple of (has_space, available_mb) where:
        - has_space: True if sufficient space is available
        - available_mb: Available space in megabytes
    """
    try:
        # Get disk usage statistics
        stat = shutil.disk_usage(path)
        available_mb = stat.free // (1024 * 1024)
        
        has_space = available_mb >= required_mb
        return has_space, available_mb
        
    except Exception:
        # If we can't check, assume space is available
        return True, 0


def get_permission_instructions() -> str:
    """Get platform-specific instructions for granting microphone permissions.
    
    Returns:
        Formatted string with permission instructions
    """
    system = platform.system()
    
    if system == "Windows":
        return (
            "Hướng dẫn cấp quyền microphone trên Windows:\n\n"
            "1. Mở Settings (Cài đặt)\n"
            "2. Chọn Privacy & Security (Quyền riêng tư & Bảo mật)\n"
            "3. Chọn Microphone (Microphone)\n"
            "4. Bật 'Let apps access your microphone'\n"
            "5. Tìm và bật quyền cho ứng dụng này\n"
            "6. Khởi động lại ứng dụng"
        )
    elif system == "Darwin":  # macOS
        return (
            "Hướng dẫn cấp quyền microphone trên macOS:\n\n"
            "1. Mở System Preferences (Tùy chọn Hệ thống)\n"
            "2. Chọn Security & Privacy (Bảo mật & Quyền riêng tư)\n"
            "3. Chọn tab Privacy (Quyền riêng tư)\n"
            "4. Chọn Microphone từ danh sách bên trái\n"
            "5. Tích vào ô bên cạnh ứng dụng này\n"
            "6. Khởi động lại ứng dụng"
        )
    else:  # Linux
        return (
            "Hướng dẫn cấp quyền microphone trên Linux:\n\n"
            "1. Kiểm tra microphone đã được kết nối\n"
            "2. Chạy: arecord -l để xem danh sách thiết bị\n"
            "3. Kiểm tra quyền truy cập: ls -l /dev/snd/\n"
            "4. Thêm user vào group audio: sudo usermod -a -G audio $USER\n"
            "5. Đăng xuất và đăng nhập lại\n"
            "6. Khởi động lại ứng dụng"
        )


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB", "500 KB")
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def estimate_recording_size(duration_seconds: float, sample_rate: int, 
                            channels: int, bit_depth: int) -> int:
    """Estimate recording file size.
    
    Args:
        duration_seconds: Recording duration in seconds
        sample_rate: Sample rate in Hz
        channels: Number of channels
        bit_depth: Bit depth (16 or 24)
        
    Returns:
        Estimated file size in bytes
    """
    # Calculate raw audio data size
    bytes_per_sample = bit_depth // 8
    samples_per_second = sample_rate * channels
    bytes_per_second = samples_per_second * bytes_per_sample
    
    # Add WAV header overhead (44 bytes)
    total_bytes = int(duration_seconds * bytes_per_second) + 44
    
    return total_bytes


def get_error_message_with_solution(error: Exception) -> Tuple[str, str]:
    """Get user-friendly error message with solution.
    
    Args:
        error: Exception that occurred
        
    Returns:
        Tuple of (title, message) for error dialog
    """
    from .exceptions import (
        DeviceError, FileSystemError, AudioFormatError, 
        StateError, AudioRecordingError
    )
    
    error_str = str(error).lower()
    
    # Device errors
    if isinstance(error, DeviceError):
        if "no audio input devices found" in error_str or "no default input" in error_str:
            return (
                "Không Tìm Thấy Microphone",
                f"{error}\n\n"
                "Giải pháp:\n"
                "• Kiểm tra microphone đã được kết nối\n"
                "• Kiểm tra driver đã được cài đặt\n"
                "• Thử kết nối lại microphone\n"
                "• Khởi động lại ứng dụng"
            )
        elif "permission" in error_str or "access" in error_str:
            return (
                "Không Có Quyền Truy Cập",
                f"{error}\n\n"
                "Ứng dụng cần quyền truy cập microphone.\n\n"
                f"{get_permission_instructions()}"
            )
        elif "device unavailable" in error_str or "invalid device" in error_str:
            return (
                "Thiết Bị Không Khả Dụng",
                f"{error}\n\n"
                "Giải pháp:\n"
                "• Microphone có thể đang được sử dụng bởi ứng dụng khác\n"
                "• Thử đóng các ứng dụng khác đang dùng microphone\n"
                "• Kết nối lại microphone\n"
                "• Chọn thiết bị microphone khác"
            )
        else:
            return (
                "Lỗi Thiết Bị",
                f"{error}\n\n"
                "Giải pháp:\n"
                "• Kiểm tra kết nối microphone\n"
                "• Thử khởi động lại ứng dụng\n"
                "• Kiểm tra cài đặt âm thanh hệ thống"
            )
    
    # File system errors
    elif isinstance(error, FileSystemError):
        if "disk is full" in error_str or "no space left" in error_str:
            return (
                "Ổ Đĩa Đầy",
                f"{error}\n\n"
                "Giải pháp:\n"
                "• Xóa các file không cần thiết\n"
                "• Chọn thư mục lưu trên ổ đĩa khác\n"
                "• Dọn dẹp Recycle Bin / Trash"
            )
        elif "permission denied" in error_str or "cannot create" in error_str:
            return (
                "Không Có Quyền Ghi File",
                f"{error}\n\n"
                "Giải pháp:\n"
                "• Chạy ứng dụng với quyền Administrator\n"
                "• Chọn thư mục lưu khác có quyền ghi\n"
                "• Kiểm tra quyền truy cập thư mục"
            )
        else:
            return (
                "Lỗi File System",
                f"{error}\n\n"
                "Giải pháp:\n"
                "• Kiểm tra quyền truy cập thư mục\n"
                "• Kiểm tra dung lượng ổ đĩa\n"
                "• Thử chọn thư mục lưu khác"
            )
    
    # Audio format errors
    elif isinstance(error, AudioFormatError):
        return (
            "Lỗi Định Dạng Audio",
            f"{error}\n\n"
            "Giải pháp:\n"
            "• Thử sample rate khác (16kHz, 44.1kHz, 48kHz)\n"
            "• Thử bit depth khác (16-bit, 24-bit)\n"
            "• Chọn thiết bị microphone khác"
        )
    
    # State errors
    elif isinstance(error, StateError):
        return (
            "Lỗi Trạng Thái",
            f"{error}\n\n"
            "Vui lòng thử lại hoặc khởi động lại ứng dụng."
        )
    
    # Generic audio recording error
    elif isinstance(error, AudioRecordingError):
        return (
            "Lỗi Thu Âm",
            f"{error}\n\n"
            "Vui lòng thử lại hoặc liên hệ hỗ trợ."
        )
    
    # Unknown error
    else:
        return (
            "Lỗi Không Xác Định",
            f"{error}\n\n"
            "Vui lòng thử lại hoặc khởi động lại ứng dụng."
        )
