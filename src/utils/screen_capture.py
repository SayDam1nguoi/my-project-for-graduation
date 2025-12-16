"""
Screen Capture Module

Capture frames from the entire screen for emotion detection.
Useful for analyzing emotions during online interviews (Zoom, Teams, etc.)
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
import platform

# Try to import mss for cross-platform screen capture
try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("Warning: mss not available. Install with: pip install mss")

# Platform-specific imports (fallback)
if platform.system() == 'Windows':
    try:
        import win32gui
        import win32ui
        import win32con
        from PIL import Image
        WINDOWS_AVAILABLE = True
    except ImportError:
        WINDOWS_AVAILABLE = False
else:
    WINDOWS_AVAILABLE = False


class WindowInfo:
    """Information about a window."""
    
    def __init__(self, hwnd: int, title: str):
        """
        Initialize WindowInfo.
        
        Args:
            hwnd: Window handle
            title: Window title
        """
        self.hwnd = hwnd
        self.title = title
    
    def __repr__(self):
        return f"WindowInfo(hwnd={self.hwnd}, title='{self.title}')"


class ScreenCapture:
    """
    Capture frames from the entire screen.
    
    Supports:
    - Full screen capture
    - Monitor selection (for multi-monitor setups)
    - Real-time frame capture for emotion detection
    """
    
    def __init__(self, monitor_number: int = 1):
        """
        Initialize ScreenCapture.
        
        Args:
            monitor_number: Monitor to capture (1 = primary, 2 = secondary, etc.)
                           0 = all monitors combined
        """
        if not MSS_AVAILABLE:
            raise RuntimeError(
                "Screen capture requires mss. "
                "Install with: pip install mss"
            )
        
        self.monitor_number = monitor_number
        self.monitor = None
        self._init_monitor()
    
    def _init_monitor(self):
        """Initialize monitor information."""
        with mss.mss() as sct:
            # Get monitor info
            if self.monitor_number == 0:
                # All monitors combined
                self.monitor = sct.monitors[0].copy()
            elif self.monitor_number <= len(sct.monitors) - 1:
                self.monitor = sct.monitors[self.monitor_number].copy()
            else:
                # Default to primary monitor
                self.monitor = sct.monitors[1].copy()
                print(f"Warning: Monitor {self.monitor_number} not found, using primary monitor")
    
    def get_monitors(self) -> List[dict]:
        """
        Get list of available monitors.
        
        Returns:
            List of monitor information dictionaries
        """
        monitors = []
        with mss.mss() as sct:
            for i, monitor in enumerate(sct.monitors):
                if i == 0:
                    monitors.append({
                        'number': 0,
                        'name': 'All Monitors',
                        'width': monitor['width'],
                        'height': monitor['height']
                    })
                else:
                    monitors.append({
                        'number': i,
                        'name': f'Monitor {i}',
                        'width': monitor['width'],
                        'height': monitor['height']
                    })
        return monitors
    
    def select_monitor(self, monitor_number: int):
        """
        Select monitor for capture.
        
        Args:
            monitor_number: Monitor to capture (1 = primary, 2 = secondary, etc.)
                           0 = all monitors combined
        """
        self.monitor_number = monitor_number
        self._init_monitor()
    
    def capture_screen(
        self,
        region: Tuple[int, int, int, int] = None
    ) -> Optional[np.ndarray]:
        """
        Capture frame from screen.
        
        Args:
            region: Optional region (x, y, width, height) to capture
            
        Returns:
            Frame as numpy array (BGR format) or None if failed
        """
        try:
            # Create a new mss instance for each capture to avoid threading issues
            with mss.mss() as sct:
                # Capture the screen
                if region is not None:
                    # Capture specific region
                    x, y, w, h = region
                    monitor = {
                        'left': self.monitor['left'] + x,
                        'top': self.monitor['top'] + y,
                        'width': w,
                        'height': h
                    }
                    screenshot = sct.grab(monitor)
                else:
                    # Capture entire monitor
                    screenshot = sct.grab(self.monitor)
                
                # Convert to numpy array
                img = np.array(screenshot)
                
                # Convert BGRA to BGR
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                return img
        
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None
    
    def capture_current(self) -> Optional[np.ndarray]:
        """
        Capture frame from current monitor.
        
        Returns:
            Frame as numpy array or None
        """
        return self.capture_screen()
    
    # No cleanup needed - using context manager for each capture


def is_screen_capture_available() -> bool:
    """
    Check if screen capture is available on this system.
    
    Returns:
        True if available
    """
    return MSS_AVAILABLE
