#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diagnostic script for screen capture issues.

This script helps identify which windows can be successfully captured
and provides detailed information about capture failures.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def diagnose_screen_capture():
    """Run comprehensive diagnostics on screen capture."""
    print("="*70)
    print("SCREEN CAPTURE DIAGNOSTIC TOOL")
    print("="*70)
    
    # Check if screen capture is available
    print("\n1. Checking screen capture availability...")
    try:
        from src.utils.screen_capture import ScreenCapture, is_screen_capture_available
        
        if not is_screen_capture_available():
            print("✗ Screen capture is NOT available!")
            print("  Please install pywin32: pip install pywin32")
            return
        
        print("✓ Screen capture is available")
    except Exception as e:
        print(f"✗ Error importing screen capture: {e}")
        return
    
    # Create screen capture instance
    print("\n2. Creating ScreenCapture instance...")
    try:
        sc = ScreenCapture()
        print("✓ ScreenCapture instance created")
    except Exception as e:
        print(f"✗ Failed to create ScreenCapture: {e}")
        return
    
    # List all windows
    print("\n3. Listing available windows...")
    try:
        windows = sc.list_windows()
        print(f"✓ Found {len(windows)} windows")
        
        if len(windows) == 0:
            print("⚠ No windows available for testing")
            return
    except Exception as e:
        print(f"✗ Failed to list windows: {e}")
        return
    
    # Display windows grouped by type
    print("\n4. Available windows:")
    print("-" * 70)
    
    # Group windows
    browser_windows = []
    meeting_windows = []
    system_windows = []
    other_windows = []
    
    for window in windows:
        title_lower = window.title.lower()
        
        if any(browser in title_lower for browser in ['chrome', 'firefox', 'edge', 'brave', 'opera']):
            browser_windows.append(window)
        elif any(app in title_lower for app in ['zoom', 'teams', 'meet', 'skype', 'discord']):
            meeting_windows.append(window)
        elif any(sys in title_lower for sys in ['explorer', 'taskbar', 'start menu']):
            system_windows.append(window)
        else:
            other_windows.append(window)
    
    if browser_windows:
        print("\n[BROWSERS]")
        for i, window in enumerate(browser_windows[:5]):
            print(f"  {i+1}. {window.title[:60]}")
    
    if meeting_windows:
        print("\n[VIDEO CONFERENCING]")
        for i, window in enumerate(meeting_windows[:5]):
            print(f"  {i+1}. {window.title[:60]}")
    
    if other_windows:
        print("\n[OTHER APPLICATIONS]")
        for i, window in enumerate(other_windows[:10]):
            print(f"  {i+1}. {window.title[:60]}")
    
    # Test capture on each window
    print("\n" + "="*70)
    print("5. Testing capture capability for each window...")
    print("="*70)
    
    successful_windows = []
    failed_windows = []
    
    test_windows = (browser_windows[:3] + meeting_windows[:3] + other_windows[:5])[:10]
    
    for i, window in enumerate(test_windows):
        print(f"\nTesting window {i+1}/{len(test_windows)}: {window.title[:50]}")
        
        # Check if window is valid
        if not sc.is_window_valid(window.hwnd):
            print("  ✗ Window is not valid (may be closed)")
            failed_windows.append((window, "Invalid window"))
            continue
        
        # Select window
        sc.select_window(window.hwnd, window.title)
        print(f"  ✓ Window selected (hwnd: {window.hwnd})")
        
        # Try to capture
        try:
            frame = sc.capture_current()
            
            if frame is not None:
                print(f"  ✓ Capture successful! Frame shape: {frame.shape}")
                successful_windows.append(window)
            else:
                print("  ✗ Capture returned None (window may be minimized or hidden)")
                failed_windows.append((window, "Capture returned None"))
        except Exception as e:
            print(f"  ✗ Capture failed with error: {e}")
            failed_windows.append((window, str(e)))
    
    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)
    
    print(f"\nTotal windows found: {len(windows)}")
    print(f"Windows tested: {len(test_windows)}")
    print(f"Successful captures: {len(successful_windows)}")
    print(f"Failed captures: {len(failed_windows)}")
    
    if successful_windows:
        print("\n✓ WINDOWS THAT CAN BE CAPTURED:")
        for window in successful_windows:
            print(f"  - {window.title[:60]}")
    
    if failed_windows:
        print("\n✗ WINDOWS THAT FAILED:")
        for window, reason in failed_windows:
            print(f"  - {window.title[:50]}")
            print(f"    Reason: {reason}")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if len(successful_windows) == 0:
        print("\n⚠ No windows could be captured successfully!")
        print("\nPossible reasons:")
        print("  1. All tested windows are minimized")
        print("  2. Windows are hidden or occluded")
        print("  3. Permission issues (try running as administrator)")
        print("  4. Windows are on a different display")
        print("\nTry:")
        print("  - Restore/maximize a window before capturing")
        print("  - Open a simple application (Notepad, Calculator)")
        print("  - Run the GUI as administrator")
    else:
        print("\n✓ Some windows can be captured successfully!")
        print("\nTo use screen capture in the GUI:")
        print("  1. Make sure the window is NOT minimized")
        print("  2. The window should be visible on screen")
        print("  3. Select one of the working windows listed above")
        print("  4. Start scanning immediately after selection")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    try:
        diagnose_screen_capture()
    except KeyboardInterrupt:
        print("\n\nDiagnostic interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
