#!/usr/bin/env python3
"""
Script to list all available audio input devices (microphones).
Displays device index, name, channels, and sample rate information.
"""

import os
import sys
import argparse
from typing import List, Dict

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import pyaudio
except ImportError:
    print("Error: PyAudio not installed!")
    print("Install with: pip install pyaudio")
    sys.exit(1)


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_audio_devices(input_only: bool = True) -> List[Dict]:
    """
    Get list of audio devices.
    
    Args:
        input_only: If True, only return input devices (microphones)
        
    Returns:
        List of device information dictionaries
    """
    p = pyaudio.PyAudio()
    devices = []
    
    try:
        device_count = p.get_device_count()
        
        for i in range(device_count):
            try:
                info = p.get_device_info_by_index(i)
                
                # Filter for input devices if requested
                if input_only and info['maxInputChannels'] == 0:
                    continue
                
                device = {
                    'index': i,
                    'name': info['name'],
                    'max_input_channels': info['maxInputChannels'],
                    'max_output_channels': info['maxOutputChannels'],
                    'default_sample_rate': int(info['defaultSampleRate']),
                    'host_api': p.get_host_api_info_by_index(info['hostApi'])['name'],
                    'is_default_input': (i == p.get_default_input_device_info()['index']),
                    'is_default_output': (i == p.get_default_output_device_info()['index'])
                }
                
                devices.append(device)
                
            except Exception as e:
                # Skip devices that can't be queried
                continue
    
    finally:
        p.terminate()
    
    return devices


def test_device(device_index: int, duration: float = 2.0) -> bool:
    """
    Test if a device can be opened and used for recording.
    
    Args:
        device_index: Device index to test
        duration: Test duration in seconds
        
    Returns:
        True if device works, False otherwise
    """
    p = pyaudio.PyAudio()
    
    try:
        # Try to open stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024
        )
        
        # Try to read some data
        import time
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                data = stream.read(1024, exception_on_overflow=False)
            except:
                stream.stop_stream()
                stream.close()
                p.terminate()
                return False
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        return True
        
    except Exception as e:
        p.terminate()
        return False


def print_device_info(device: Dict, detailed: bool = False, test: bool = False):
    """
    Print formatted device information.
    
    Args:
        device: Device information dictionary
        detailed: Show detailed information
        test: Test the device
    """
    # Device index and name
    index = device['index']
    name = device['name']
    
    # Highlight default device
    if device['is_default_input']:
        print(f"{Colors.OKGREEN}{Colors.BOLD}[{index}] {name} (DEFAULT INPUT){Colors.ENDC}")
    elif device['is_default_output']:
        print(f"{Colors.OKCYAN}[{index}] {name} (DEFAULT OUTPUT){Colors.ENDC}")
    else:
        print(f"{Colors.BOLD}[{index}] {name}{Colors.ENDC}")
    
    # Basic info
    print(f"    Input Channels:  {device['max_input_channels']}")
    print(f"    Output Channels: {device['max_output_channels']}")
    print(f"    Sample Rate:     {device['default_sample_rate']} Hz")
    
    if detailed:
        print(f"    Host API:        {device['host_api']}")
    
    # Test device if requested
    if test and device['max_input_channels'] > 0:
        print(f"    Testing device...", end=' ', flush=True)
        if test_device(index, duration=1.0):
            print(f"{Colors.OKGREEN}✓ Working{Colors.ENDC}")
        else:
            print(f"{Colors.FAIL}✗ Failed{Colors.ENDC}")
    
    print()


def print_summary(devices: List[Dict]):
    """
    Print summary statistics.
    
    Args:
        devices: List of device information dictionaries
    """
    total = len(devices)
    input_devices = sum(1 for d in devices if d['max_input_channels'] > 0)
    output_devices = sum(1 for d in devices if d['max_output_channels'] > 0)
    
    print(f"{Colors.BOLD}Summary:{Colors.ENDC}")
    print(f"  Total devices:  {total}")
    print(f"  Input devices:  {input_devices}")
    print(f"  Output devices: {output_devices}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='List available audio input devices (microphones)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all input devices
  python list_audio_devices.py
  
  # List all devices (input and output)
  python list_audio_devices.py --all
  
  # Show detailed information
  python list_audio_devices.py --detailed
  
  # Test each device
  python list_audio_devices.py --test
  
  # Get device info in JSON format
  python list_audio_devices.py --json
        """
    )
    
    parser.add_argument('--all', action='store_true',
                        help='Show all devices (input and output)')
    parser.add_argument('--detailed', action='store_true',
                        help='Show detailed device information')
    parser.add_argument('--test', action='store_true',
                        help='Test each input device')
    parser.add_argument('--json', action='store_true',
                        help='Output in JSON format')
    
    args = parser.parse_args()
    
    # Get devices
    try:
        devices = get_audio_devices(input_only=not args.all)
    except Exception as e:
        print(f"{Colors.FAIL}Error: Failed to enumerate audio devices{Colors.ENDC}")
        print(f"Details: {e}")
        sys.exit(1)
    
    if not devices:
        print(f"{Colors.WARNING}No audio devices found!{Colors.ENDC}")
        print("\nPossible reasons:")
        print("  - No microphone connected")
        print("  - Audio drivers not installed")
        print("  - Permission denied (check system settings)")
        sys.exit(1)
    
    # Output format
    if args.json:
        import json
        print(json.dumps(devices, indent=2))
        return
    
    # Print header
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("="*70)
    print("Available Audio Devices".center(70))
    print("="*70)
    print(f"{Colors.ENDC}\n")
    
    # Print devices
    for device in devices:
        print_device_info(device, detailed=args.detailed, test=args.test)
    
    # Print summary
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print_summary(devices)
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")
    
    # Print usage tips
    print(f"{Colors.BOLD}Usage Tips:{Colors.ENDC}")
    print("  • Use the device index [N] when configuring audio capture")
    print("  • Default input device is marked with (DEFAULT INPUT)")
    print("  • Recommended sample rate: 16000 Hz for speech recognition")
    print("  • Use mono (1 channel) for speech analysis")
    
    # Find default input device
    default_input = next((d for d in devices if d['is_default_input']), None)
    if default_input:
        print(f"\n{Colors.OKGREEN}Default input device: [{default_input['index']}] {default_input['name']}{Colors.ENDC}")
    
    print(f"\n{Colors.OKCYAN}To use a specific device, set device_index in AudioConfig{Colors.ENDC}")
    print(f"{Colors.OKCYAN}Example: AudioConfig(device_index={devices[0]['index']}){Colors.ENDC}")


if __name__ == '__main__':
    main()
