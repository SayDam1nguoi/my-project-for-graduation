#!/usr/bin/env python3
"""
Laptop Speech-to-Text Optimization Script

This script automatically optimizes the speech-to-text system for laptop microphones
and CPU-only processing. It provides:
- Automatic configuration backup and application
- System diagnostics and hardware recommendations
- Audio quality testing and analysis
- Performance optimization tips

Usage:
    python scripts/optimize_for_laptop.py
    python scripts/optimize_for_laptop.py --test-audio
    python scripts/optimize_for_laptop.py --restore-backup
"""

import os
import sys
import argparse
import shutil
import platform
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import psutil
except ImportError:
    print("Error: psutil not installed!")
    print("Install with: pip install psutil")
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("Error: PyYAML not installed!")
    print("Install with: pip install pyyaml")
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


# Configuration paths
CONFIG_DIR = Path("config")
CURRENT_CONFIG = CONFIG_DIR / "speech_config.yaml"
OPTIMIZED_CONFIG = CONFIG_DIR / "cpu_optimized_speech_config.yaml"
BACKUP_DIR = CONFIG_DIR / "backups"


@dataclass
class SystemDiagnostics:
    """System hardware diagnostics."""
    cpu_cores: int
    cpu_frequency: float  # MHz
    total_ram: float  # GB
    available_ram: float  # GB
    platform_system: str
    platform_release: str
    python_version: str
    audio_devices: List[Dict] = field(default_factory=list)
    
    def get_recommended_model(self) -> str:
        """
        Recommend Whisper model based on hardware.
        
        Returns:
            Recommended model name
        """
        if self.cpu_cores < 4 or self.available_ram < 2:
            return "tiny"
        elif self.cpu_cores < 8 or self.available_ram < 4:
            return "base"
        else:
            return "small"
    
    def get_recommended_beam_size(self) -> int:
        """
        Recommend beam size based on CPU.
        
        Returns:
            Recommended beam size
        """
        if self.cpu_cores < 4:
            return 3
        elif self.cpu_cores < 8:
            return 6
        else:
            return 8


def get_audio_devices() -> List[Dict]:
    """
    Get list of available audio input devices.
    
    Returns:
        List of device information dictionaries
    """
    try:
        import pyaudio
        
        p = pyaudio.PyAudio()
        devices = []
        
        try:
            device_count = p.get_device_count()
            
            for i in range(device_count):
                try:
                    info = p.get_device_info_by_index(i)
                    
                    # Only include input devices
                    if info['maxInputChannels'] == 0:
                        continue
                    
                    device = {
                        'index': i,
                        'name': info['name'],
                        'channels': info['maxInputChannels'],
                        'sample_rate': int(info['defaultSampleRate']),
                        'is_default': (i == p.get_default_input_device_info()['index'])
                    }
                    
                    devices.append(device)
                    
                except Exception:
                    continue
        
        finally:
            p.terminate()
        
        return devices
        
    except ImportError:
        print_warning("PyAudio not installed - cannot detect audio devices")
        return []
    except Exception as e:
        print_warning(f"Failed to enumerate audio devices: {e}")
        return []


def get_system_diagnostics() -> SystemDiagnostics:
    """
    Collect system hardware diagnostics.
    
    Returns:
        SystemDiagnostics object with hardware information
    """
    # CPU information
    cpu_cores = psutil.cpu_count(logical=False) or psutil.cpu_count()
    
    try:
        cpu_freq = psutil.cpu_freq()
        cpu_frequency = cpu_freq.current if cpu_freq else 0.0
    except:
        cpu_frequency = 0.0
    
    # Memory information
    memory = psutil.virtual_memory()
    total_ram = memory.total / (1024 ** 3)  # Convert to GB
    available_ram = memory.available / (1024 ** 3)  # Convert to GB
    
    # Platform information
    platform_system = platform.system()
    platform_release = platform.release()
    python_version = platform.python_version()
    
    # Audio devices
    audio_devices = get_audio_devices()
    
    return SystemDiagnostics(
        cpu_cores=cpu_cores,
        cpu_frequency=cpu_frequency,
        total_ram=total_ram,
        available_ram=available_ram,
        platform_system=platform_system,
        platform_release=platform_release,
        python_version=python_version,
        audio_devices=audio_devices
    )


def display_system_info(diagnostics: SystemDiagnostics):
    """
    Display formatted system information.
    
    Args:
        diagnostics: SystemDiagnostics object
    """
    print(f"{Colors.BOLD}Hardware Information:{Colors.ENDC}\n")
    
    # CPU
    print(f"  CPU Cores:        {diagnostics.cpu_cores}")
    if diagnostics.cpu_frequency > 0:
        print(f"  CPU Frequency:    {diagnostics.cpu_frequency:.0f} MHz")
    
    # Memory
    print(f"  Total RAM:        {diagnostics.total_ram:.1f} GB")
    print(f"  Available RAM:    {diagnostics.available_ram:.1f} GB")
    
    # Platform
    print(f"  Operating System: {diagnostics.platform_system} {diagnostics.platform_release}")
    print(f"  Python Version:   {diagnostics.python_version}")
    
    # Audio devices
    print(f"\n{Colors.BOLD}Audio Devices:{Colors.ENDC}\n")
    
    if not diagnostics.audio_devices:
        print_warning("  No audio input devices detected")
    else:
        for device in diagnostics.audio_devices:
            if device['is_default']:
                print(f"  {Colors.OKGREEN}[{device['index']}] {device['name']} (DEFAULT){Colors.ENDC}")
            else:
                print(f"  [{device['index']}] {device['name']}")
            print(f"      Channels: {device['channels']}, Sample Rate: {device['sample_rate']} Hz")
    
    print()


def display_recommendations(diagnostics: SystemDiagnostics):
    """
    Display hardware-based recommendations.
    
    Args:
        diagnostics: SystemDiagnostics object
    """
    recommended_model = diagnostics.get_recommended_model()
    recommended_beam_size = diagnostics.get_recommended_beam_size()
    
    print(f"{Colors.BOLD}Recommended Settings:{Colors.ENDC}\n")
    
    # Model recommendation
    print(f"  Whisper Model:    {Colors.OKGREEN}{recommended_model}{Colors.ENDC}")
    
    if recommended_model == "tiny":
        print(f"    {Colors.WARNING}→ Your system has limited resources. The 'tiny' model is fastest.{Colors.ENDC}")
    elif recommended_model == "base":
        print(f"    {Colors.OKGREEN}→ Good balance of speed and accuracy for your system.{Colors.ENDC}")
    else:
        print(f"    {Colors.OKGREEN}→ Your system can handle the 'small' model for better accuracy.{Colors.ENDC}")
    
    # Beam size recommendation
    print(f"\n  Beam Size:        {Colors.OKGREEN}{recommended_beam_size}{Colors.ENDC}")
    
    if recommended_beam_size <= 3:
        print(f"    {Colors.WARNING}→ Lower beam size for faster processing on limited CPU.{Colors.ENDC}")
    elif recommended_beam_size <= 6:
        print(f"    {Colors.OKGREEN}→ Balanced beam size for good accuracy and performance.{Colors.ENDC}")
    else:
        print(f"    {Colors.OKGREEN}→ Higher beam size for better accuracy with your CPU.{Colors.ENDC}")
    
    # Performance expectations
    print(f"\n{Colors.BOLD}Performance Expectations:{Colors.ENDC}\n")
    
    if diagnostics.cpu_cores < 4:
        print(f"  {Colors.WARNING}CPU Usage:        70-85% (limited cores){Colors.ENDC}")
        print(f"  {Colors.WARNING}Latency:          5-7 seconds per chunk{Colors.ENDC}")
        print(f"  {Colors.OKGREEN}Accuracy:         ~80-85%{Colors.ENDC}")
    elif diagnostics.cpu_cores < 8:
        print(f"  {Colors.OKGREEN}CPU Usage:        60-70%{Colors.ENDC}")
        print(f"  {Colors.OKGREEN}Latency:          4-5 seconds per chunk{Colors.ENDC}")
        print(f"  {Colors.OKGREEN}Accuracy:         ~85-90%{Colors.ENDC}")
    else:
        print(f"  {Colors.OKGREEN}CPU Usage:        50-60%{Colors.ENDC}")
        print(f"  {Colors.OKGREEN}Latency:          3-4 seconds per chunk{Colors.ENDC}")
        print(f"  {Colors.OKGREEN}Accuracy:         ~90%+{Colors.ENDC}")
    
    # Memory check
    if diagnostics.available_ram < 2:
        print(f"\n  {Colors.WARNING}⚠ Low available RAM ({diagnostics.available_ram:.1f} GB){Colors.ENDC}")
        print(f"    → Close other applications for better performance")
    
    # Audio device check
    if not diagnostics.audio_devices:
        print(f"\n  {Colors.FAIL}⚠ No audio input devices detected{Colors.ENDC}")
        print(f"    → Connect a microphone or check audio drivers")
    elif len(diagnostics.audio_devices) == 1:
        print(f"\n  {Colors.OKGREEN}✓ Audio device detected and ready{Colors.ENDC}")
    else:
        print(f"\n  {Colors.OKCYAN}ℹ Multiple audio devices detected ({len(diagnostics.audio_devices)}){Colors.ENDC}")
        print(f"    → Default device will be used unless specified")
    
    print()


@dataclass
class AudioQualityMetrics:
    """Metrics for audio quality analysis."""
    rms_level: float
    peak_level: float
    snr_estimate: float
    clipping_detected: bool
    noise_floor: float
    dynamic_range: float
    
    def is_acceptable(self) -> bool:
        """
        Check if audio quality is acceptable.
        
        Returns:
            True if quality is acceptable
        """
        return (
            0.01 <= self.rms_level <= 0.3 and
            self.peak_level < 0.95 and
            not self.clipping_detected
        )
    
    def get_recommendations(self) -> List[str]:
        """
        Get recommendations for improvement.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if self.rms_level < 0.01:
            recommendations.append("Speak louder or move closer to microphone")
        elif self.rms_level > 0.3:
            recommendations.append("Speak softer or move away from microphone")
        
        if self.clipping_detected:
            recommendations.append("Reduce input volume to prevent clipping")
        
        if self.snr_estimate < 10:
            recommendations.append("Find quieter environment or use noise cancellation")
        
        if self.dynamic_range < 20:
            recommendations.append("Improve microphone positioning for better signal")
        
        return recommendations


def calculate_audio_metrics(audio_data: 'np.ndarray') -> AudioQualityMetrics:
    """
    Calculate audio quality metrics.
    
    Args:
        audio_data: Audio samples (float32, -1 to 1) or int16
        
    Returns:
        AudioQualityMetrics object
    """
    import numpy as np
    
    # Convert int16 to float32 and normalize
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # RMS level
    rms_level = float(np.sqrt(np.mean(audio_data ** 2)))
    
    # Peak level
    peak_level = float(np.max(np.abs(audio_data)))
    
    # Clipping detection
    clipping_detected = peak_level > 0.95
    
    # Noise floor estimation (bottom 10% of signal)
    sorted_abs = np.sort(np.abs(audio_data))
    noise_floor = float(np.mean(sorted_abs[:len(sorted_abs) // 10]))
    
    # SNR estimate (rough approximation)
    if noise_floor > 0:
        snr_estimate = 20 * np.log10(rms_level / noise_floor) if rms_level > 0 else 0
    else:
        snr_estimate = 60.0  # Very quiet
    
    # Dynamic range
    if noise_floor > 0:
        dynamic_range = 20 * np.log10(peak_level / noise_floor) if peak_level > 0 else 0
    else:
        dynamic_range = 60.0
    
    return AudioQualityMetrics(
        rms_level=rms_level,
        peak_level=peak_level,
        snr_estimate=float(snr_estimate),
        clipping_detected=clipping_detected,
        noise_floor=noise_floor,
        dynamic_range=float(dynamic_range)
    )


def test_audio_quality():
    """
    Test and analyze audio quality.
    
    Records a 3-second audio sample, analyzes it, and provides recommendations.
    """
    print_header("Audio Quality Test")
    
    try:
        import pyaudio
        import numpy as np
    except ImportError as e:
        print_error(f"Required library not installed: {e}")
        print("Install with: pip install pyaudio numpy")
        return
    
    # Check if audio enhancer is available
    try:
        from src.speech_analysis.laptop_audio_enhancer import LaptopAudioEnhancer
        enhancer = LaptopAudioEnhancer(sample_rate=16000)
        has_enhancer = True
    except ImportError:
        print_warning("Audio enhancer not available - showing original audio only")
        has_enhancer = False
    
    # Audio parameters
    SAMPLE_RATE = 16000
    DURATION = 3.0
    CHUNK_SIZE = 1024
    
    print(f"{Colors.BOLD}Recording 3 seconds of audio...{Colors.ENDC}")
    print("Please speak normally into your microphone.")
    print()
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    try:
        # Open stream
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        
        # Record audio
        frames = []
        num_chunks = int(SAMPLE_RATE * DURATION / CHUNK_SIZE)
        
        for i in range(num_chunks):
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            frames.append(data)
            
            # Progress indicator
            progress = (i + 1) / num_chunks
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"\r  Recording: [{bar}] {progress*100:.0f}%", end='', flush=True)
        
        print()  # New line after progress
        
        # Stop stream
        stream.stop_stream()
        stream.close()
        
        print_success("Recording complete!")
        
        # Convert to numpy array
        audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        audio_float = audio_data.astype(np.float32) / 32768.0
        
        # Analyze original audio
        print(f"\n{Colors.BOLD}Original Audio Quality:{Colors.ENDC}\n")
        original_metrics = calculate_audio_metrics(audio_float)
        display_audio_metrics(original_metrics, "Original")
        
        # Analyze enhanced audio if available
        if has_enhancer:
            try:
                enhanced_audio = enhancer.enhance_audio(audio_float)
                print(f"\n{Colors.BOLD}Enhanced Audio Quality:{Colors.ENDC}\n")
                enhanced_metrics = calculate_audio_metrics(enhanced_audio)
                display_audio_metrics(enhanced_metrics, "Enhanced")
                
                # Show improvement
                print(f"\n{Colors.BOLD}Enhancement Impact:{Colors.ENDC}\n")
                rms_change = ((enhanced_metrics.rms_level - original_metrics.rms_level) / original_metrics.rms_level * 100) if original_metrics.rms_level > 0 else 0
                snr_change = enhanced_metrics.snr_estimate - original_metrics.snr_estimate
                
                if rms_change > 5:
                    print(f"  RMS Level:        {Colors.OKGREEN}+{rms_change:.1f}% (improved){Colors.ENDC}")
                elif rms_change < -5:
                    print(f"  RMS Level:        {rms_change:.1f}% (reduced)")
                else:
                    print(f"  RMS Level:        {rms_change:.1f}% (minimal change)")
                
                if snr_change > 2:
                    print(f"  SNR:              {Colors.OKGREEN}+{snr_change:.1f} dB (improved){Colors.ENDC}")
                elif snr_change < -2:
                    print(f"  SNR:              {snr_change:.1f} dB (reduced)")
                else:
                    print(f"  SNR:              {snr_change:.1f} dB (minimal change)")
                
            except Exception as e:
                print_warning(f"Enhancement failed: {e}")
        
        # Show recommendations
        print(f"\n{Colors.BOLD}Recommendations:{Colors.ENDC}\n")
        
        metrics_to_check = enhanced_metrics if has_enhancer else original_metrics
        recommendations = metrics_to_check.get_recommendations()
        
        if not recommendations:
            print_success("Audio quality is good! No changes needed.")
        else:
            for rec in recommendations:
                print(f"  • {rec}")
        
        # Overall assessment
        print(f"\n{Colors.BOLD}Overall Assessment:{Colors.ENDC}\n")
        
        if metrics_to_check.is_acceptable():
            print_success("Audio quality is acceptable for speech recognition")
        else:
            print_warning("Audio quality could be improved - see recommendations above")
        
    except Exception as e:
        print_error(f"Audio test failed: {e}")
    
    finally:
        p.terminate()
    
    print()


def display_audio_metrics(metrics: AudioQualityMetrics, label: str = "Audio"):
    """
    Display formatted audio metrics.
    
    Args:
        metrics: AudioQualityMetrics object
        label: Label for the metrics
    """
    # RMS level
    rms_color = Colors.OKGREEN if 0.01 <= metrics.rms_level <= 0.3 else Colors.WARNING
    print(f"  RMS Level:        {rms_color}{metrics.rms_level:.4f}{Colors.ENDC}")
    
    # Peak level
    peak_color = Colors.OKGREEN if metrics.peak_level < 0.95 else Colors.FAIL
    print(f"  Peak Level:       {peak_color}{metrics.peak_level:.4f}{Colors.ENDC}")
    
    # Clipping
    if metrics.clipping_detected:
        print(f"  Clipping:         {Colors.FAIL}YES{Colors.ENDC}")
    else:
        print(f"  Clipping:         {Colors.OKGREEN}NO{Colors.ENDC}")
    
    # SNR
    snr_color = Colors.OKGREEN if metrics.snr_estimate >= 15 else Colors.WARNING
    print(f"  SNR Estimate:     {snr_color}{metrics.snr_estimate:.1f} dB{Colors.ENDC}")
    
    # Dynamic range
    print(f"  Dynamic Range:    {metrics.dynamic_range:.1f} dB")
    print(f"  Noise Floor:      {metrics.noise_floor:.6f}")


def print_usage_tips():
    """Print usage tips for optimal performance."""
    print(f"{Colors.BOLD}Usage Tips for Best Results:{Colors.ENDC}\n")
    
    print(f"{Colors.OKCYAN}Microphone Positioning:{Colors.ENDC}")
    print("  • Position microphone 15-30cm from your mouth")
    print("  • Avoid covering the microphone with your hand")
    print("  • Keep microphone away from fans and vents")
    
    print(f"\n{Colors.OKCYAN}Environment:{Colors.ENDC}")
    print("  • Use in a quiet room when possible")
    print("  • Close windows to reduce outside noise")
    print("  • Turn off fans or move away from them")
    print("  • Minimize keyboard typing during recording")
    
    print(f"\n{Colors.OKCYAN}Speaking:{Colors.ENDC}")
    print("  • Speak clearly at normal volume")
    print("  • Maintain consistent distance from microphone")
    print("  • Avoid sudden loud sounds or shouting")
    print("  • Pause briefly between sentences")
    
    print(f"\n{Colors.OKCYAN}Configuration:{Colors.ENDC}")
    print("  • Configuration file: config/speech_config.yaml")
    print("  • Backups saved to: config/backups/")
    print("  • To restore backup: python scripts/optimize_for_laptop.py --restore-backup")
    
    print(f"\n{Colors.OKGREEN}Your system is now optimized for laptop speech-to-text!{Colors.ENDC}")
    print()


def print_header(text: str):
    """Print formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("=" * 70)
    print(text.center(70))
    print("=" * 70)
    print(f"{Colors.ENDC}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def backup_config() -> Optional[Path]:
    """
    Backup current configuration file.
    
    Returns:
        Path to backup file, or None if backup failed
    """
    try:
        # Create backup directory if it doesn't exist
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check if current config exists
        if not CURRENT_CONFIG.exists():
            print_warning(f"No existing configuration found at {CURRENT_CONFIG}")
            return None
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = BACKUP_DIR / f"speech_config_backup_{timestamp}.yaml"
        
        # Copy current config to backup
        shutil.copy2(CURRENT_CONFIG, backup_path)
        
        print_success(f"Configuration backed up to: {backup_path}")
        return backup_path
        
    except Exception as e:
        print_error(f"Failed to backup configuration: {e}")
        return None


def apply_optimized_config() -> bool:
    """
    Apply CPU-optimized configuration.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Check if optimized config exists
        if not OPTIMIZED_CONFIG.exists():
            print_error(f"Optimized configuration not found at {OPTIMIZED_CONFIG}")
            return False
        
        # Copy optimized config to current config
        shutil.copy2(OPTIMIZED_CONFIG, CURRENT_CONFIG)
        
        print_success(f"Applied CPU-optimized configuration")
        return True
        
    except Exception as e:
        print_error(f"Failed to apply optimized configuration: {e}")
        return False


def restore_config(backup_path: Path) -> bool:
    """
    Restore configuration from backup.
    
    Args:
        backup_path: Path to backup file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not backup_path.exists():
            print_error(f"Backup file not found: {backup_path}")
            return False
        
        # Copy backup to current config
        shutil.copy2(backup_path, CURRENT_CONFIG)
        
        print_success(f"Configuration restored from: {backup_path}")
        return True
        
    except Exception as e:
        print_error(f"Failed to restore configuration: {e}")
        return False


def list_backups() -> List[Path]:
    """
    List available backup files.
    
    Returns:
        List of backup file paths, sorted by date (newest first)
    """
    if not BACKUP_DIR.exists():
        return []
    
    backups = list(BACKUP_DIR.glob("speech_config_backup_*.yaml"))
    backups.sort(reverse=True)  # Newest first
    return backups


def handle_restore_backup() -> int:
    """
    Handle backup restoration with user interaction.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print_header("Restore Configuration Backup")
    
    backups = list_backups()
    
    if not backups:
        print_warning("No backup files found")
        return 1
    
    print(f"{Colors.BOLD}Available backups:{Colors.ENDC}\n")
    
    for i, backup in enumerate(backups, 1):
        # Extract timestamp from filename
        timestamp_str = backup.stem.replace("speech_config_backup_", "")
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            date_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        except:
            date_str = timestamp_str
        
        print(f"  [{i}] {date_str}")
    
    print()
    
    try:
        choice = input(f"{Colors.BOLD}Select backup to restore (1-{len(backups)}) or 'q' to quit: {Colors.ENDC}").strip()
        
        if choice.lower() == 'q':
            print("Cancelled")
            return 0
        
        index = int(choice) - 1
        if index < 0 or index >= len(backups):
            print_error("Invalid selection")
            return 1
        
        backup_path = backups[index]
        
        # Confirm restoration
        confirm = input(f"{Colors.WARNING}This will overwrite your current configuration. Continue? (y/n): {Colors.ENDC}").strip().lower()
        
        if confirm not in ['y', 'yes']:
            print("Cancelled")
            return 0
        
        if restore_config(backup_path):
            print_success("Configuration restored successfully!")
            return 0
        else:
            return 1
            
    except (ValueError, IndexError):
        print_error("Invalid input")
        return 1
    except KeyboardInterrupt:
        print("\nCancelled")
        return 0


def apply_optimization() -> bool:
    """
    Apply full optimization with backup.
    
    Returns:
        True if successful, False otherwise
    """
    # Step 1: Backup current configuration
    print("Step 1: Backing up current configuration...")
    backup_path = backup_config()
    
    if backup_path is None and CURRENT_CONFIG.exists():
        # Backup failed but config exists - ask user
        response = input(f"{Colors.WARNING}Backup failed. Continue anyway? (y/n): {Colors.ENDC}").strip().lower()
        if response not in ['y', 'yes']:
            print("Optimization cancelled")
            return False
    
    # Step 2: Apply optimized configuration
    print("\nStep 2: Applying CPU-optimized configuration...")
    if not apply_optimized_config():
        # Rollback if backup exists
        if backup_path and backup_path.exists():
            print_warning("Rolling back to previous configuration...")
            restore_config(backup_path)
        return False
    
    print()
    print_success("Configuration updated successfully!")
    
    # Show what was changed
    print(f"\n{Colors.BOLD}Key optimizations applied:{Colors.ENDC}")
    print("  • Audio enhancement enabled for laptop microphones")
    print("  • High-pass filter to remove low-frequency noise")
    print("  • Noise gate to suppress background noise")
    print("  • Volume normalization for consistent levels")
    print("  • CPU-optimized Whisper settings (base model, beam size 6)")
    print("  • Increased chunk duration for better context")
    print("  • Vietnamese optimizer enabled")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Optimize speech-to-text for laptop microphones and CPU processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full optimization
  python scripts/optimize_for_laptop.py
  
  # Test audio quality after optimization
  python scripts/optimize_for_laptop.py --test-audio
  
  # Restore previous configuration
  python scripts/optimize_for_laptop.py --restore-backup
  
  # Show system diagnostics only
  python scripts/optimize_for_laptop.py --diagnostics-only
        """
    )
    
    parser.add_argument('--test-audio', action='store_true',
                        help='Test audio quality after optimization')
    parser.add_argument('--restore-backup', action='store_true',
                        help='Restore previous configuration from backup')
    parser.add_argument('--diagnostics-only', action='store_true',
                        help='Show system diagnostics without applying changes')
    parser.add_argument('--skip-test', action='store_true',
                        help='Skip audio quality test prompt')
    
    args = parser.parse_args()
    
    # Print welcome message
    print_header("Laptop Speech-to-Text Optimization")
    
    print(f"{Colors.BOLD}This script will:{Colors.ENDC}")
    print("  1. Analyze your system hardware")
    print("  2. Backup your current configuration")
    print("  3. Apply CPU-optimized settings")
    print("  4. Enable audio enhancement for laptop microphones")
    print("  5. Provide performance recommendations")
    print()
    
    # Handle restore backup
    if args.restore_backup:
        return handle_restore_backup()
    
    # Show system diagnostics
    print_header("System Diagnostics")
    diagnostics = get_system_diagnostics()
    display_system_info(diagnostics)
    
    # Show recommendations
    print_header("Hardware Recommendations")
    display_recommendations(diagnostics)
    
    # Exit if diagnostics only
    if args.diagnostics_only:
        return 0
    
    # Apply optimization
    print_header("Applying Optimization")
    
    if not apply_optimization():
        print_error("Optimization failed!")
        return 1
    
    print_success("Optimization completed successfully!")
    
    # Offer audio quality test
    if not args.skip_test:
        print()
        response = input(f"{Colors.BOLD}Would you like to test audio quality now? (y/n): {Colors.ENDC}").strip().lower()
        if response in ['y', 'yes']:
            test_audio_quality()
    elif args.test_audio:
        test_audio_quality()
    
    # Print usage tips
    print_header("Next Steps")
    print_usage_tips()
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Operation cancelled by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
