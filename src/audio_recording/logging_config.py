"""Logging configuration for audio recording module."""

import logging
import sys
from pathlib import Path
from typing import Optional


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> None:
    """Configure logging for the audio_recording module.
    
    Sets up logging with appropriate handlers and formatters for both
    console and file output.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        log_file: Optional path to log file. If None, only console logging is used.
        format_string: Optional custom format string. If None, uses default format.
    
    Example:
        >>> from pathlib import Path
        >>> configure_logging(
        ...     level=logging.DEBUG,
        ...     log_file=Path("logs/audio_recording.log")
        ... )
    """
    # Default format includes timestamp, level, module, and message
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get the root logger for audio_recording module
    logger = logging.getLogger("src.audio_recording")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log file specified)
    if log_file is not None:
        log_file = Path(log_file)
        
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger to avoid duplicate logs
    logger.propagate = False
    
    logger.info("Logging configured: level=%s, log_file=%s", logging.getLevelName(level), log_file)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("This is a log message")
    """
    return logging.getLogger(name)


def set_level(level: int) -> None:
    """Set logging level for the audio_recording module.
    
    Args:
        level: Logging level (e.g., logging.DEBUG, logging.INFO)
        
    Example:
        >>> set_level(logging.DEBUG)
    """
    logger = logging.getLogger("src.audio_recording")
    logger.setLevel(level)
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(level)
    
    logger.info("Logging level changed to: %s", logging.getLevelName(level))


def disable_logging() -> None:
    """Disable all logging for the audio_recording module.
    
    Useful for testing or when logging is not desired.
    
    Example:
        >>> disable_logging()
    """
    logger = logging.getLogger("src.audio_recording")
    logger.disabled = True


def enable_logging() -> None:
    """Enable logging for the audio_recording module.
    
    Re-enables logging after it has been disabled.
    
    Example:
        >>> enable_logging()
    """
    logger = logging.getLogger("src.audio_recording")
    logger.disabled = False
