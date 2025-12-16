"""
Logging setup for speech analysis module.

This module configures logging for the speech analysis system,
including file logging with rotation and console output.
"""

import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional


# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logger(
    log_file: str = "logs/speech_analysis.log",
    log_level: str = "INFO",
    max_log_size_mb: int = 10,
    backup_count: int = 3,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup and configure logger for speech analysis.
    
    Args:
        log_file: Path to log file.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        max_log_size_mb: Maximum log file size in MB before rotation.
        backup_count: Number of backup log files to keep.
        console_output: Whether to also output logs to console.
        
    Returns:
        Configured logger instance.
    """
    global _logger
    
    # If logger already exists, return it
    if _logger is not None:
        return _logger
    
    # Create logger
    logger = logging.getLogger("speech_analysis")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation
    max_bytes = max_log_size_mb * 1024 * 1024  # Convert MB to bytes
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Store global reference
    _logger = logger
    
    logger.info("Speech analysis logger initialized")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Log level: {log_level}")
    
    return logger


def get_logger() -> logging.Logger:
    """
    Get the speech analysis logger instance.
    
    If logger hasn't been setup yet, initializes it with default settings.
    
    Returns:
        Logger instance.
    """
    global _logger
    
    if _logger is None:
        _logger = setup_logger()
    
    return _logger


def shutdown_logger() -> None:
    """
    Shutdown the logger and close all handlers.
    
    This should be called when the application is closing to ensure
    all log messages are flushed and files are properly closed.
    """
    global _logger
    
    if _logger is not None:
        _logger.info("Shutting down speech analysis logger")
        
        # Close and remove all handlers
        for handler in _logger.handlers[:]:
            handler.close()
            _logger.removeHandler(handler)
        
        _logger = None


def set_log_level(level: str) -> None:
    """
    Change the logging level dynamically.
    
    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    """
    logger = get_logger()
    logger.setLevel(getattr(logging, level.upper()))
    
    for handler in logger.handlers:
        handler.setLevel(getattr(logging, level.upper()))
    
    logger.info(f"Log level changed to {level}")


# Convenience functions for logging
def debug(message: str, *args, **kwargs) -> None:
    """Log a debug message."""
    get_logger().debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs) -> None:
    """Log an info message."""
    get_logger().info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs) -> None:
    """Log a warning message."""
    get_logger().warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs) -> None:
    """Log an error message."""
    get_logger().error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs) -> None:
    """Log a critical message."""
    get_logger().critical(message, *args, **kwargs)


def exception(message: str, *args, **kwargs) -> None:
    """Log an exception with traceback."""
    get_logger().exception(message, *args, **kwargs)


# Context manager for temporary log level changes
class LogLevelContext:
    """Context manager for temporarily changing log level."""
    
    def __init__(self, level: str):
        """
        Initialize context manager.
        
        Args:
            level: Temporary log level to use.
        """
        self.level = level
        self.original_level = None
    
    def __enter__(self):
        """Enter context - save current level and set new level."""
        logger = get_logger()
        self.original_level = logging.getLevelName(logger.level)
        set_log_level(self.level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original log level."""
        if self.original_level:
            set_log_level(self.original_level)
        return False


# Example usage:
if __name__ == "__main__":
    # Setup logger with custom settings
    logger = setup_logger(
        log_file="logs/test_speech_analysis.log",
        log_level="DEBUG",
        max_log_size_mb=5,
        backup_count=2,
        console_output=True
    )
    
    # Test logging at different levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test convenience functions
    info("Using convenience function")
    warning("This is a warning via convenience function")
    
    # Test context manager
    info("Current log level: INFO")
    with LogLevelContext("DEBUG"):
        debug("This debug message will be shown")
    debug("This debug message will NOT be shown (back to INFO)")
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        exception("An error occurred")
    
    # Shutdown
    shutdown_logger()
    print("Logger test complete")
