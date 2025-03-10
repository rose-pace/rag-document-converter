"""
Logging utilities for RAG Document Converter.

This module provides functions for configuring logging and obtaining 
loggers with consistent settings.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    log_to_console: bool = True
) -> None:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level (default: logging.INFO)
        log_file: Optional path to log file
        format_string: Optional custom format string
        log_to_console: Whether to log to console
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        log_file = Path(log_file) if isinstance(log_file, str) else log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Name for the logger
        level: Optional specific level for this logger
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        logger.setLevel(level)
    
    return logger


class LogContext:
    """
    Context manager for adding context to log messages.
    
    Usage:
        with LogContext(logger, {'document': 'example.md'}):
            logger.info('Processing')  # Will include context in log
    """
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        """
        Initialize the log context.
        
        Args:
            logger: Logger to add context to
            context: Dictionary of context information to add
        """
        self.logger = logger
        self.context = context
        self.old_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        """
        Enter the context manager.
        """
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager.
        """
        logging.setLogRecordFactory(self.old_factory)
