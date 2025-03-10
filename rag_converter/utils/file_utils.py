"""
File handling utilities for RAG Document Converter.

This module provides functions for working with files and directories
including reading, writing, and path manipulation.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Union, List, Iterator

from rag_converter.utils.logging_utils import get_logger

logger = get_logger(__name__)


def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    
    Returns:
        Path object for the directory
    """
    directory = Path(directory) if isinstance(directory, str) else directory
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_default_encoding() -> str:
    """
    Get the default file encoding from environment variable or use utf-8.
    
    Returns:
        Encoding string to use for file operations
    """
    return os.environ.get('FILE_ENCODING', 'utf-8')


def read_file(file_path: Union[str, Path], encoding: Optional[str] = None) -> str:
    """
    Read text from a file.
    
    Args:
        file_path: Path to the file
        encoding: Character encoding (defaults to FILE_ENCODING env var or utf-8)
    
    Returns:
        File contents as string
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If the file can't be read
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    encoding = encoding or get_default_encoding()
    
    if not file_path.exists():
        logger.error(f'File not found: {file_path}')
        raise FileNotFoundError(f'File not found: {file_path}')
    
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f'Error reading file {file_path}: {str(e)}')
        raise


def write_file(file_path: Union[str, Path], content: str, encoding: Optional[str] = None) -> None:
    """
    Write text to a file.
    
    Args:
        file_path: Path to the file
        content: Content to write
        encoding: Character encoding (defaults to FILE_ENCODING env var or utf-8)
    
    Raises:
        IOError: If the file can't be written
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    encoding = encoding or get_default_encoding()
    
    # Ensure parent directory exists
    ensure_directory(file_path.parent)
    
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
    except Exception as e:
        logger.error(f'Error writing to file {file_path}: {str(e)}')
        raise


def find_files(directory: Union[str, Path], pattern: str = '*.md') -> List[Path]:
    """
    Find files matching a pattern in a directory.
    
    Args:
        directory: Directory to search in
        pattern: Glob pattern to match files
    
    Returns:
        List of matching file paths
    """
    directory = Path(directory) if isinstance(directory, str) else directory
    
    if not directory.exists():
        logger.warning(f'Directory not found: {directory}')
        return []
    
    return list(directory.glob(pattern))


def copy_file(source: Union[str, Path], destination: Union[str, Path], 
             overwrite: bool = False) -> Path:
    """
    Copy a file from source to destination.
    
    Args:
        source: Source file path
        destination: Destination file path
        overwrite: Whether to overwrite if destination exists
    
    Returns:
        Path to the destination file
    
    Raises:
        FileNotFoundError: If source doesn't exist
        FileExistsError: If destination exists and overwrite is False
    """
    source = Path(source) if isinstance(source, str) else source
    destination = Path(destination) if isinstance(destination, str) else destination
    
    if not source.exists():
        logger.error(f'Source file not found: {source}')
        raise FileNotFoundError(f'Source file not found: {source}')
    
    if destination.exists() and not overwrite:
        logger.error(f'Destination file exists: {destination}')
        raise FileExistsError(f'Destination file exists: {destination}')
    
    # Ensure parent directory exists
    ensure_directory(destination.parent)
    
    try:
        shutil.copy2(source, destination)
        return destination
    except Exception as e:
        logger.error(f'Error copying file from {source} to {destination}: {str(e)}')
        raise
