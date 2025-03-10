"""
YAML processing utilities for RAG Document Converter.

This module provides functions for safely loading and dumping YAML content,
extracting YAML blocks from text, and validating YAML structures.
"""

import yaml
from pathlib import Path
import re
from typing import Dict, Any, List, Optional, Union, TextIO

from rag_converter.utils.logging_utils import get_logger

logger = get_logger(__name__)


def safe_load_yaml(content: Union[str, TextIO]) -> Dict[str, Any]:
    """
    Safely load YAML content.
    
    Args:
        content: YAML content as string or file-like object
    
    Returns:
        Dictionary containing parsed YAML
        
    Raises:
        yaml.YAMLError: If YAML parsing fails
    """
    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as e:
        logger.error(f'Error parsing YAML: {e}')
        raise


def safe_load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely load YAML from a file.
    
    Args:
        file_path: Path to the YAML file
    
    Returns:
        Dictionary containing parsed YAML
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    
    if not file_path.exists():
        logger.error(f'YAML file not found: {file_path}')
        raise FileNotFoundError(f'File not found: {file_path}')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            return safe_load_yaml(f)
        except yaml.YAMLError as e:
            logger.error(f'Error parsing YAML file {file_path}: {e}')
            raise


def safe_dump_yaml(data: Any, stream: Optional[TextIO] = None, **kwargs) -> Optional[str]:
    """
    Safely dump data to YAML.
    
    Args:
        data: Python object to dump to YAML
        stream: Optional stream to write to
        **kwargs: Additional arguments to yaml.dump
    
    Returns:
        YAML string if stream is None, otherwise None
        
    Raises:
        yaml.YAMLError: If YAML dumping fails
    """
    try:
        return yaml.dump(data, stream=stream, default_flow_style=False, sort_keys=False, **kwargs)
    except yaml.YAMLError as e:
        logger.error(f'Error dumping to YAML: {e}')
        raise


def extract_yaml_blocks(content: str) -> List[Dict[str, Any]]:
    """
    Extract YAML blocks from text content.
    
    Args:
        content: Text content to extract YAML blocks from
    
    Returns:
        List of dictionaries containing:
        - start: Start position of the block
        - end: End position of the block
        - content: Raw YAML content
        - parsed: Parsed YAML content (or None if invalid)
    """
    yaml_blocks = []
    yaml_pattern = r'```yaml\s+(.*?)\s+```'
    yaml_matches = re.finditer(yaml_pattern, content, re.DOTALL)
    
    for match in yaml_matches:
        try:
            parsed_yaml = safe_load_yaml(match.group(1))
            yaml_blocks.append({
                'start': match.start(),
                'end': match.end(),
                'content': match.group(1),
                'parsed': parsed_yaml
            })
        except yaml.YAMLError:
            logger.warning(f'Invalid YAML block at position {match.start()}')
            yaml_blocks.append({
                'start': match.start(),
                'end': match.end(),
                'content': match.group(1),
                'parsed': None
            })
    
    return yaml_blocks


def validate_yaml_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate YAML data against a simple schema.
    
    Args:
        data: YAML data to validate
        schema: Schema dictionary specifying required fields and types
    
    Returns:
        True if valid, False otherwise
    """
    # Simple schema validation - a more robust solution would use jsonschema
    for field, field_type in schema.items():
        if field not in data:
            logger.warning(f'Required field missing: {field}')
            return False
        
        if not isinstance(data[field], field_type):
            logger.warning(f'Field {field} has wrong type: expected {field_type}, got {type(data[field])}')
            return False
    
    return True
