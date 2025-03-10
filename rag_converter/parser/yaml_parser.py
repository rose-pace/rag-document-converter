"""
YAML parser module for RAG Document Converter.

This module handles extraction and validation of YAML blocks in markdown documents.
"""

import re
import logging
from typing import List, Dict, Any
import yaml
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class YamlParser:
    """
    Parser class for extracting and validating YAML blocks.
    """
    
    def __init__(self):
        """
        Initialize the YAML parser.
        """
        logger.debug('YAML parser initialized')
    
    def extract_yaml_blocks(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract YAML blocks from document content.
        
        Args:
            content: Raw markdown content
            
        Returns:
            List of YAML blocks with their position and parsed content
        """
        logger.debug('Extracting YAML blocks')
        
        # Find all YAML blocks
        yaml_pattern = r'```yaml\s+(.*?)\s+```'
        yaml_matches = re.finditer(yaml_pattern, content, re.DOTALL)
        
        yaml_blocks = []
        for match in yaml_matches:
            try:
                parsed_yaml = yaml.safe_load(match.group(1))
                yaml_blocks.append({
                    'start': match.start(),
                    'end': match.end(),
                    'content': match.group(1),
                    'parsed': parsed_yaml
                })
            except yaml.YAMLError:
                logger.warning(f'Invalid YAML block at position {match.start()}')
        
        logger.debug(f'Extracted {len(yaml_blocks)} YAML blocks')
        return yaml_blocks
    
    def validate_yaml_block(self, yaml_content: str, schema_class: Any = None) -> Dict[str, Any]:
        """
        Validate YAML content against a schema.
        
        Args:
            yaml_content: Raw YAML content
            schema_class: Optional Pydantic model to validate against
            
        Returns:
            Dictionary containing the validated YAML data
            
        Raises:
            yaml.YAMLError: If the YAML is invalid
            ValidationError: If the YAML doesn't match the schema
        """
        try:
            # First, safely load the YAML
            data = yaml.safe_load(yaml_content)
            
            # If a schema class is provided, validate against it
            if schema_class:
                validated_data = schema_class(**data)
                return validated_data.dict()
            
            return data
            
        except yaml.YAMLError as e:
            logger.error(f'YAML parsing error: {str(e)}')
            raise
        except ValidationError as e:
            logger.error(f'YAML validation error: {str(e)}')
            raise
