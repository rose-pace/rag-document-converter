"""
String manipulation utilities for RAG Document Converter.

This module provides functions for working with text content including
regex helpers, string normalization, and entity extraction.
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional, Pattern, Match, Tuple


def normalize_string(text: str) -> str:
    """
    Normalize a string by removing special characters and extra whitespace.
    
    Args:
        text: Input string
    
    Returns:
        Normalized string
    """
    # Remove diacritics and normalize to ASCII where possible
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])
    
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_text_between(text: str, start_pattern: str, end_pattern: str) -> List[str]:
    """
    Extract all text between two patterns.
    
    Args:
        text: Input text
        start_pattern: Regex pattern marking the start
        end_pattern: Regex pattern marking the end
    
    Returns:
        List of extracted text segments
    """
    pattern = f'{start_pattern}(.*?){end_pattern}'
    matches = re.finditer(pattern, text, re.DOTALL)
    return [match.group(1) for match in matches]


def find_entities(text: str, patterns: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Find entities in text using regex patterns.
    
    Args:
        text: Input text
        patterns: Dictionary mapping entity types to lists of regex patterns
    
    Returns:
        List of entity dictionaries with type, name, and position
    """
    entities = []
    
    for entity_type, entity_patterns in patterns.items():
        for pattern in entity_patterns:
            compiled_pattern = re.compile(pattern)
            for match in compiled_pattern.finditer(text):
                # Extract actual entity name (assumes first capture group is the entity name)
                entity_name = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
                
                entities.append({
                    'name': entity_name,
                    'type': entity_type,
                    'position': match.start(),
                    'match': match.group(0)
                })
    
    # Sort by position
    entities.sort(key=lambda e: e['position'])
    return entities


def apply_entity_identifiers(text: str, entities_map: Dict[str, str]) -> str:
    """
    Apply entity identifiers to text.
    
    Args:
        text: Input text
        entities_map: Dictionary mapping entity names to identifiers
    
    Returns:
        Text with entity identifiers applied
    """
    result = text
    
    # Sort entity names by length (longest first) to avoid partial replacements
    sorted_entities = sorted(entities_map.keys(), key=len, reverse=True)
    
    for entity_name in sorted_entities:
        identifier = entities_map[entity_name]
        # Create pattern to find entity name with word boundaries, not already tagged
        pattern = rf'\b{re.escape(entity_name)}\b(?!\s*\[{re.escape(identifier)}\])'
        
        # Replace first occurrence only
        result = re.sub(pattern, f'{entity_name} [{identifier}]', result, count=1)
    
    return result


def extract_relationships(text: str, patterns: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
    """
    Extract relationships from text using patterns.
    
    Args:
        text: Input text
        patterns: List of (pattern, relationship_type) tuples
    
    Returns:
        List of relationship dictionaries
    """
    relationships = []
    
    for pattern, rel_type in patterns:
        matches = re.finditer(pattern, text)
        
        for match in matches:
            source = match.group(1)
            relation = match.group(2)
            target = match.group(3)
            
            relationships.append({
                'type': rel_type,
                'source': source,
                'relation': relation,
                'target': target,
                'description': match.group(0)
            })
    
    return relationships
