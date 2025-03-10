"""
Entity identifier module for RAG Document Converter.

This module handles the generation and application of standardized identifiers
for entities in the document.
"""

import re
import logging
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from rag_converter.config import ConverterConfig

logger = logging.getLogger(__name__)

class EntityIdentifier:
    """
    Class for generating and applying entity identifiers.
    
    This class creates standardized identifiers for entities and applies them
    throughout the document content.
    """
    
    def __init__(self, config: ConverterConfig):
        """
        Initialize the entity identifier.
        
        Args:
            config: Configuration containing entity prefix definitions
        """
        self.config = config
        logger.info('Entity identifier initialized')
    
    def generate_entity_identifiers(self, entities: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate standardized identifiers for entities.
        
        Args:
            entities: List of entities extracted from the document
            
        Returns:
            Dict mapping entity names to standardized identifiers
        """
        identifiers = {}
        
        for entity in entities:
            entity_type = entity.get('type', 'concept').lower()
            entity_name = entity.get('name', 'unknown')
            
            # Get appropriate prefix from config
            prefix_obj = self.config.entity_prefixes.get(entity_type)
            prefix = prefix_obj.prefix if prefix_obj else 'UNK'
            
            # Generate subtype component if available
            subtype = ''
            if entity_type == 'deity' and 'pantheon' in entity:
                pantheon = entity.get('pantheon', '').lower()
                if 'archosian' in pantheon:
                    subtype = 'ARCH'
                elif 'nefic' in pantheon:
                    subtype = 'NEF'
                elif pantheon:
                    # Use first 4 chars of pantheon name if provided
                    subtype = re.sub(r'[^A-Za-z0-9]', '', pantheon).upper()[:4]
            
            # Generate name component
            name_component = re.sub(r'[^A-Za-z0-9]', '', entity_name).upper()
            if len(name_component) > 10:
                name_component = name_component[:10]
            
            # Assemble identifier
            if subtype:
                identifier = f'{prefix}_{subtype}_{name_component}'
            else:
                identifier = f'{prefix}_{name_component}'
            
            # Ensure identifier is unique
            counter = 1
            base_identifier = identifier
            while identifier in identifiers.values():
                identifier = f'{base_identifier}_{counter}'
                counter += 1
            
            identifiers[entity_name] = identifier
            logger.debug(f'Generated identifier {identifier} for entity {entity_name}')
        
        logger.info(f'Generated {len(identifiers)} entity identifiers')
        return identifiers
    
    def apply_entity_identifiers(self, document_structure: Dict[str, Any], 
                                identifiers: Dict[str, str]) -> Dict[str, Any]:
        """
        Apply entity identifiers to document content.
        
        Args:
            document_structure: Parsed document structure
            identifiers: Dict mapping entity names to identifiers
            
        Returns:
            Updated document structure with identifiers applied
        """
        logger.info('Applying entity identifiers to document')
        
        # Apply to sections
        for i, section in enumerate(document_structure['sections']):
            content = section['content']
            
            # For each entity, add identifier on first mention
            for entity_name, identifier in identifiers.items():
                # Create pattern to find exact entity name with word boundaries
                pattern = r'\b' + re.escape(entity_name) + r'\b(?!\s*\[' + re.escape(identifier) + r'\])'
                
                # Replace first occurrence only
                content = re.sub(pattern, f'{entity_name} [{identifier}]', content, count=1)
            
            # Update section content
            document_structure['sections'][i]['content'] = content
            
            # Also update header if needed
            header = section['header']
            for entity_name, identifier in identifiers.items():
                if entity_name in header and identifier not in header:
                    document_structure['sections'][i]['header'] = header.replace(
                        entity_name, 
                        f'{entity_name} [{identifier}]'
                    )
        
        # Update entities with their identifiers
        for i, entity in enumerate(document_structure['entities']):
            entity_name = entity.get('name')
            if entity_name and entity_name in identifiers:
                document_structure['entities'][i]['identifier'] = identifiers[entity_name]
        
        logger.info('Entity identifiers applied to document')
        return document_structure
