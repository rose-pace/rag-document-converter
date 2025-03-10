"""
Entity parser module for RAG Document Converter.

This module identifies potential named entities in document content.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class Entity(BaseModel):
    """
    Model for a detected entity.
    
    Args:
        name: Entity name
        type: Entity type (deity, location, etc.)
        position: Position in the document
        context: Surrounding context
        confidence: Detection confidence (0.0-1.0)
    """
    name: str
    type: str
    position: int
    context: Optional[str] = None
    confidence: float = 1.0

class EntityParser:
    """
    Parser class for identifying named entities in content.
    """
    
    def __init__(self, entity_patterns: List[Dict[str, Any]] = None):
        """
        Initialize the entity parser.
        
        Args:
            entity_patterns: List of entity patterns for detection
        """
        self.entity_patterns = entity_patterns or []
        logger.debug('Entity parser initialized')
    
    def extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract potential named entities from document content.
        
        Args:
            content: Raw markdown content
            
        Returns:
            List of detected entities
        """
        logger.debug('Extracting named entities')
        
        entities = []
        
        # Apply each pattern to the content
        for pattern_def in self.entity_patterns:
            entity_type = pattern_def.get('type', 'unknown')
            patterns = pattern_def.get('patterns', [])
            
            for pattern in patterns:
                try:
                    # Apply the regex pattern to find matches
                    matches = re.finditer(pattern, content)
                    
                    for match in matches:
                        # The first group usually contains the entity name
                        if match.groups():
                            entity_name = match.group(1)
                        else:
                            entity_name = match.group(0)
                        
                        # Extract some context around the match
                        start_ctx = max(0, match.start() - 30)
                        end_ctx = min(len(content), match.end() + 30)
                        context = content[start_ctx:end_ctx]
                        
                        # Create entity object
                        entity = Entity(
                            name=entity_name,
                            type=entity_type,
                            position=match.start(),
                            context=context,
                            confidence=0.8  # Default confidence for pattern matches
                        )
                        
                        entities.append(entity.dict())
                except re.error as e:
                    logger.error(f'Invalid regex pattern for {entity_type}: {pattern}. Error: {str(e)}')
        
        # If no patterns defined, use basic entity detection
        if not self.entity_patterns:
            basic_entities = self._basic_entity_detection(content)
            entities.extend(basic_entities)
        
        # Remove duplicates
        unique_entities = self._remove_duplicate_entities(entities)
        
        logger.debug(f'Extracted {len(unique_entities)} entities')
        return unique_entities
    
    def _basic_entity_detection(self, content: str) -> List[Dict[str, Any]]:
        """
        Perform basic entity detection with common patterns.
        
        Args:
            content: Raw markdown content
            
        Returns:
            List of detected entities
        """
        entities = []
        
        # Basic patterns for common entity types
        patterns = [
            # Deities
            (r'(?:God|Goddess|Deity) of ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', 'deity'),
            # Locations
            (r'(?:The |the )?((?:[A-Z][a-z]+ )*(?:Kingdom|Empire|Realm|Forest|Mountain|Sea|Ocean|River|Lake|City|Town|Village|Land|Island|Plane))', 'location'),
            # Events
            (r'(?:The |the )?((?:[A-Z][a-z]+ )*(?:War|Battle|Invasion|Conflict|Rebellion|Revolution|Coronation|Event))', 'event'),
            # Items
            (r'(?:The |the )?((?:[A-Z][a-z]+ )*(?:Sword|Axe|Staff|Wand|Shield|Armor|Crown|Ring|Amulet|Talisman|Artifact))', 'item'),
            # Factions
            (r'(?:The |the )?((?:[A-Z][a-z]+ )*(?:Guild|Order|Brotherhood|Sisterhood|Cult|Faction|Clan|Tribe|House|Family))', 'faction'),
        ]
        
        for pattern, entity_type in patterns:
            matches = re.finditer(pattern, content)
            
            for match in matches:
                entity_name = match.group(1)
                
                # Extract context
                start_ctx = max(0, match.start() - 30)
                end_ctx = min(len(content), match.end() + 30)
                context = content[start_ctx:end_ctx]
                
                entity = Entity(
                    name=entity_name,
                    type=entity_type,
                    position=match.start(),
                    context=context,
                    confidence=0.6  # Lower confidence for basic detection
                )
                
                entities.append(entity.dict())
        
        return entities
    
    def _remove_duplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate entities based on name and type.
        
        Args:
            entities: List of detected entities
            
        Returns:
            Deduplicated list of entities
        """
        unique_entities = {}
        
        for entity in entities:
            key = f"{entity['name']}|{entity['type']}"
            
            if key not in unique_entities or entity['confidence'] > unique_entities[key]['confidence']:
                unique_entities[key] = entity
        
        return list(unique_entities.values())
