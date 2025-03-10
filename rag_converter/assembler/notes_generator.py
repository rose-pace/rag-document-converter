"""
Notes generator module for RAG Document Converter.

This module contains the NotesGenerator class that creates or updates the
document notes section with metadata and tags.
"""

import logging
import datetime
from typing import Dict, Any, List, Optional, Set
import re

from rag_converter.config import load_config

logger = logging.getLogger(__name__)

class NotesGenerator:
    """
    Creates or updates document notes section.
    
    This class handles the generation and formatting of the document notes
    section, including metadata and tags.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the notes generator.
        
        Args:
            config_path: Optional path to a custom configuration file
        """
        self.config = load_config(config_path)
        logger.info('Notes generator initialized')
    
    def create_document_notes(self, document_structure: Dict[str, Any]) -> str:
        """
        Create or update document notes section.
        
        Args:
            document_structure: Document structure with metadata
            
        Returns:
            Formatted document notes section
        """
        logger.info('Creating document notes section')
        
        # Get existing notes or create new ones
        notes = document_structure.get('doc_notes', {})
        
        # Ensure all required fields are present
        notes = self._ensure_required_fields(notes, document_structure)
        
        # Generate tags if needed
        if 'Tags' not in notes or not notes['Tags']:
            notes['Tags'] = self._generate_tags(document_structure)
        
        # Format the tags
        tags_str = self._format_tags(notes['Tags'])
        
        # Assemble document notes section
        doc_notes = f"""## Document Notes
```yaml
Document Version: {notes['Document Version']}
Version Date: {notes['Version Date']}
Collection: {notes['Collection']}
Tags:
{tags_str}
```"""
        
        logger.debug('Document notes section created')
        return doc_notes
    
    def _ensure_required_fields(self, notes: Dict[str, Any], document_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure all required fields are present in document notes.
        
        Args:
            notes: Existing document notes
            document_structure: Complete document structure
            
        Returns:
            Updated notes with all required fields
        """
        # Make a copy to avoid modifying the original
        updated_notes = notes.copy()
        
        # Ensure document version
        if 'Document Version' not in updated_notes:
            updated_notes['Document Version'] = '1.0'
        
        # Ensure version date
        if 'Version Date' not in updated_notes:
            updated_notes['Version Date'] = datetime.datetime.now().strftime('%Y-%m-%d')
        
        # Determine collection if not present
        if 'Collection' not in updated_notes:
            updated_notes['Collection'] = self._determine_collection(document_structure)
        
        return updated_notes
    
    def _determine_collection(self, document_structure: Dict[str, Any]) -> str:
        """
        Determine appropriate collection for the document.
        
        Args:
            document_structure: Document structure with content
            
        Returns:
            Collection name
        """
        title = document_structure.get('title', '').lower()
        content = document_structure.get('raw_content', '').lower()
        entities = document_structure.get('entities', [])
        
        # Check entity types present
        entity_types = [entity.get('type', '') for entity in entities]
        
        # Check title and content for collection clues
        if any(term in title for term in ['deity', 'god', 'pantheon', 'divine']):
            return 'Cosmology'
        elif any(term in title for term in ['location', 'city', 'region', 'place']):
            return 'Geography'
        elif any(term in title for term in ['history', 'timeline', 'event']):
            return 'History'
        elif any(term in title for term in ['item', 'artifact', 'weapon']):
            return 'Items'
        elif any(term in title for term in ['faction', 'organization', 'guild']):
            return 'Organizations'
        
        # Check entity types if no clues in title
        if 'deity' in entity_types:
            return 'Cosmology'
        elif 'location' in entity_types:
            return 'Geography'
        elif 'event' in entity_types:
            return 'History'
        elif 'item' in entity_types:
            return 'Items'
        elif 'faction' in entity_types:
            return 'Organizations'
        
        # Default collection
        return 'Uncategorized'
    
    def _generate_tags(self, document_structure: Dict[str, Any]) -> List[str]:
        """
        Generate tags based on document content.
        
        Args:
            document_structure: Document structure with content and entities
            
        Returns:
            List of generated tags
        """
        tags = set()
        title = document_structure.get('title', '').lower()
        content = document_structure.get('raw_content', '').lower()
        entities = document_structure.get('entities', [])
        
        # Add tags based on entity types
        entity_types = [entity.get('type', '') for entity in entities]
        for entity_type in entity_types:
            if entity_type:
                tags.add(f'{entity_type}s')  # Pluralize entity type
        
        # Add tags based on title
        if 'pantheon' in title:
            tags.add('pantheons')
        if any(term in title for term in ['deity', 'god', 'goddess']):
            tags.add('deities')
        if any(term in title for term in ['history', 'chronicle', 'timeline']):
            tags.add('history')
        if any(term in title for term in ['magic', 'spell', 'arcane']):
            tags.add('magic')
        if 'divine' in title:
            tags.add('divine')
        
        # Add collection as a tag
        collection = document_structure.get('doc_notes', {}).get('Collection', '')
        if collection and collection != 'Uncategorized':
            tags.add(collection.lower())
        
        # Add default tags if we have too few
        if len(tags) < 3:
            default_tags = ['rpg', 'setting', 'lore', 'worldbuilding']
            for tag in default_tags:
                if len(tags) >= 3:
                    break
                if tag not in tags:
                    tags.add(tag)
        
        return sorted(list(tags))
    
    def _format_tags(self, tags: List[str]) -> str:
        """
        Format tags for YAML inclusion.
        
        Args:
            tags: List of tag strings
            
        Returns:
            Formatted tags string
        """
        return '\n'.join([f'  - {tag}' for tag in tags])
