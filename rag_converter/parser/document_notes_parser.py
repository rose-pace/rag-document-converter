"""
Document notes parser module for RAG Document Converter.

This module extracts document metadata section from markdown documents.
"""

import re
import logging
import datetime
from typing import Dict, Any
import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class DocumentNotes(BaseModel):
    """
    Model for document notes/metadata.
    
    Args:
        document_version: Document version number
        version_date: Date of the version
        collection: Collection the document belongs to
        tags: List of document tags
        author: Optional document author
        status: Optional document status
    """
    document_version: str = Field(default='1.0', alias='Document Version')
    version_date: str = Field(default_factory=lambda: datetime.datetime.now().strftime('%Y-%m-%d'), alias='Version Date')
    collection: str = Field(default='Uncategorized', alias='Collection')
    tags: list = Field(default_factory=lambda: ['uncategorized'], alias='Tags')
    author: str = Field(default='', alias='Author')
    status: str = Field(default='Draft', alias='Status')
    
    class Config:
        allow_population_by_field_name = True

class DocumentNotesParser:
    """
    Parser class for extracting document metadata section.
    """
    
    def __init__(self):
        """
        Initialize the document notes parser.
        """
        logger.debug('Document notes parser initialized')
    
    def extract_document_notes(self, content: str) -> Dict[str, Any]:
        """
        Extract document notes section if present.
        
        Args:
            content: Raw markdown content
            
        Returns:
            Dictionary containing document notes or default values
        """
        logger.debug('Extracting document notes')
        
        # Look for document notes section with YAML
        doc_notes_pattern = r'## Document Notes\s+```yaml\s+(.*?)\s+```'
        match = re.search(doc_notes_pattern, content, re.DOTALL)
        
        if match:
            # Parse existing YAML
            try:
                notes_yaml = yaml.safe_load(match.group(1))
                # Validate with the model
                notes = DocumentNotes(**notes_yaml)
                return notes.dict(by_alias=True)
            except yaml.YAMLError:
                logger.warning('Invalid YAML in document notes, using defaults')
            except Exception as e:
                logger.warning(f'Error parsing document notes: {str(e)}')
        
        # Attempt to infer some metadata from content if no notes section
        inferred_notes = self._infer_metadata(content)
        
        # Create default document notes
        notes = DocumentNotes(**inferred_notes)
        return notes.dict(by_alias=True)
    
    def _infer_metadata(self, content: str) -> Dict[str, Any]:
        """
        Infer document metadata from content.
        
        Args:
            content: Raw markdown content
            
        Returns:
            Dictionary of inferred metadata
        """
        inferred = {}
        
        # Try to determine collection from content
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).lower()
            
            if any(term in title for term in ['deity', 'god', 'pantheon', 'divine']):
                inferred['Collection'] = 'Cosmology'
            elif any(term in title for term in ['location', 'city', 'region', 'geography']):
                inferred['Collection'] = 'Geography'
            elif any(term in title for term in ['creature', 'monster', 'beast']):
                inferred['Collection'] = 'Bestiary'
            elif any(term in title for term in ['item', 'artifact', 'equipment']):
                inferred['Collection'] = 'Items'
            elif any(term in title for term in ['faction', 'guild', 'organization']):
                inferred['Collection'] = 'Organizations'
            elif any(term in title for term in ['event', 'history', 'timeline']):
                inferred['Collection'] = 'History'
        
        # Generate tags based on content
        tags = []
        
        # Check for common topics in content
        if re.search(r'\b(?:deity|deities|god|goddess|pantheon|divine)\b', content, re.IGNORECASE):
            tags.append('deities')
        if re.search(r'\b(?:magic|spell|arcane|mystic)\b', content, re.IGNORECASE):
            tags.append('magic')
        if re.search(r'\b(?:war|battle|conflict|fight)\b', content, re.IGNORECASE):
            tags.append('conflict')
        if re.search(r'\b(?:king|queen|ruler|sovereign|throne)\b', content, re.IGNORECASE):
            tags.append('royalty')
        
        # Add at least one tag if none were found
        if not tags:
            tags = ['lore']
        
        # Add general tags
        tags.extend(['rpg', 'setting'])
        
        # Deduplicate tags
        inferred['Tags'] = list(dict.fromkeys(tags))
        
        return inferred
