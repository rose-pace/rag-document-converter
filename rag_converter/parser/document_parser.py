"""
Document parser module for RAG Document Converter.

This module contains the DocumentParser class that orchestrates 
the parsing of document structure by integrating other parser components.
"""

import logging
from typing import Dict, Any, Optional

from rag_converter.parser.section_parser import SectionParser
from rag_converter.parser.yaml_parser import YamlParser
from rag_converter.parser.entity_parser import EntityParser
from rag_converter.parser.document_notes_parser import DocumentNotesParser
from rag_converter.config import load_config, ConverterConfig

logger = logging.getLogger(__name__)

class DocumentParser:
    """
    Main parser class that orchestrates document parsing.
    
    This class integrates section, yaml, entity, and document notes parsers
    to analyze the structure of a markdown document.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the document parser.
        
        Args:
            config_path: Optional path to a custom configuration file
        """
        self.config = load_config(config_path)
        self.section_parser = SectionParser()
        self.yaml_parser = YamlParser()
        self.entity_parser = EntityParser(self.config.entity_patterns)
        self.notes_parser = DocumentNotesParser()
        
        logger.info('Document parser initialized')
    
    def parse_document(self, content: str) -> Dict[str, Any]:
        """
        Parse the markdown document structure.
        
        Args:
            content: Raw markdown content
            
        Returns:
            Dictionary containing parsed document structure
        """
        logger.info('Parsing document structure')
        
        # Extract title (first H1)
        title = self._extract_title(content)
        
        # Extract document sections based on headers
        sections = self.section_parser.extract_sections(content)
        
        # Extract YAML blocks
        yaml_blocks = self.yaml_parser.extract_yaml_blocks(content)
        
        # Extract document notes if present
        doc_notes = self.notes_parser.extract_document_notes(content)
        
        # Extract entities
        entities = self.entity_parser.extract_entities(content)
        
        # Assemble document structure
        document_structure = {
            'title': title,
            'doc_notes': doc_notes,
            'sections': sections,
            'yaml_blocks': yaml_blocks,
            'entities': entities,
            'raw_content': content
        }
        
        logger.info(f'Document parsing completed: found {len(sections)} sections and {len(entities)} entities')
        return document_structure
    
    def _extract_title(self, content: str) -> str:
        """
        Extract document title from content.
        
        Args:
            content: Raw markdown content
            
        Returns:
            Document title or 'Untitled Document' if not found
        """
        import re
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        return title_match.group(1) if title_match else 'Untitled Document'
