"""
Document assembler module for RAG Document Converter.

This module contains the DocumentAssembler class that assembles the final
document from optimized components.
"""

import logging
from typing import Dict, Any, List, Optional

from rag_converter.config import load_config
from rag_converter.assembler.footer_generator import FooterGenerator
from rag_converter.assembler.notes_generator import NotesGenerator

logger = logging.getLogger(__name__)

class DocumentAssembler:
    """
    Assembles final document from optimized components.
    
    This class is responsible for combining all document components into
    a complete, well-structured document optimized for RAG.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the document assembler.
        
        Args:
            config_path: Optional path to a custom configuration file
        """
        self.config = load_config(config_path)
        self.footer_generator = FooterGenerator(config_path)
        self.notes_generator = NotesGenerator(config_path)
        
        logger.info('Document assembler initialized')
    
    def assemble_document(self, document_structure: Dict[str, Any]) -> str:
        """
        Assemble the final document from optimized components.
        
        Args:
            document_structure: Optimized document structure
            
        Returns:
            Complete document content as a string
        """
        logger.info('Assembling final document')
        
        # Get document title
        title = document_structure.get('title', 'Untitled Document')
        
        # Generate document notes section
        doc_notes = self.notes_generator.create_document_notes(document_structure)
        
        # Get document body from sections
        body = self._assemble_body(document_structure['sections'])
        
        # Generate document footer
        footer = self.footer_generator.create_footer(document_structure)
        
        # Get document summary if available
        summary = self._get_document_summary(document_structure)
        
        # Use template from configuration
        template = self.config.document_template.full_template
        
        # Fill template with content
        document = template.format(
            title=title,
            doc_notes=doc_notes,
            summary=summary,
            body=body,
            footer=footer,
            version=document_structure.get('doc_notes', {}).get('Document Version', '1.0'),
            date=document_structure.get('doc_notes', {}).get('Version Date', ''),
            collection=document_structure.get('doc_notes', {}).get('Collection', 'Uncategorized'),
            tags=self._format_tags(document_structure.get('doc_notes', {}).get('Tags', []))
        )
        
        logger.info('Document assembly completed')
        return document
    
    def _assemble_body(self, sections: List[Dict[str, Any]]) -> str:
        """
        Assemble the document body from sections.
        
        Args:
            sections: List of document sections
            
        Returns:
            Formatted document body
        """
        body = ''
        
        for section in sections:
            # Add section header
            body += f"{section['header']}\n\n"
            
            # Add section summary if available (for level 2 headers)
            if section.get('level') == 2 and 'summary' in section:
                body += f"{section['summary']}\n\n"
            
            # Add section content
            body += f"{section['content']}\n\n"
        
        return body.strip()
    
    def _get_document_summary(self, document_structure: Dict[str, Any]) -> str:
        """
        Get or generate the document summary.
        
        Args:
            document_structure: Document structure
            
        Returns:
            Document summary string
        """
        # Check if we have an explicit summary
        if 'summary' in document_structure:
            return document_structure['summary']
        
        # Otherwise, try to use the first section's summary if it's level 2
        if (document_structure['sections'] and 
            document_structure['sections'][0].get('level') == 2 and 
            'summary' in document_structure['sections'][0]):
            return document_structure['sections'][0]['summary']
        
        # Default to empty string if no summary found
        return ''
    
    def _format_tags(self, tags: List[str]) -> str:
        """
        Format tags for inclusion in YAML.
        
        Args:
            tags: List of tag strings
            
        Returns:
            Formatted tags string with proper indentation
        """
        return '\n'.join([f'  - {tag}' for tag in tags])
