"""
Section parser module for RAG Document Converter.

This module extracts document sections based on markdown headers.
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SectionParser:
    """
    Parser class for extracting document sections based on headers.
    """
    
    def __init__(self):
        """
        Initialize the section parser.
        """
        logger.debug('Section parser initialized')
    
    def extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract document sections based on headers.
        
        Args:
            content: Raw markdown content
            
        Returns:
            List of sections with their headers, level, and content
        """
        logger.debug('Extracting document sections')
        
        # Split content by headers
        header_pattern = r'^(#{1,6} .+)$'
        parts = re.split(header_pattern, content, flags=re.MULTILINE)
        
        sections = []
        current_header = None
        current_level = 0
        current_content = ''
        
        for i, part in enumerate(parts):
            if i % 2 == 1:  # This is a header
                if current_header:
                    # Save previous section
                    sections.append({
                        'header': current_header,
                        'level': current_level,
                        'content': current_content.strip()
                    })
                
                # Start new section
                current_header = part
                current_level = len(re.match(r'^(#+)', part).group(1))
                current_content = ''
            elif i % 2 == 0 and i > 0:  # This is content
                current_content = part
        
        # Add the last section
        if current_header:
            sections.append({
                'header': current_header,
                'level': current_level,
                'content': current_content.strip()
            })
        
        logger.debug(f'Extracted {len(sections)} sections')
        return sections
    
    def get_section_hierarchy(self, sections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert flat section list to nested hierarchy.
        
        Args:
            sections: List of sections from extract_sections()
            
        Returns:
            Nested dictionary representing section hierarchy
        """
        hierarchy = {'title': 'Document Root', 'level': 0, 'children': []}
        stack = [hierarchy]
        
        for section in sections:
            level = section['level']
            
            # Find the appropriate parent for this section
            while len(stack) > 1 and stack[-1]['level'] >= level:
                stack.pop()
            
            # Create section node
            section_node = {
                'title': section['header'].replace('#' * level, '').strip(),
                'level': level,
                'content': section['content'],
                'children': []
            }
            
            # Add to parent
            stack[-1]['children'].append(section_node)
            
            # Add to stack
            stack.append(section_node)
        
        return hierarchy
