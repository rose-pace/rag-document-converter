"""
Vocabulary controller module for RAG Document Converter.

This module handles the application of controlled vocabulary terms
to maintain consistent terminology throughout the document.
"""

import re
import logging
from typing import Dict, List, Any

from rag_converter.config import ConverterConfig

logger = logging.getLogger(__name__)

class VocabularyController:
    """
    Class for applying controlled vocabulary to document content.
    
    This class ensures consistent terminology by replacing alternative terms
    with preferred terms from the controlled vocabulary.
    """
    
    def __init__(self, config: ConverterConfig):
        """
        Initialize the vocabulary controller.
        
        Args:
            config: Configuration containing controlled vocabulary definitions
        """
        self.config = config
        logger.info('Vocabulary controller initialized')
    
    def apply_controlled_vocabulary(self, document_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply controlled vocabulary to document content.
        
        Args:
            document_structure: Parsed document structure
            
        Returns:
            Updated document structure with standardized vocabulary
        """
        logger.info('Applying controlled vocabulary to document content')
        
        # Process each section
        for i, section in enumerate(document_structure['sections']):
            content = section['content']
            
            # For each term in controlled vocabulary, replace alternatives with preferred term
            for term_name, term_obj in self.config.controlled_vocabulary.items():
                preferred_term = term_obj.preferred
                alternatives = term_obj.alternatives
                
                for alt_term in alternatives:
                    # Skip if alternative is empty or same as preferred
                    if not alt_term or alt_term == preferred_term:
                        continue
                        
                    # Create pattern to find exact alternative term with word boundaries
                    pattern = r'\b' + re.escape(alt_term) + r'\b'
                    
                    # Replace occurrences
                    content = re.sub(pattern, preferred_term, content)
            
            # Update section content
            document_structure['sections'][i]['content'] = content
            
            # Also standardize the header
            header = section['header']
            for term_name, term_obj in self.config.controlled_vocabulary.items():
                preferred_term = term_obj.preferred
                alternatives = term_obj.alternatives
                
                for alt_term in alternatives:
                    if alt_term and alt_term != preferred_term and alt_term in header:
                        document_structure['sections'][i]['header'] = header.replace(alt_term, preferred_term)
        
        # Also standardize the document title
        title = document_structure.get('title', '')
        for term_name, term_obj in self.config.controlled_vocabulary.items():
            preferred_term = term_obj.preferred
            alternatives = term_obj.alternatives
            
            for alt_term in alternatives:
                if alt_term and alt_term != preferred_term and alt_term in title:
                    document_structure['title'] = title.replace(alt_term, preferred_term)
        
        logger.info('Controlled vocabulary applied')
        return document_structure
    
    def check_vocabulary_consistency(self, document_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check document content for vocabulary consistency issues.
        
        Args:
            document_structure: Parsed document structure
            
        Returns:
            List of inconsistency issues found
        """
        issues = []
        
        # Compile all content for checking
        all_content = document_structure.get('title', '')
        for section in document_structure.get('sections', []):
            all_content += ' ' + section.get('header', '')
            all_content += ' ' + section.get('content', '')
        
        # Check for each alternative term
        for term_name, term_obj in self.config.controlled_vocabulary.items():
            preferred_term = term_obj.preferred
            alternatives = term_obj.alternatives
            
            for alt_term in alternatives:
                # Skip if alternative is empty or same as preferred
                if not alt_term or alt_term == preferred_term:
                    continue
                    
                # Look for occurrences of the alternative term
                pattern = r'\b' + re.escape(alt_term) + r'\b'
                matches = re.finditer(pattern, all_content)
                
                for match in matches:
                    issues.append({
                        'type': 'vocabulary_inconsistency',
                        'term': alt_term,
                        'preferred': preferred_term,
                        'position': match.start()
                    })
        
        if issues:
            logger.warning(f'Found {len(issues)} vocabulary consistency issues')
            
        return issues
