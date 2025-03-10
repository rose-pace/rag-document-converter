"""
Section summarizer module for RAG Document Converter.

This module handles the generation of concise summaries for document sections,
with optional LLM-based enhancement.
"""

import re
import logging
from typing import Dict, List, Any, Optional

from rag_converter.llm.client import LLMClient

logger = logging.getLogger(__name__)

class SectionSummarizer:
    """
    Class for generating summaries for document sections.
    
    This class creates concise summaries of document sections, either using
    simple extraction or LLM-based summarization if enabled.
    """
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize the section summarizer.
        
        Args:
            use_llm: Whether to use LLM for enhanced summarization
        """
        self.use_llm = use_llm
        logger.info('Section summarizer initialized')
    
    def generate_section_summaries(self, document_structure: Dict[str, Any],
                                 llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
        """
        Generate summaries for major document sections.
        
        Args:
            document_structure: Parsed document structure
            llm_client: Optional LLM client for enhanced summarization
            
        Returns:
            Updated document structure with section summaries
        """
        logger.info('Generating section summaries')
        
        # If LLM is enabled and client is provided, use it for enhanced summaries
        if self.use_llm and llm_client:
            return self._generate_summaries_with_llm(document_structure, llm_client)
        
        # Otherwise use simple extraction
        for i, section in enumerate(document_structure['sections']):
            # Only generate summaries for level 2 headers (major sections)
            if section['level'] == 2:
                document_structure['sections'][i]['summary'] = self._extract_simple_summary(section['content'])
        
        logger.info('Section summaries generated using simple extraction')
        return document_structure
    
    def _extract_simple_summary(self, content: str) -> str:
        """
        Extract a simple summary from section content.
        
        Args:
            content: Section content text
            
        Returns:
            Extracted summary (first couple of sentences)
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        # Use first two sentences if available
        if sentences and len(sentences) >= 2:
            summary = ' '.join(sentences[:2])
            return summary
        elif sentences:
            return sentences[0]
        
        # Default summary if extraction fails
        return 'This section contains relevant information about the topic.'
    
    def _generate_summaries_with_llm(self, document_structure: Dict[str, Any],
                                   llm_client: LLMClient) -> Dict[str, Any]:
        """
        Generate high-quality section summaries using LLM.
        
        Args:
            document_structure: Parsed document structure
            llm_client: LLM client for generating summaries
            
        Returns:
            Updated document structure with LLM-generated summaries
        """
        logger.info('Generating section summaries using LLM')
        
        for i, section in enumerate(document_structure['sections']):
            # Only generate summaries for level 2 headers (major sections)
            if section['level'] == 2:
                header_text = section['header'].replace('#', '').strip()
                
                # Create prompt for LLM
                prompt = f"""
                Please provide a concise 2-3 sentence summary of the following section from a fantasy/RPG document.
                The summary should capture the key information and essence of the section.
                
                Section Title: {header_text}
                
                Section Content:
                {section['content'][:1000]}  # Using first 1000 chars for token efficiency
                
                Your summary (2-3 sentences only):
                """
                
                # Get summary from LLM
                summary = llm_client.generate_content(prompt)
                
                # Clean up and validate summary
                summary = summary.strip()
                
                # Truncate if too long
                if len(summary) > 500:
                    sentences = re.split(r'(?<=[.!?])\s+', summary)
                    summary = ' '.join(sentences[:3])
                
                # Add summary to section
                document_structure['sections'][i]['summary'] = summary
        
        logger.info('LLM-enhanced section summaries generated')
        return document_structure
