"""
LLM summarizer module for generating high-quality section summaries.

This module provides functions for generating concise, informative summaries
of document sections using LLMs.
"""

import logging
from typing import Dict, List, Any, Optional

from rag_converter.llm.client import LLMClient

logger = logging.getLogger(__name__)

def generate_summaries_with_llm(sections: List[Dict[str, Any]], 
                              llm_client: LLMClient) -> List[Dict[str, Any]]:
    """
    Use LLM to generate high-quality summaries for document sections.
    
    Args:
        sections: List of document sections
        llm_client: Initialized LLM client
        
    Returns:
        Updated list of sections with summaries added
    """
    logger.info('Generating section summaries with LLM')
    
    # Create a copy to avoid modifying the original
    updated_sections = sections.copy()
    
    for i, section in enumerate(updated_sections):
        # Only generate summaries for level 2 headers (major sections)
        if section.get('level') == 2:
            logger.info(f'Generating summary for section: {section.get("header", "Unnamed section")}')
            
            # Create summary prompt
            prompt = _create_summary_prompt(section)
            
            # Create system message
            system_message = (
                'You are an expert in summarizing fantasy RPG lore. '
                'Create concise, informative summaries that capture key information.'
            )
            
            try:
                # Generate response
                response = llm_client.generate(
                    prompt=prompt,
                    system_message=system_message,
                    temperature=0.3,  # Slightly low temperature for more factual summaries
                    max_tokens=150  # Limit summary length
                )
                
                # Process and add summary
                summary = _process_summary_response(response.content)
                updated_sections[i]['summary'] = summary
                
            except Exception as e:
                logger.error(f'Error generating summary for section {i}: {str(e)}')
                # Create fallback summary
                updated_sections[i]['summary'] = _create_fallback_summary(section)
        
    return updated_sections


def _create_summary_prompt(section: Dict[str, Any]) -> str:
    """
    Create a prompt for summary generation.
    
    Args:
        section: Section data dictionary
        
    Returns:
        Formatted prompt string
    """
    # Extract section header without markdown symbols
    header = section.get('header', 'Unnamed section')
    clean_header = header.lstrip('#').strip()
    
    # Limit content length to avoid token limits
    content = section.get('content', '')
    content_sample = content[:2500] if len(content) > 2500 else content
    
    prompt = f"""
Please provide a concise 2-3 sentence summary of the following section from a fantasy RPG setting document.
The summary should capture the key facts and essence of the section.

Section Title: {clean_header}

Section Content:
```
{content_sample}
```

Create a summary that:
1. Captures the most important information
2. Uses clear, direct language
3. Is appropriate for a reference document
4. Is between 2-3 sentences long
5. Avoids introductory phrases like "This section describes..." or "In this section..."

Summary:
"""
    
    return prompt


def _process_summary_response(response_text: str) -> str:
    """
    Process and clean the summary response from the LLM.
    
    Args:
        response_text: Raw response from the LLM
        
    Returns:
        Cleaned and formatted summary text
    """
    # Remove any "Summary:" prefix the LLM might have added
    summary = response_text.strip()
    
    if summary.lower().startswith('summary:'):
        summary = summary[8:].strip()
    
    # Remove quotes if the LLM wrapped the summary in them
    if (summary.startswith('"') and summary.endswith('"')) or \
       (summary.startswith("'") and summary.endswith("'")):
        summary = summary[1:-1].strip()
    
    # If summary is too long (more than 5 sentences), truncate
    sentences = summary.split('.')
    if len(sentences) > 5:
        summary = '.'.join(sentences[:3]) + '.'
    
    return summary


def _create_fallback_summary(section: Dict[str, Any]) -> str:
    """
    Create a fallback summary when LLM generation fails.
    
    Args:
        section: Section data dictionary
        
    Returns:
        Simple fallback summary
    """
    # Extract section header without markdown symbols
    header = section.get('header', 'Unnamed section')
    clean_header = header.lstrip('#').strip()
    
    # Create a simple summary based on the first sentence or two
    content = section.get('content', '')
    sentences = content.split('.')
    
    if len(sentences) >= 2:
        # Use first two sentences
        fallback = f"{sentences[0].strip()}.{sentences[1].strip()}."
        # Limit length
        if len(fallback) > 240:
            return fallback[:240] + '...'
        return fallback
    
    # If we can't extract sentences, use generic summary
    return f"This section contains information about {clean_header}."
