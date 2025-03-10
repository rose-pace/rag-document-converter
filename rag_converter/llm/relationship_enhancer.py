"""
Relationship enhancer module for LLM-based relationship extraction.

This module provides functions to extract semantic relationships between
entities using LLMs, including prompt construction and response parsing.
"""

import json
import logging
import yaml
import re
from typing import Dict, List, Any, Optional

from rag_converter.llm.client import LLMClient

logger = logging.getLogger(__name__)

def structure_relationships_with_llm(document_structure: Dict[str, Any], 
                                    entity_identifiers: Dict[str, str],
                                    llm_client: LLMClient) -> Dict[str, Any]:
    """
    Use LLM to extract and structure relationships between entities.
    
    Args:
        document_structure: Document structure with sections
        entity_identifiers: Dictionary mapping entity names to identifiers
        llm_client: Initialized LLM client
        
    Returns:
        Updated document structure with relationships added
    """
    logger.info('Extracting entity relationships with LLM')
    
    # Create a copy of the document structure to avoid modifying the original
    updated_structure = document_structure.copy()
    updated_structure['sections'] = document_structure['sections'].copy()
    
    # Process each section
    for i, section in enumerate(updated_structure['sections']):
        # Skip short sections or headers
        if len(section['content']) < 100:
            continue
            
        # Create relationship extraction prompt
        prompt = _create_relationship_extraction_prompt(
            section['content'], 
            entity_identifiers
        )
        
        # Create system message
        system_message = (
            'You are an expert in extracting semantic relationships between entities in fantasy RPG documents. '
            'Focus on identifying clear relationships and return them in the requested YAML format with no additional text.'
        )
        
        try:
            # Generate response
            response = llm_client.generate(
                prompt=prompt,
                system_message=system_message,
                temperature=0.2  # Lower temperature for more consistent extraction
            )
            
            # Process relationships YAML
            relationships_yaml = _extract_yaml_from_response(response.content)
            
            if relationships_yaml:
                # Only add relationships block if relationships were found
                if 'Relationships:' in relationships_yaml and 'Type:' in relationships_yaml:
                    # Add relationships to section content
                    updated_structure['sections'][i]['content'] += f'\n\n{relationships_yaml}\n'
                    logger.info(f'Added relationships to section: {section.get("header", "Unnamed section")}')
                
        except Exception as e:
            logger.error(f'Error extracting relationships for section {i}: {str(e)}')
    
    return updated_structure


def _create_relationship_extraction_prompt(content: str, entity_identifiers: Dict[str, str]) -> str:
    """
    Create a prompt for relationship extraction.
    
    Args:
        content: Section content to analyze
        entity_identifiers: Dictionary mapping entity names to identifiers
        
    Returns:
        Formatted prompt string
    """
    # Limit content length to avoid token limits
    content_sample = content[:2500] if len(content) > 2500 else content
    
    # Format entity list for context - limit to 10 entities
    entity_list = list(entity_identifiers.items())[:10]
    entity_context = '\n'.join([
        f'- {name} [{identifier}]'
        for name, identifier in entity_list
    ])
    
    if not entity_context:
        entity_context = "No entities identified yet"
    
    prompt = f"""
From the following text, identify meaningful relationships between entities.
Focus on clear relationships like:

1. Hierarchical (commands, reports to, serves)
2. Family (parent, sibling, descendant)
3. Creation (created by, inventor of)
4. Conflict (enemy of, rival to)
5. Alliance (ally, friend, partner)
6. Spatial (contains, located in)
7. Temporal (precedes, follows)
8. Ownership (possesses, owns)
9. Attribution (has quality, has power)

Text to analyze:
```
{content_sample}
```

Known entities and their identifiers:
{entity_context}

Return ONLY a YAML block with the relationships you can confidently extract.
Use this exact format:

```yaml
Relationships:
  - Type: relationship_type
    Source: entity_name [IDENTIFIER]
    Target: entity_name [IDENTIFIER]
    Description: "Clear description of the relationship"
```

If you can't identify any clear relationships, return an empty YAML block.
Include entity identifiers when available, but you can also define relationships
for entities not in the provided list if you're confident they exist in the text.
"""
    
    return prompt


def _extract_yaml_from_response(response_text: str) -> str:
    """
    Extract YAML content from LLM response.
    
    Args:
        response_text: Text response from the LLM
        
    Returns:
        Extracted and validated YAML content as a string, or empty string if invalid
    """
    # Extract content between YAML code fence markers
    yaml_pattern = r'```(?:yaml)?\s*(.*?)```'
    match = re.search(yaml_pattern, response_text, re.DOTALL)
    
    if not match:
        # Try finding just the YAML content without code fences
        yaml_pattern = r'(?:^|\n)Relationships:\s*\n(?:\s+-.+\n)+'
        match = re.search(yaml_pattern, response_text, re.DOTALL)
        if match:
            yaml_content = match.group(0)
        else:
            logger.warning('No YAML content found in LLM response')
            return ''
    else:
        yaml_content = match.group(1)
    
    # Validate YAML
    try:
        # Parse YAML to check validity
        parsed_yaml = yaml.safe_load(yaml_content)
        
        # Check if it contains relationships
        if not parsed_yaml or 'Relationships' not in parsed_yaml:
            logger.warning('Invalid relationships YAML structure')
            return ''
            
        # If valid, return formatted YAML block
        return f"```yaml\n{yaml_content.strip()}\n```"
        
    except yaml.YAMLError as e:
        logger.error(f'Invalid YAML in relationship extraction: {str(e)}')
        return ''
