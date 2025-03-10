"""
Entity enhancer module for LLM-based entity detection.

This module provides functions to enhance entity detection using LLMs,
including prompt construction and response parsing.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from rag_converter.llm.client import LLMClient

logger = logging.getLogger(__name__)

def enhance_entities_with_llm(content: str, entities: List[Dict[str, Any]], 
                             llm_client: LLMClient) -> List[Dict[str, Any]]:
    """
    Use LLM to enhance entity detection in the document.
    
    Args:
        content: Document content to analyze
        entities: List of entities already detected by traditional methods
        llm_client: Initialized LLM client
        
    Returns:
        Enhanced list of entities
    """
    logger.info('Enhancing entity detection with LLM')
    
    # Create entity extraction prompt
    prompt = _create_entity_extraction_prompt(content, entities)
    
    # Create system message
    system_message = (
        'You are an expert in entity extraction from fantasy RPG documents. '
        'Identify named entities such as deities, locations, events, items, factions, '
        'creatures, and concepts. Return only the JSON output with no additional text.'
    )
    
    # Generate response
    try:
        response = llm_client.generate(
            prompt=prompt,
            system_message=system_message,
            temperature=0.2  # Lower temperature for more consistent extraction
        )
        
        # Parse LLM response
        enhanced_entities = _parse_entity_extraction_response(response.content, entities)
        
        logger.info(f'LLM identified {len(enhanced_entities) - len(entities)} additional entities')
        return enhanced_entities
        
    except Exception as e:
        logger.error(f'Error enhancing entities with LLM: {str(e)}')
        # Return original entities if there was an error
        return entities


def _create_entity_extraction_prompt(content: str, existing_entities: List[Dict[str, Any]]) -> str:
    """
    Create a prompt for entity extraction.
    
    Args:
        content: Document content to analyze
        existing_entities: List of entities already detected
        
    Returns:
        Formatted prompt string
    """
    # Limit content length to avoid token limits
    content_sample = content[:3000] if len(content) > 3000 else content
    
    # Format existing entities for context
    existing_entities_str = '\n'.join([
        f"- {entity.get('name', 'Unknown')} (Type: {entity.get('type', 'unknown')})"
        for entity in existing_entities[:5]  # Limit to first 5 for brevity
    ])
    
    if not existing_entities_str:
        existing_entities_str = "None identified yet"
    
    prompt = f"""
Please identify all named entities in the following fantasy RPG document text.
Focus on these entity types: deities, locations, events, items, factions, creatures, and concepts.

For each entity, provide:
1. A name (the primary identifier)
2. An entity type from the list above
3. A brief description (1-2 sentences)
4. Any aliases or alternative names
5. Associated attributes or domains (if applicable)

Text to analyze:
```
{content_sample}
```

Some entities already identified:
{existing_entities_str}

Return your analysis as a JSON array of entity objects with the following structure:
```json
[
  {{
    "name": "Entity Name",
    "type": "entity_type",
    "description": "Brief description",
    "aliases": ["alias1", "alias2"],
    "attributes": ["attribute1", "attribute2"]
  }},
  ...
]
```

Only include entities you are confident about. Focus on quality over quantity.
"""
    
    return prompt


def _parse_entity_extraction_response(response_text: str, existing_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse LLM response for entity extraction.
    
    Args:
        response_text: Text response from the LLM
        existing_entities: List of entities already detected
        
    Returns:
        Enhanced list of entities
    """
    # Create a copy of existing entities
    all_entities = existing_entities.copy()
    
    # Track existing entity names for deduplication
    existing_names = {entity.get('name', '').lower() for entity in existing_entities}
    
    try:
        # Extract JSON content from response
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_content = response_text[json_start:json_end]
            entities_data = json.loads(json_content)
            
            # Process and add new entities
            for entity_data in entities_data:
                entity_name = entity_data.get('name', '').strip()
                
                # Skip if entity name is empty or already exists
                if not entity_name or entity_name.lower() in existing_names:
                    continue
                    
                # Create standardized entity object
                new_entity = {
                    'name': entity_name,
                    'type': entity_data.get('type', 'concept').lower(),
                    'description': entity_data.get('description', ''),
                    'position': -1  # Mark as LLM-detected
                }
                
                # Add optional fields if present
                if 'aliases' in entity_data and entity_data['aliases']:
                    new_entity['aliases'] = entity_data['aliases']
                    
                if 'attributes' in entity_data and entity_data['attributes']:
                    new_entity['attributes'] = entity_data['attributes']
                
                # Add to entities list and track the name
                all_entities.append(new_entity)
                existing_names.add(entity_name.lower())
        
        else:
            logger.warning('No JSON array found in LLM response')
            
    except json.JSONDecodeError as e:
        logger.error(f'Error parsing entity extraction response: {str(e)}')
        logger.debug(f'Failed JSON content: {response_text}')
    except Exception as e:
        logger.error(f'Unexpected error processing entity extraction: {str(e)}')
    
    return all_entities
