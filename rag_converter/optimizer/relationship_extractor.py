"""
Relationship extractor module for RAG Document Converter.

This module handles the extraction and structuring of relationships 
between entities in the document.
"""

import re
import logging
import yaml
from typing import Dict, List, Any, Optional, Tuple

from rag_converter.llm.client import LLMClient

logger = logging.getLogger(__name__)

class RelationshipExtractor:
    """
    Class for extracting and structuring relationships between entities.
    
    This class identifies relationships in text and formats them as structured YAML.
    """
    
    def __init__(self, use_llm: bool = False):
        """
        Initialize the relationship extractor.
        
        Args:
            use_llm: Whether to use LLM for enhanced relationship extraction
        """
        self.use_llm = use_llm
        
        # Define relationship patterns
        self.relationship_patterns = [
            (r'(\w+) is the (father|mother|parent|child|sibling) of (\w+)', 'family'),
            (r'(\w+) (rules|governs|controls) (\w+)', 'authority'),
            (r'(\w+) (created|made|formed) (\w+)', 'creation'),
            (r'(\w+) (worships|venerates|serves) (\w+)', 'religious'),
            (r'(\w+) is (located|situated|found) in (\w+)', 'location'),
        ]
        
        logger.info('Relationship extractor initialized')
    
    def structure_relationships(self, document_structure: Dict[str, Any], 
                              entity_identifiers: Dict[str, str],
                              llm_client: Optional[LLMClient] = None) -> Dict[str, Any]:
        """
        Extract and structure relationships between entities.
        
        Args:
            document_structure: Parsed document structure
            entity_identifiers: Dict mapping entity names to identifiers
            llm_client: Optional LLM client for enhanced extraction
            
        Returns:
            Updated document structure with structured relationships
        """
        logger.info('Extracting and structuring entity relationships')
        
        # If LLM is enabled and client is provided, use it for enhanced extraction
        if self.use_llm and llm_client:
            return self._structure_relationships_with_llm(document_structure, entity_identifiers, llm_client)
        
        # Otherwise use pattern-based extraction
        extracted_relationships = []
        
        # Process each section
        for i, section in enumerate(document_structure['sections']):
            content = section['content']
            
            # Look for relationship patterns
            section_relationships = self._extract_pattern_relationships(content, entity_identifiers)
            
            # Add relationships to the section content if any found
            if section_relationships:
                # Format relationships as YAML
                yaml_block = self._format_relationships_as_yaml(section_relationships)
                
                # Add to section content if not already present
                if 'Relationships:' not in content:
                    document_structure['sections'][i]['content'] += f'\n\n{yaml_block}\n'
            
            extracted_relationships.extend(section_relationships)
        
        # Store extracted relationships in document structure
        document_structure['relationships'] = extracted_relationships
        
        logger.info(f'Extracted {len(extracted_relationships)} relationships')
        return document_structure
    
    def _extract_pattern_relationships(self, content: str, 
                                     entity_identifiers: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Extract relationships using regex patterns.
        
        Args:
            content: Text content to analyze
            entity_identifiers: Dict mapping entity names to identifiers
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Get entity names
        entity_names = list(entity_identifiers.keys())
        
        # Look for relationship patterns
        for pattern, rel_type in self.relationship_patterns:
            matches = re.finditer(pattern, content)
            
            for match in matches:
                source = match.group(1)
                relation = match.group(2)
                target = match.group(3)
                
                # Check if source and target are known entities
                if source in entity_names and target in entity_names:
                    source_id = entity_identifiers.get(source, '')
                    target_id = entity_identifiers.get(target, '')
                    
                    relationships.append({
                        'type': rel_type,
                        'source': source,
                        'source_id': source_id,
                        'target': target,
                        'target_id': target_id,
                        'description': f'{source} {relation} {target}'
                    })
        
        return relationships
    
    def _format_relationships_as_yaml(self, relationships: List[Dict[str, str]]) -> str:
        """
        Format relationships as a YAML block.
        
        Args:
            relationships: List of relationship dictionaries
            
        Returns:
            Formatted YAML block as a string
        """
        if not relationships:
            return ''
            
        yaml_dict = {'Relationships': []}
        
        for rel in relationships:
            yaml_dict['Relationships'].append({
                'Type': rel['type'],
                'Source': f"{rel['source']} [{rel['source_id']}]" if rel['source_id'] else rel['source'],
                'Target': f"{rel['target']} [{rel['target_id']}]" if rel['target_id'] else rel['target'],
                'Description': rel['description']
            })
        
        return f"```yaml\n{yaml.safe_dump(yaml_dict, sort_keys=False, default_flow_style=False)}```"
    
    def _structure_relationships_with_llm(self, document_structure: Dict[str, Any],
                                        entity_identifiers: Dict[str, str],
                                        llm_client: LLMClient) -> Dict[str, Any]:
        """
        Extract relationships using LLM.
        
        Args:
            document_structure: Parsed document structure
            entity_identifiers: Dict mapping entity names to identifiers
            llm_client: LLM client for generating and processing prompts
            
        Returns:
            Updated document structure with LLM-extracted relationships
        """
        logger.info('Using LLM to extract entity relationships')
        
        # First, use pattern matching to get baseline relationships
        pattern_relationships = []
        for section in document_structure['sections']:
            pattern_relationships.extend(
                self._extract_pattern_relationships(section['content'], entity_identifiers)
            )
        
        # Prepare entity list for LLM context
        entity_list = '\n'.join([
            f"- {name} [{entity_id}]" 
            for name, entity_id in entity_identifiers.items()
        ])
        
        # Process each major section with LLM
        for i, section in enumerate(document_structure['sections']):
            # Only process sections with substantial content
            if len(section['content']) < 50 or 'Relationships:' in section['content']:
                continue
                
            # Create prompt for LLM
            prompt = f"""
            From the following text, identify all relationships between entities.
            For each relationship, provide:
            - Relationship type (e.g., family, authority, creation, alliance, location, etc.)
            - Source entity
            - Target entity
            - Brief description
            
            Known entities:
            {entity_list}
            
            Text:
            {section['content'][:1500]}  # Using first 1500 chars for token efficiency
            
            Return ONLY a YAML block in this format (nothing else):
            ```yaml
            Relationships:
              - Type: [type]
                Source: [source entity name] [[source_id]]
                Target: [target entity name] [[target_id]]
                Description: "[description]"
            ```
            If no relationships are found, return "No relationships identified."
            """
            
            # Process with LLM client
            llm_result = llm_client.generate_content(prompt)
            
            # Extract YAML from result
            yaml_match = re.search(r'```yaml\s+(.*?)\s+```', llm_result, re.DOTALL)
            yaml_block = ''
            
            if yaml_match:
                yaml_content = yaml_match.group(1)
                # Validate YAML structure
                try:
                    yaml_data = yaml.safe_load(yaml_content)
                    if yaml_data and 'Relationships' in yaml_data:
                        yaml_block = f"```yaml\n{yaml_content}```"
                except yaml.YAMLError:
                    logger.warning(f'Invalid YAML returned by LLM for section {i}')
                    continue
            
            # Add YAML block to section if valid
            if yaml_block and 'No relationships identified' not in llm_result:
                document_structure['sections'][i]['content'] += f'\n\n{yaml_block}\n'
        
        # Combine pattern and LLM relationships into document structure
        document_structure['relationships'] = pattern_relationships
        
        return document_structure
