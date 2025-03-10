"""
Footer generator module for RAG Document Converter.

This module contains the FooterGenerator class that creates document footers
with appendices and cross-references.
"""

import logging
from typing import Dict, Any, List, Optional, Set

from rag_converter.config import load_config

logger = logging.getLogger(__name__)

class FooterGenerator:
    """
    Creates document footer with appendices and cross-references.
    
    This class handles the generation of the document footer section,
    including appendices and structured cross-references.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the footer generator.
        
        Args:
            config_path: Optional path to a custom configuration file
        """
        self.config = load_config(config_path)
        logger.info('Footer generator initialized')
    
    def create_footer(self, document_structure: Dict[str, Any]) -> str:
        """
        Create the document footer.
        
        Args:
            document_structure: Document structure with sections and entities
            
        Returns:
            Formatted footer section
        """
        logger.info('Creating document footer')
        
        # Extract existing appendices if present
        appendices = self._extract_appendices(document_structure)
        
        # Create cross-references section
        cross_references = self.create_cross_references(document_structure)
        
        # Assemble complete footer
        footer = '## Appendices\n\n'
        if appendices:
            footer += f'{appendices}\n\n'
        else:
            footer += 'No additional appendices.\n\n'
        
        footer += '## Related Documents\n\n'
        footer += f'{cross_references}\n'
        
        logger.debug('Footer creation completed')
        return footer
    
    def _extract_appendices(self, document_structure: Dict[str, Any]) -> str:
        """
        Extract existing appendices from document.
        
        Args:
            document_structure: Document structure with sections
            
        Returns:
            Extracted appendices content or empty string
        """
        appendix_content = ''
        
        # Look for sections with "appendix" in the header
        for section in document_structure.get('sections', []):
            header = section.get('header', '').lower()
            if 'appendix' in header and not header.startswith('## appendix'):
                # Extract content but don't include the header itself
                appendix_content += f"{section.get('content', '')}\n\n"
        
        return appendix_content.strip()
    
    def create_cross_references(self, document_structure: Dict[str, Any]) -> str:
        """
        Create cross-references section.
        
        Args:
            document_structure: Document structure with entities and content
            
        Returns:
            Formatted cross-references section
        """
        title = document_structure.get('title', '')
        collection = document_structure.get('doc_notes', {}).get('Collection', 'Uncategorized')
        entities = document_structure.get('entities', [])
        
        # Identify related documents based on entities and content
        direct_refs = self._identify_direct_references(document_structure)
        related_collections = self._identify_related_collections(collection, title, entities)
        parent_docs, child_docs = self._identify_document_relationships(title, collection, entities)
        
        # Format as YAML block
        yaml_block = '```yaml\nCross References:\n'
        
        # Add direct references
        yaml_block += '  Direct References:\n'
        if direct_refs:
            for ref in sorted(direct_refs):
                yaml_block += f'    - {ref}\n'
        else:
            yaml_block += '    [] # No direct references found\n'
        
        # Add collection membership
        yaml_block += '\n  Collection Membership:\n'
        yaml_block += f'    Primary Collection: {collection}\n'
        yaml_block += '    Related Collections:\n'
        if related_collections:
            for coll in sorted(related_collections):
                yaml_block += f'      - {coll}\n'
        else:
            yaml_block += '      [] # No related collections\n'
        
        # Add document relationships
        yaml_block += '\n  Document Relationships:\n'
        yaml_block += '    Parent Documents:\n'
        if parent_docs:
            for doc in sorted(parent_docs):
                yaml_block += f'      - {doc}\n'
        else:
            yaml_block += '      [] # No parent documents\n'
            
        yaml_block += '    Child Documents:\n'
        if child_docs:
            for doc in sorted(child_docs):
                yaml_block += f'      - {doc}\n'
        else:
            yaml_block += '      [] # No child documents\n'
            
        yaml_block += '```'
        
        return yaml_block
    
    def _identify_direct_references(self, document_structure: Dict[str, Any]) -> Set[str]:
        """
        Identify directly referenced documents.
        
        Args:
            document_structure: Document structure with content
            
        Returns:
            Set of directly referenced documents
        """
        import re
        references = set()
        
        # Look for markdown-style links
        content = document_structure.get('raw_content', '')
        link_pattern = r'\[([^\]]+)\]\(([^)]+\.md)\)'
        
        for match in re.finditer(link_pattern, content):
            link_text = match.group(1)
            link_target = match.group(2)
            references.add(f'"{link_text}" ({link_target})')
        
        return references
    
    def _identify_related_collections(self, primary_collection: str, title: str, entities: List[Dict[str, Any]]) -> Set[str]:
        """
        Identify related collections based on content.
        
        Args:
            primary_collection: Primary collection name
            title: Document title
            entities: List of entities in the document
            
        Returns:
            Set of related collection names
        """
        collections = set()
        
        # Map entity types to collections
        type_to_collection = {
            'deity': 'Divine Entities',
            'location': 'Geography',
            'event': 'History',
            'item': 'Artifacts',
            'faction': 'Organizations',
            'creature': 'Bestiary',
            'concept': 'Concepts'
        }
        
        # Add collections based on entity types
        entity_types = {entity.get('type', 'concept') for entity in entities}
        for entity_type in entity_types:
            if entity_type in type_to_collection:
                collections.add(type_to_collection[entity_type])
        
        # Add collections based on document title
        title_lower = title.lower()
        if any(term in title_lower for term in ['deity', 'god', 'goddess', 'divine']):
            collections.add('Divine Entities')
        if any(term in title_lower for term in ['history', 'chronicle', 'era']):
            collections.add('History')
            
        # Remove the primary collection from related collections
        if primary_collection in collections:
            collections.remove(primary_collection)
            
        return collections
    
    def _identify_document_relationships(self, title: str, collection: str, entities: List[Dict[str, Any]]) -> tuple:
        """
        Identify parent and child document relationships.
        
        Args:
            title: Document title
            collection: Document collection
            entities: List of entities in the document
            
        Returns:
            Tuple of (parent_docs, child_docs) sets
        """
        parent_docs = set()
        child_docs = set()
        
        # Determine potential parent documents based on collection
        if collection == 'Divine Entities':
            parent_docs.add('"Pantheons Overview" (pantheons-overview.md)')
        elif collection == 'Geography':
            parent_docs.add('"World Atlas" (world-atlas.md)')
        elif collection == 'History':
            parent_docs.add('"Timeline" (timeline.md)')
        
        # Determine potential child documents based on entities
        entity_names = {}
        for entity in entities:
            entity_type = entity.get('type', '')
            name = entity.get('name', '')
            if name:
                entity_names[name] = entity_type
        
        # For deities, add potential children
        if 'deity' in entity_names.values():
            deity_names = [name for name, etype in entity_names.items() if etype == 'deity']
            if deity_names:
                for name in deity_names[:2]:  # Limit to first two to avoid cluttering
                    filename = name.lower().replace(' ', '-')
                    child_docs.add(f'"{name} Worship" ({filename}-worship.md)')
                    
        # For locations, add potential children
        if 'location' in entity_names.values():
            location_names = [name for name, etype in entity_names.items() if etype == 'location']
            if location_names:
                for name in location_names[:2]:  # Limit to first two
                    filename = name.lower().replace(' ', '-')
                    child_docs.add(f'"{name} Points of Interest" ({filename}-points-of-interest.md)')
        
        return parent_docs, child_docs
