"""
Main document optimizer module for RAG Document Converter.

This module contains the DocumentOptimizer class that orchestrates all
optimization processes to enhance document structure for RAG.
"""

import logging
from typing import Dict, Any, Optional, List

from rag_converter.optimizer.entity_identifier import EntityIdentifier
from rag_converter.optimizer.relationship_extractor import RelationshipExtractor
from rag_converter.optimizer.section_summarizer import SectionSummarizer
from rag_converter.optimizer.vocabulary_controller import VocabularyController
from rag_converter.llm.client import LLMClient
from rag_converter.config import load_config

logger = logging.getLogger(__name__)

class DocumentOptimizer:
    """
    Main class that orchestrates document optimization processes.
    
    This class integrates all optimization components to enhance the document
    structure for improved RAG performance.
    """
    
    def __init__(self, llm_client: LLMClient = None, config_path: Optional[str] = None):
        """
        Initialize the document optimizer.
        
        Args:
            use_llm: Whether to use LLM for enhanced processing
            config_path: Optional path to a custom configuration file
        """
        self.config = load_config(config_path)
        self.use_llm = llm_client is not None
        self.llm_client = LLMClient()
        
        # Initialize optimization components
        self.entity_identifier = EntityIdentifier(self.config)
        self.relationship_extractor = RelationshipExtractor(use_llm=self.use_llm)
        self.section_summarizer = SectionSummarizer(use_llm=self.use_llm)
        self.vocabulary_controller = VocabularyController(self.config)
                
        logger.info('Document optimizer initialized')
    
    def optimize_document(self, document_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize document structure for RAG processing.
        
        Args:
            document_structure: Parsed document structure from DocumentParser
            
        Returns:
            Optimized document structure
        """
        logger.info('Starting document optimization process')
        # TODO: All of this should use the LLM in multple passes
        # 1. Generate standardized identifiers for entities
        entity_identifiers = self.entity_identifier.generate_entity_identifiers(
            document_structure['entities']
        )
        
        # 2. Apply entity identifiers to content
        document_structure = self.entity_identifier.apply_entity_identifiers(
            document_structure, 
            entity_identifiers
        )
        
        # 3. Generate section summaries
        document_structure = self.section_summarizer.generate_section_summaries(
            document_structure,
            self.llm_client if self.use_llm else None
        )
        
        # 4. Extract and structure relationships
        document_structure = self.relationship_extractor.structure_relationships(
            document_structure,
            entity_identifiers,
            self.llm_client if self.use_llm else None
        )
        
        # 5. Apply controlled vocabulary : TODO: update so llm defines the controlled vocabulary and keeps passing it in to each successive document optimizer
        document_structure = self.vocabulary_controller.apply_controlled_vocabulary(
            document_structure
        )
        
        logger.info('Document optimization completed')
        return document_structure
