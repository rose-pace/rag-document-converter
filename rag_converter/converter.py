"""
Core converter module for RAG Document Converter.

This module contains the DocumentConverter class that orchestrates the entire
conversion process by integrating parser, optimizer, and assembler components.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union

from rag_converter.parser.document_parser import DocumentParser
from rag_converter.optimizer.document_optimizer import DocumentOptimizer
from rag_converter.assembler.document_assembler import DocumentAssembler
from rag_converter.llm import LLMClient, create_llm_client

logger = logging.getLogger(__name__)

class DocumentConverter:
    """
    Main class that orchestrates the document conversion process.
    
    This class integrates all components needed for converting markdown documents
    into RAG-optimized format, including optional LLM integration.
    """
    
    def __init__(self, use_llm: bool = False, config_path: Optional[str] = None):
        """
        Initialize the document converter.
        
        Args:
            use_llm: Whether to use LLM for enhanced processing
            config_path: Optional path to a custom configuration file
        """
        self.use_llm = use_llm
        self.config_path = config_path
        
        # Initialize LLM client if enabled
        self.llm_client: LLMClient = None
        if use_llm:
            self.llm_client = create_llm_client()
            logger.info('LLM integration enabled')
        else:
            logger.info('Running in traditional code mode only')
        
        # Initialize components
        self.parser = DocumentParser()
        self.optimizer = DocumentOptimizer(llm_client=self.llm_client)
        self.assembler = DocumentAssembler()
    
    def convert_document(self, input_path: Union[str, Path], output_path: Union[str, Path]) -> bool:
        """
        Convert a markdown document to RAG-optimized format.
        
        Args:
            input_path: Path to the input markdown document
            output_path: Path where the optimized document will be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert paths to Path objects if they're strings
            input_path = Path(input_path) if isinstance(input_path, str) else input_path
            output_path = Path(output_path) if isinstance(output_path, str) else output_path
            
            # Ensure input file exists
            if not input_path.exists():
                logger.error(f'Input file not found: {input_path}')
                return False
                
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load the document
            logger.info(f'Processing document: {input_path}')
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Process the document through the pipeline
            # 1. Parse document structure
            document_structure = self.parser.parse_document(content)
            
            # 2. Optimize document 
            """
            TODO: This should be done in multiple passes using the LLM
                1. Generate standardized vocabulary
                    - Use LLM for vocabulary generation
                    - Vocabulary should be updated and stored during each pass
                    - May run multiple times to refine vocabulary before continuing with other steps
                2. Apply vocabulary to content
                    - This can be implemented with standard code or a simpler model
                3. Extract Entities
                    - Use LLM for entity extraction
                    - Use entity config for generating entity code
                    - Report on non-recognized entities
                4. Extract Relationships
                    - Use LLM for relationship extraction
                    - Use entity config for generating relationship code
                    - Report on non-recognized relationships
                5. Summarize Sections
            """
            optimized_document = self.optimizer.optimize_document(document_structure)
            
            # 3. Assemble final document
            final_document = self.assembler.assemble_document(optimized_document)
            
            # Save the processed document
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(final_document)
                
            logger.info(f'Optimized document saved to: {output_path}')
            return True
            
        except Exception as e:
            logger.error(f'Error converting document: {str(e)}', exc_info=True)
            return False
    
    def batch_convert(self, input_dir: Union[str, Path], output_dir: Union[str, Path], 
                     file_pattern: str = '*.md') -> Dict[str, bool]:
        """
        Convert multiple markdown documents in a directory.
        
        Args:
            input_dir: Path to directory containing input documents
            output_dir: Path where optimized documents will be saved
            file_pattern: Glob pattern to match markdown files
            
        Returns:
            Dictionary mapping filenames to conversion success status
        """
        # Convert paths to Path objects if they're strings
        input_dir = Path(input_dir) if isinstance(input_dir, str) else input_dir
        output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Process each file matching the pattern
        for input_file in input_dir.glob(file_pattern):
            relative_path = input_file.relative_to(input_dir)
            output_file = output_dir / relative_path
            
            # Create parent directories if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert the document
            success = self.convert_document(input_file, output_file)
            results[str(relative_path)] = success
        
        # Log summary
        success_count = sum(1 for success in results.values() if success)
        logger.info(f'Batch conversion completed: {success_count}/{len(results)} files successful')
        
        return results
