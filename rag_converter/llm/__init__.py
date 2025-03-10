"""
LLM integration package for RAG Document Converter.

This package provides optional LLM-enhanced processing capabilities
using either Ollama (for local processing) or Anthropic (for cloud-based processing).
"""

__version__ = '0.1.0'

from rag_converter.llm.client import LLMClient, create_llm_client
from rag_converter.llm.entity_enhancer import enhance_entities_with_llm
from rag_converter.llm.relationship_enhancer import structure_relationships_with_llm
from rag_converter.llm.summarizer import generate_summaries_with_llm

__all__ = [
    'LLMClient',
    'create_llm_client',
    'enhance_entities_with_llm',
    'structure_relationships_with_llm',
    'generate_summaries_with_llm',
]
