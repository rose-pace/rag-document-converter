"""
Assembler package for RAG Document Converter.

This package contains components responsible for assembling the final
optimized document from processed components.
"""

from rag_converter.assembler.document_assembler import DocumentAssembler
from rag_converter.assembler.footer_generator import FooterGenerator
from rag_converter.assembler.notes_generator import NotesGenerator

__all__ = ['DocumentAssembler', 'FooterGenerator', 'NotesGenerator']
