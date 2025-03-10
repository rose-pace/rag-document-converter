# RAG Document Converter Project
The purpose of this project is to convert markdown files to follow the instructions of rag-document-instructions.md so that they are optimized for ingestion into a RAG vector store. This project will use a combination of traditional coding logic and LLM inference to parse and convert the unstructured documents.

## Pseudocode outline
The file rag-document-converter.py contains pseudocode to help define some of the coding requirements necessary for implementing the project. The implementation of this pseudocode will be split across a larger project structure defined below.

## Project Structure
The file project-structure.yaml outlines a modular project structure that implements the RAG document converter with proper separation of concerns. Here's an explanation of the key design decisions:

## Core Architecture
The project follows a modular architecture with clear separation between:

- Parser Components - Handle document analysis and structure extraction
- Optimizer Components - Apply transformations to optimize content for RAG
- Assembler Components - Construct the final optimized document
- LLM Integration - Optional components for AI-enhanced processing
- Utilities - Shared helper functions and tools

## Module Breakdown
### Parser Package
Contains components that extract and analyze document structure:

- DocumentParser - Coordinates parsing of the full document
- SectionParser - Extracts sections based on headers
- YAMLParser - Handles YAML block extraction and validation
- EntityParser - Identifies potential named entities
- DocumentNotesParser - Extracts document metadata

### Optimizer Package
Contains components that transform document content:

- DocumentOptimizer - Coordinates optimization processes
- EntityIdentifier - Generates and applies entity identifiers
- RelationshipExtractor - Extracts and structures relationships
- SectionSummarizer - Generates concise summaries
- VocabularyController - Applies controlled vocabulary

### Assembler Package
Contains components that construct the final document:

- DocumentAssembler - Assembles the complete document
- FooterGenerator - Creates document footer with references
- NotesGenerator - Formats document metadata section

### LLM Package
Contains optional AI-enhanced processing components:

- LLMClient - Handles LLM API integration
- EntityEnhancer - Uses LLM for better entity detection
- RelationshipEnhancer - Extracts semantic relationships with LLM
- Summarizer - Generates high-quality summaries with LLM

## Testing and Configuration
The structure includes comprehensive test directories mirroring the package structure, along with:

- Test fixtures - Sample documents for testing
- Configuration files - Default settings and patterns
- Example documents - Before/after conversion examples

## Benefits of This Structure

- Modularity - Each component has a single responsibility
- Testability - Easy to write focused unit tests
- Flexibility - Optional LLM integration
- Extensibility - Easy to add new parsers, optimizers, or assemblers
- Maintainability - Clear organization makes code easier to maintain