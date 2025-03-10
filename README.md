# RAG Document Converter

A tool for converting documents to a format optimized for Retrieval-Augmented Generation (RAG) systems.

## Overview

This project provides a converter that transforms standard markdown documents into a structured format optimized for RAG systems. The converter:

- Adds standardized document metadata
- Generates concise section summaries
- Identifies entities and adds standardized identifiers
- Extracts and structures relationships
- Applies controlled vocabulary
- Adds cross-references and appendices

## Features

- **Entity Identification**: Automatically detects and marks entities with standardized identifiers
- **Relationship Extraction**: Identifies and structures relationships between entities
- **Section Summarization**: Generates concise summaries for major sections
- **Controlled Vocabulary**: Enforces consistent terminology
- **Cross-References**: Adds structured cross-references to related documents
- **LLM Integration**: Optional enhancement using Large Language Models (supports Ollama, OpenAI, and Anthropic)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/rag-document-converter.git
cd rag-document-converter

# Install dependencies
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Convert a single file
python -m rag_converter.cli --input input.md --output output.md

# Convert all files in a directory
python -m rag_converter.cli --input input_dir --output output_dir

# Enable LLM enhancement using Ollama
python -m rag_converter.cli --input input.md --output output.md --use-llm --llm-provider ollama --llm-model llama3

# Use a custom configuration
python -m rag_converter.cli --input input.md --output output.md --config config.yaml
```

### Python API

```python
from rag_converter import create_converter

# Create converter with default settings
converter = create_converter()

# Convert a document
result = converter.convert_document("input.md", "output.md")

# Check conversion result
if result.success:
    print(f"Successfully converted document with {len(result.entities_found)} entities")
else:
    print(f"Conversion failed: {result.errors}")

# Create converter with custom settings
config = {
    "use_llm": True,
    "llm_config": {
        "provider": "ollama",
        "model": "llama3"
    },
    "generate_summaries": True,
    "apply_entity_identifiers": True,
    "structure_relationships": True
}
converter = create_converter(config)
```

## Using with Ollama

This project supports integration with [Ollama](https://ollama.ai/) for local LLM processing. To use Ollama:

1. Install Ollama following the instructions on their website
2. Start the Ollama server: `ollama serve`
3. Pull the models you want to use:
   ```
   ollama pull llama3
   ollama pull phi
   ollama pull mistral
   ```
4. Run the converter with LLM enhancement:
   ```
   python -m rag_converter.cli --input input.md --output output.md --use-llm --llm-provider ollama --llm-model llama3
   ```

## Multi-Model Approach

For optimal results, the converter can use different specialized models for different tasks:

- **llama3**: General-purpose model for entity recognition
- **phi**: Specialized model for summarization
- **mistral**: Specialized model for relationship extraction

Enable multi-model processing by setting `use_multi_model: true` in your configuration.

## Configuration

You can customize the converter behavior with a YAML or JSON configuration file:

```yaml
# Example configuration
use_llm: true
llm_config:
  provider: "ollama"
  model: "llama3"
  base_url: "http://localhost:11434"
  use_multi_model: true
  summarization_model: "phi"
  entity_model: "llama3"
  relationship_model: "mistral"

generate_summaries: true
apply_entity_identifiers: true
structure_relationships: true
apply_controlled_vocabulary: true

entity_prefixes:
  deity: "DEI"
  location: "LOC"
  event: "EVT"
  item: "ITM"
  faction: "FAC"
  creature: "CRE"
  concept: "CON"

controlled_vocabulary:
  deity: ["god", "divine being", "immortal"]
  plane: ["realm", "dimension", "world"]
```

## Environment Variables

The converter supports the following environment variables:

```bash
# Anthropic settings
ANTHROPIC_API_KEY=your-anthropic-api-key
ANTHROPIC_MODEL=claude-3-opus-20240229
ANTHROPIC_MAX_TOKENS=4000
ANTHROPIC_TEMPERATURE=0.7

# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT=120
OLLAMA_TEMPERATURE=0.7
OLLAMA_MAX_TOKENS=2048

# Processing settings
MAX_CHUNK_SIZE=1000
OVERLAP_SIZE=200

# Logging configuration
LOG_LEVEL=INFO

# File handling
FILE_ENCODING=utf-8

# Configuration paths
RAG_CONFIG_DIR=./config
RAG_DEFAULT_CONFIG=./config/default_config.yaml
RAG_ENTITY_PATTERNS=./config/entity_patterns.yaml

# LLM Provider selection
LLM_PROVIDER=ollama
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
