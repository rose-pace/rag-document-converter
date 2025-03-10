"""
Configuration handling for RAG Document Converter.

This module contains constants, configuration classes, and functions
for loading and validating configuration settings.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml
from pydantic import BaseModel, Field

# Constants for document structure
DOCUMENT_TEMPLATE = '''# {title}

## Document Notes
```yaml
Document Version: {version}
Version Date: {date}
Collection: {collection}
Tags:
{tags}
```

{summary}

{body}

{footer}
'''

# Constants for entity identifier prefixes
ENTITY_TYPE_PREFIXES = {
    'deity': 'DEI',
    'location': 'LOC',
    'event': 'EVT',
    'item': 'ITM',
    'faction': 'FAC',
    'creature': 'CRE',
    'concept': 'CON',
}

# Controlled vocabulary mapping
CONTROLLED_VOCABULARY = {
    'deity': ['god', 'divine being', 'immortal'],
    'plane': ['realm', 'dimension', 'world'],
    'causes': ['results in', 'leads to', 'creates'],
    'contains': ['houses', 'includes', 'holds'],
    'rules': ['governs', 'controls', 'dominates'],
}

# Default file paths - allow override via environment variables
DEFAULT_CONFIG_DIR = Path(os.environ.get('RAG_CONFIG_DIR', Path(__file__).parent.parent / 'config'))
DEFAULT_CONFIG_FILE = Path(os.environ.get('RAG_DEFAULT_CONFIG', DEFAULT_CONFIG_DIR / 'default_config.yaml'))
ENTITY_PATTERNS_FILE = Path(os.environ.get('RAG_ENTITY_PATTERNS', DEFAULT_CONFIG_DIR / 'entity_patterns.yaml'))


class EntityPrefix(BaseModel):
    """
    Model for entity type prefix configuration.
    
    Args:
        prefix: The prefix to use for this entity type
        description: Description of this entity type
    """
    prefix: str = Field(..., description='The prefix to use for this entity type')
    description: str = Field('', description='Description of this entity type')


class ControlledTerm(BaseModel):
    """
    Model for controlled vocabulary term configuration.
    
    Args:
        preferred: The preferred term to use
        alternatives: Alternative terms that should be replaced
    """
    preferred: str = Field(..., description='The preferred term to use')
    alternatives: List[str] = Field(default_factory=list, description='Alternative terms that should be replaced')


class DocumentTemplate(BaseModel):
    """
    Model for document template configuration.
    
    Args:
        title: Title template
        notes: Document notes template
        body: Body template
        footer: Footer template
        full_template: Complete document template
    """
    title: str = Field('{title}', description='Title template')
    notes: str = Field('', description='Document notes template')
    body: str = Field('', description='Body template')
    footer: str = Field('', description='Footer template')
    full_template: str = Field('', description='Complete document template')


class EntityPattern(BaseModel):
    """
    Model for entity recognition pattern configuration.
    
    Args:
        type: Entity type
        patterns: Regex patterns for this entity type
        examples: Example matches
    """
    type: str = Field(..., description='Entity type')
    patterns: List[str] = Field(..., description='Regex patterns for this entity type')
    examples: List[str] = Field(default_factory=list, description='Example matches')


class ConverterConfig(BaseModel):
    """
    Main configuration model for the document converter.
    
    Args:
        entity_prefixes: Dictionary of entity prefixes by type
        controlled_vocabulary: Dictionary of controlled vocabulary terms
        document_template: Document template configuration
        entity_patterns: List of entity recognition patterns
    """
    entity_prefixes: Dict[str, EntityPrefix] = Field(default_factory=dict)
    controlled_vocabulary: Dict[str, ControlledTerm] = Field(default_factory=dict)
    document_template: DocumentTemplate = Field(default_factory=DocumentTemplate)
    entity_patterns: List[EntityPattern] = Field(default_factory=list)


def get_env_value(key: str, default: Any = None) -> Any:
    """
    Get a value from environment variables with fallback.
    
    Args:
        key: Environment variable name
        default: Default value if not found in environment
        
    Returns:
        Value from environment or default
    """
    return os.environ.get(key, default)


def load_yaml_config(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Safely load a YAML configuration file.
    
    Args:
        file_path: Path to the YAML file
    
    Returns:
        Dictionary containing the loaded configuration
    
    Raises:
        FileNotFoundError: If the file doesn't exist
        yaml.YAMLError: If the YAML is invalid
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    
    if not file_path.exists():
        raise FileNotFoundError(f'Configuration file not found: {file_path}')
    
    with open(file_path, 'r', encoding=get_env_value('FILE_ENCODING', 'utf-8')) as f:
        try:
            config = yaml.safe_load(f)
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f'Error parsing YAML configuration: {e}')


def load_entity_patterns(file_path: Optional[Union[str, Path]] = None) -> List[EntityPattern]:
    """
    Load entity recognition patterns from a YAML file.
    
    Args:
        file_path: Path to the entity patterns file (uses default if None)
    
    Returns:
        List of EntityPattern objects
    """
    if file_path is None:
        file_path = ENTITY_PATTERNS_FILE
    
    config = load_yaml_config(file_path)
    patterns = []
    
    for pattern_dict in config.get('entity_patterns', []):
        pattern = EntityPattern(**pattern_dict)
        patterns.append(pattern)
    
    return patterns


def load_config(config_path: Optional[Union[str, Path]] = None) -> ConverterConfig:
    """
    Load the converter configuration.
    
    Args:
        config_path: Path to a custom configuration file (uses default if None)
    
    Returns:
        ConverterConfig object with the loaded configuration
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_FILE
    
    # Load base configuration
    config_dict = load_yaml_config(config_path)
    
    # Process entity prefixes
    entity_prefixes = {}
    for entity_type, prefix_data in config_dict.get('entity_prefixes', {}).items():
        if isinstance(prefix_data, str):
            entity_prefixes[entity_type] = EntityPrefix(prefix=prefix_data)
        else:
            entity_prefixes[entity_type] = EntityPrefix(**prefix_data)
    
    # Process controlled vocabulary
    controlled_vocab = {}
    for term, term_data in config_dict.get('controlled_vocabulary', {}).items():
        if isinstance(term_data, list):
            controlled_vocab[term] = ControlledTerm(preferred=term, alternatives=term_data)
        else:
            controlled_vocab[term] = ControlledTerm(**term_data)
    
    # Use constants as defaults if not in config
    if not entity_prefixes:
        for entity_type, prefix in ENTITY_TYPE_PREFIXES.items():
            entity_prefixes[entity_type] = EntityPrefix(prefix=prefix)
    
    if not controlled_vocab:
        for term, alternatives in CONTROLLED_VOCABULARY.items():
            controlled_vocab[term] = ControlledTerm(preferred=term, alternatives=alternatives)
    
    # Load document template
    template_dict = config_dict.get('document_template', {})
    if not template_dict.get('full_template'):
        template_dict['full_template'] = DOCUMENT_TEMPLATE
    
    document_template = DocumentTemplate(**template_dict)
    
    # Load entity patterns if not provided in main config
    entity_patterns = []
    if 'entity_patterns' in config_dict:
        for pattern_dict in config_dict['entity_patterns']:
            entity_patterns.append(EntityPattern(**pattern_dict))
    else:
        try:
            entity_patterns = load_entity_patterns()
        except FileNotFoundError:
            # No entity patterns file, use empty list
            pass
    
    # Create the final configuration
    return ConverterConfig(
        entity_prefixes=entity_prefixes,
        controlled_vocabulary=controlled_vocab,
        document_template=document_template,
        entity_patterns=entity_patterns
    )


def get_default_config() -> ConverterConfig:
    """
    Get the default configuration using the built-in constants.
    
    Returns:
        ConverterConfig object with default settings
    """
    # Create entity prefixes
    entity_prefixes = {}
    for entity_type, prefix in ENTITY_TYPE_PREFIXES.items():
        entity_prefixes[entity_type] = EntityPrefix(prefix=prefix)
    
    # Create controlled vocabulary
    controlled_vocab = {}
    for term, alternatives in CONTROLLED_VOCABULARY.items():
        controlled_vocab[term] = ControlledTerm(preferred=term, alternatives=alternatives)
    
    # Create document template
    document_template = DocumentTemplate(full_template=DOCUMENT_TEMPLATE)
    
    return ConverterConfig(
        entity_prefixes=entity_prefixes,
        controlled_vocabulary=controlled_vocab,
        document_template=document_template
    )
