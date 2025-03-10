"""
LLM client module for interacting with language models.

This module provides client classes for interacting with Ollama (local)
and Anthropic (cloud) language models.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

class LLMResponse(BaseModel):
    """
    Standardized response model for LLM outputs.
    
    Args:
        content: The text content from the LLM response
        raw_response: The original response object from the provider
    """
    content: str
    raw_response: Optional[Any] = None


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    This class defines the interface that all LLM client implementations must follow.
    """
    
    @abstractmethod
    def generate(self, prompt: str, system_message: Optional[str] = None, 
                temperature: float = 0.7, max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Generate a response from the language model.
        
        Args:
            prompt: The user prompt to send to the model
            system_message: Optional system message for context
            temperature: Controls randomness (0.0-1.0)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            LLMResponse object containing the generated text
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM service is available.
        
        Returns:
            True if the service is available, False otherwise
        """
        pass


class OllamaClient(LLMClient):
    """
    Client for interacting with Ollama local language models.
    """
    
    def __init__(self, model_name: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the Ollama client.
        
        Args:
            model_name: Name of the model to use (defaults to env var OLLAMA_MODEL or 'llama2')
            base_url: Base URL for the Ollama API (defaults to env var OLLAMA_BASE_URL)
        """
        self.model_name = model_name or os.environ.get('OLLAMA_MODEL', 'llama2')
        self.base_url = base_url or os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
        self._client = None
        self._initialize_client()
        
        logger.info(f'Initialized Ollama client with model: {self.model_name}')
    
    def _initialize_client(self) -> None:
        """
        Initialize the Ollama client.
        """
        try:
            import ollama
            # Set the base URL for the client
            ollama.base_url = self.base_url
            self._client = ollama
            logger.info('Ollama client initialized successfully')
        except ImportError:
            logger.warning('Ollama package not installed. Run: pip install ollama')
            self._client = None
        except Exception as e:
            logger.error(f'Failed to initialize Ollama client: {str(e)}')
            self._client = None
    
    def generate(self, prompt: str, system_message: Optional[str] = None,
                temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Generate a response from the Ollama model.
        
        Args:
            prompt: The user prompt to send to the model
            system_message: Optional system message for context
            temperature: Controls randomness (0.0-1.0, defaults to env var OLLAMA_TEMPERATURE)
            max_tokens: Maximum number of tokens to generate (defaults to env var OLLAMA_MAX_TOKENS)
            
        Returns:
            LLMResponse object containing the generated text
        """
        if not self._client:
            logger.error('Ollama client not initialized')
            return LLMResponse(content='Error: Ollama client not initialized')
        
        try:
            # Get defaults from environment variables if not specified
            temp = temperature if temperature is not None else float(os.environ.get('OLLAMA_TEMPERATURE', '0.7'))
            max_tok = max_tokens or int(os.environ.get('OLLAMA_MAX_TOKENS', '0')) or None
            timeout = int(os.environ.get('OLLAMA_TIMEOUT', '120'))
            
            options = {
                'temperature': temp,
                'timeout': timeout
            }
            
            if max_tok:
                options['num_predict'] = max_tok
                
            messages = []
            if system_message:
                messages.append({'role': 'system', 'content': system_message})
            
            messages.append({'role': 'user', 'content': prompt})
            
            response = self._client.chat(model=self.model_name, messages=messages, options=options)
            
            return LLMResponse(
                content=response['message']['content'],
                raw_response=response
            )
        except Exception as e:
            logger.error(f'Error generating response from Ollama: {str(e)}')
            return LLMResponse(content=f'Error: {str(e)}')
    
    def is_available(self) -> bool:
        """
        Check if Ollama service is available.
        
        Returns:
            True if Ollama is available, False otherwise
        """
        if not self._client:
            return False
        
        try:
            # List models to check if Ollama is responsive
            self._client.list()
            return True
        except Exception:
            return False


class AnthropicClient(LLMClient):
    """
    Client for interacting with Anthropic Claude language models.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the Anthropic client.
        
        Args:
            api_key: Anthropic API key (defaults to env var ANTHROPIC_API_KEY)
            model_name: Name of the model to use (defaults to env var ANTHROPIC_MODEL)
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.model_name = model_name or os.environ.get('ANTHROPIC_MODEL', 'claude-3-haiku-20240307')
        self._client = None
        self._initialize_client()
        
        logger.info(f'Initialized Anthropic client with model: {self.model_name}')
    
    def _initialize_client(self) -> None:
        """
        Initialize the Anthropic client.
        """
        if not self.api_key:
            logger.error('No Anthropic API key provided')
            return
            
        try:
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
            logger.info('Anthropic client initialized successfully')
        except ImportError:
            logger.warning('Anthropic package not installed. Run: pip install anthropic')
            self._client = None
        except Exception as e:
            logger.error(f'Failed to initialize Anthropic client: {str(e)}')
            self._client = None
    
    def generate(self, prompt: str, system_message: Optional[str] = None,
                temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Generate a response from the Anthropic model.
        
        Args:
            prompt: The user prompt to send to the model
            system_message: Optional system message for context
            temperature: Controls randomness (defaults to env var ANTHROPIC_TEMPERATURE)
            max_tokens: Maximum number of tokens to generate (defaults to env var ANTHROPIC_MAX_TOKENS)
            
        Returns:
            LLMResponse object containing the generated text
        """
        if not self._client:
            logger.error('Anthropic client not initialized')
            return LLMResponse(content='Error: Anthropic client not initialized')
        
        try:
            # Get defaults from environment variables if not specified
            temp = temperature if temperature is not None else float(os.environ.get('ANTHROPIC_TEMPERATURE', '0.7'))
            max_tok = max_tokens or int(os.environ.get('ANTHROPIC_MAX_TOKENS', '4000'))
            
            kwargs = {
                'model': self.model_name,
                'temperature': temp,
                'messages': [{'role': 'user', 'content': prompt}]
            }
            
            if system_message:
                kwargs['system'] = system_message
                
            if max_tok:
                kwargs['max_tokens'] = max_tok
                
            response = self._client.messages.create(**kwargs)
            
            return LLMResponse(
                content=response.content[0].text,
                raw_response=response
            )
        except Exception as e:
            logger.error(f'Error generating response from Anthropic: {str(e)}')
            return LLMResponse(content=f'Error: {str(e)}')
    
    def is_available(self) -> bool:
        """
        Check if Anthropic service is available.
        
        Returns:
            True if Anthropic client is initialized and API key is set, False otherwise
        """
        return self._client is not None and self.api_key is not None


class LLMConfig(BaseModel):
    """
    Configuration for LLM client.
    
    Args:
        provider: LLM provider to use ('ollama' or 'anthropic')
        model_name: Name of the model to use
        api_key: API key for cloud providers
        base_url: Base URL for API (for Ollama)
    """
    provider: str = Field(os.environ.get('LLM_PROVIDER', 'ollama'), description='LLM provider to use ("ollama" or "anthropic")')
    model_name: str = Field(None, description='Name of the model to use')
    api_key: Optional[str] = Field(None, description='API key for cloud providers')
    base_url: str = Field(None, description='Base URL for API (for Ollama)')
    temperature: float = Field(None, description='Temperature for generation')
    max_tokens: int = Field(None, description='Maximum tokens for generation')
    
    @field_validator('provider')
    def validate_provider(cls, v):
        """
        Validate that the provider is supported.
        """
        if v.lower() not in ('ollama', 'anthropic'):
            raise ValueError('Provider must be "ollama" or "anthropic"')
        return v.lower()


def create_llm_client(config: Optional[Union[Dict, LLMConfig]] = None) -> LLMClient:
    """
    Create an LLM client based on configuration.
    
    This function will try to create a client in this order:
    1. Using provided configuration
    2. Using Ollama if available
    3. Using Anthropic if API key is available
    4. Raise an exception if no client can be created
    
    Args:
        config: Configuration dictionary or LLMConfig object
        
    Returns:
        An initialized LLM client
        
    Raises:
        ValueError: If no suitable client can be created
    """
    if config is None:
        config = {}
    
    # Convert dict to config object if needed
    if isinstance(config, dict):
        config = LLMConfig(**config)
    
    # Determine model names from environment if not specified
    model_name = config.model_name
    base_url = config.base_url
    api_key = config.api_key
    
    # Try to create client based on specified provider
    provider = config.provider or os.environ.get('LLM_PROVIDER', 'ollama')
    
    if provider == 'ollama':
        client = OllamaClient(
            model_name=model_name,
            base_url=base_url
        )
        if client.is_available():
            return client
            
        # If Ollama is specified but not available, log warning
        logger.warning('Ollama specified but not available, trying Anthropic')
    
    # Try Anthropic if Ollama is not available or Anthropic is specified
    if provider == 'anthropic' or api_key or os.environ.get('ANTHROPIC_API_KEY'):
        client = AnthropicClient(api_key=api_key, model_name=model_name)
        if client.is_available():
            return client
    
    # If we reached here, try Ollama one more time as a fallback (if not already tried)
    if provider != 'ollama':
        client = OllamaClient(model_name=model_name, base_url=base_url)
        if client.is_available():
            logger.info('Using Ollama as fallback')
            return client
    
    # If we still don't have a client, raise error
    raise ValueError(
        'Could not create LLM client. Make sure Ollama is running or '
        'ANTHROPIC_API_KEY environment variable is set.'
    )
