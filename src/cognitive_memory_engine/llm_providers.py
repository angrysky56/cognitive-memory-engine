"""
Flexible LLM Provider System

Provides a unified interface for different LLM providers with dynamic model discovery.
Supports Ollama, OpenAI, Google (Gemini), and Anthropic.
"""

from abc import ABC, abstractmethod

import anthropic
import google.genai as genai
import ollama
import openai


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    async def get_available_models(self) -> list[str]:
        """List available models for this provider."""
        pass

    @abstractmethod
    def get_current_model(self) -> str:
        """Get the current model name."""
        pass

    @abstractmethod
    def set_model(self, model_name: str) -> None:
        """Set the model to use."""
        pass

    @abstractmethod
    def validate(self) -> None:
        """Validate the provider configuration."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama provider implementation."""

    def validate(self) -> None:
        if ollama is None:
            raise ImportError("ollama package not installed. Run: pip install ollama")
        try:
            ollama.show(self.config.model)
        except Exception as e:
            raise RuntimeError(f"Ollama model {self.config.model} not available: {e}") from e

    async def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        response = ollama.generate(
            model=self.config.model,
            prompt=prompt.strip(),
            options={"temperature": temperature, **kwargs}
        )
        return response['response']

    async def get_available_models(self) -> list[str]:
        """List available Ollama models."""
        try:
            models = ollama.list()
            return [model['name'] for model in models.get('models', [])]
        except Exception:
            return []

    def get_current_model(self) -> str:
        return self.config.model

    def set_model(self, model_name: str) -> None:
        self.config.model = model_name


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation."""

    def validate(self) -> None:
        if openai is None:
            raise ImportError("openai package not installed. Run: pip install openai")
        if not self.config.api_key:
            raise RuntimeError("OpenAI API key not provided")

    async def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )

        response = client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt.strip()}],
            temperature=temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )

        return response.choices[0].message.content

    async def get_available_models(self) -> list[str]:
        """List available OpenAI models."""
        try:
            client = openai.OpenAI(api_key=self.config.api_key)
            models = client.models.list()
            return [model.id for model in models.data if 'gpt' in model.id]
        except Exception:
            return ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

    def get_current_model(self) -> str:
        return self.config.model

    def set_model(self, model_name: str) -> None:
        self.config.model = model_name


class GoogleProvider(LLMProvider):
    """Google (Gemini) provider implementation using the new unified SDK."""

    def validate(self) -> None:
        if genai is None:
            raise ImportError("google-genai package not installed. Run: pip install google-genai")
        if not self.config.api_key:
            raise RuntimeError("Google API key not provided")

    async def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        from google.genai import types

        try:
            # Create client with API key
            client = genai.Client(api_key=self.config.api_key)

            # Generate content using the correct API
            response = client.models.generate_content(
                model=self.config.model,
                contents=prompt.strip(),
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=self.config.max_tokens,
                    **kwargs
                )
            )

            return response.text or ""

        except Exception:
            # Fallback to basic generation if advanced config fails
            client = genai.Client(api_key=self.config.api_key)
            response = client.models.generate_content(
                model=self.config.model,
                contents=prompt.strip()
            )
            return response.text or ""

    async def get_available_models(self) -> list[str]:
        """List available Gemini models."""
        try:
            client = genai.Client(api_key=self.config.api_key)
            # Try to list models, fallback to known models if unavailable
            models = client.models.list()
            return [model.name for model in models if model.name is not None]
        except Exception:
            # Return known available models as fallback
            return [
                "gemini-2.0-flash-001",
                "gemini-1.5-flash-001",
                "gemini-1.5-pro-001",
                "gemini-2.0-flash-exp"
            ]

    def get_current_model(self) -> str:
        return self.config.model

    def set_model(self, model_name: str) -> None:
        self.config.model = model_name


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) provider implementation."""

    def validate(self) -> None:
        if anthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        if not self.config.api_key:
            raise RuntimeError("Anthropic API key not provided")

    async def generate(self, prompt: str, temperature: float = 0.7, **kwargs) -> str:
        client = anthropic.Anthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url if self.config.base_url != "https://api.anthropic.com" else None
        )

        response = client.messages.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt.strip()}],
            temperature=temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )

        # Extract text from content blocks
        return "".join(block.text for block in response.content if hasattr(block, 'text'))

    async def get_available_models(self) -> list[str]:
        """List available Anthropic models."""
        # Anthropic doesn't have a list models API, so we return known models
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
            "claude-4-sonnet",
            "claude-4-opus"
        ]

    def get_current_model(self) -> str:
        return self.config.model

    def set_model(self, model_name: str) -> None:
        self.config.model = model_name


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    _providers = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
        "google": GoogleProvider,
        "anthropic": AnthropicProvider
    }

    @classmethod
    def create(cls, config) -> LLMProvider:
        """Create an LLM provider based on configuration."""
        provider_name = config.provider.lower()

        if provider_name not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_name}. Supported: {list(cls._providers.keys())}")

        provider_class = cls._providers[provider_name]
        provider = provider_class(config)
        provider.validate()

        return provider

    @classmethod
    def register_provider(cls, name: str, provider_class: type[LLMProvider]):
        """Register a custom provider."""
        cls._providers[name.lower()] = provider_class

    @classmethod
    def list_providers(cls) -> list[str]:
        """List available providers."""
        return list(cls._providers.keys())


async def list_all_available_models(config) -> dict[str, list[str]]:
    """List all available models across all configured providers."""
    available_models = {}

    for provider_name in LLMProviderFactory.list_providers():
        try:
            # Create a temporary config for this provider
            temp_config = type('Config', (), {
                'provider': provider_name,
                'api_key': config.api_key if hasattr(config, 'api_key') else '',
                'base_url': getattr(config, 'base_url', ''),
                'model': '',
                'max_tokens': 8164
            })()

            provider = LLMProviderFactory.create(temp_config)
            models = await provider.get_available_models()
            if models:
                available_models[provider_name] = models
        except Exception:
            # Skip providers that aren't configured or available
            continue

    return available_models
