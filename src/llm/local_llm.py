"""Unified LocalLLM interface using provider abstraction"""

from typing import Optional, Iterator
from loguru import logger
from config.config import get_config
from src.llm.providers import get_provider, BaseProvider


class LocalLLM:
    """Unified interface for local LLM providers (Ollama, OpenAI-compatible, etc.)"""

    def __init__(self, provider_type: str = None, model_name: str = None, base_url: str = None):
        """
        Initialize LocalLLM with explicit provider selection

        Args:
            provider_type: "ollama" or "openai_compatible" (uses config if None)
            model_name: Model identifier (uses config if None)
            base_url: Base URL for provider (uses config if None)
        """
        self.config = get_config()
        self.provider_type = provider_type or self.config.llm_provider
        self.model_name = model_name or self.config.llm_model
        self.base_url = base_url or self.config.llm_base_url

        # Create provider instance
        self.provider: BaseProvider = get_provider(
            provider_type=self.provider_type,
            model_name=self.model_name,
            base_url=self.base_url,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens,
            top_p=self.config.llm_top_p
        )

        logger.info(f"LocalLLM: {self.provider_type} provider, model={self.model_name}")

    def generate(self, prompt: str, temperature: Optional[float] = None,
                max_tokens: Optional[int] = None) -> str:
        """
        Generate text using configured provider

        Args:
            prompt: Input prompt
            temperature: Override default temperature (0.0-2.0, None uses config)
            max_tokens: Override default max tokens (None uses config)

        Returns:
            Generated text
        """
        try:
            # Pass through to provider, allowing None values
            if temperature is not None or max_tokens is not None:
                # Create temporary provider with overrides for this call
                temp_provider = get_provider(
                    provider_type=self.provider_type,
                    model_name=self.model_name,
                    base_url=self.base_url,
                    temperature=temperature if temperature is not None else self.config.llm_temperature,
                    max_tokens=max_tokens if max_tokens is not None else self.config.llm_max_tokens,
                    top_p=self.config.llm_top_p
                )
                return temp_provider.generate(prompt)
            else:
                return self.provider.generate(prompt)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Stream text generation (yields tokens)

        Args:
            prompt: Input prompt

        Yields:
            Text tokens
        """
        try:
            for token in self.provider.stream(prompt):
                yield token
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise

    def is_available(self) -> bool:
        """Check if LLM provider is available and healthy"""
        try:
            return self.provider.health_check()
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False

    def list_models(self) -> list:
        """List available models from provider"""
        try:
            return self.provider.list_models()
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []
