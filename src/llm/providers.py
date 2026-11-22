"""LLM Provider abstraction - supports multiple inference backends"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Iterator
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger


class BaseProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, model_name: str, base_url: str, temperature: float = 0.3,
                 max_tokens: int = 1024, top_p: float = 0.9, timeout: int = 60):
        """
        Initialize provider with explicit parameter handling

        Args:
            model_name: Model identifier (e.g., "mistral-7b", "gpt-3.5-turbo")
            base_url: Base URL for provider API
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Max tokens to generate
            top_p: Nucleus sampling parameter (0.0-1.0)
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = self._validate_temperature(temperature)
        self.max_tokens = self._validate_max_tokens(max_tokens)
        self.top_p = self._validate_top_p(top_p)
        self.timeout = timeout
        self.session = self._create_session()

    @staticmethod
    def _validate_temperature(temp: Optional[float]) -> float:
        """Validate and use temperature, allowing zero"""
        if temp is None:
            return 0.3
        if not 0.0 <= temp <= 2.0:
            raise ValueError(f"Temperature must be 0.0-2.0, got {temp}")
        return temp

    @staticmethod
    def _validate_max_tokens(tokens: Optional[int]) -> int:
        """Validate and use max_tokens, allowing zero (but typically >= 128)"""
        if tokens is None:
            return 1024
        if tokens <= 0:
            raise ValueError(f"Max tokens must be > 0, got {tokens}")
        return tokens

    @staticmethod
    def _validate_top_p(top_p: Optional[float]) -> float:
        """Validate and use top_p, allowing zero"""
        if top_p is None:
            return 0.9
        if not 0.0 <= top_p <= 1.0:
            raise ValueError(f"top_p must be 0.0-1.0, got {top_p}")
        return top_p

    def _create_session(self) -> requests.Session:
        """Create session with retry strategy and connection pooling"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate text from prompt"""
        pass

    @abstractmethod
    def stream(self, prompt: str) -> Iterator[str]:
        """Stream text generation (yields tokens)"""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if provider is available"""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models"""
        pass


class OllamaProvider(BaseProvider):
    """Ollama local LLM provider"""

    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434",
                 temperature: float = 0.3, max_tokens: int = 1024, top_p: float = 0.9):
        super().__init__(model_name, base_url, temperature, max_tokens, top_p)
        logger.info(f"Initialized OllamaProvider: {model_name} at {base_url}")

    def generate(self, prompt: str) -> str:
        """Generate text using Ollama /api/generate endpoint"""
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
                "stream": False
            }

            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def stream(self, prompt: str) -> Iterator[str]:
        """Stream text generation from Ollama"""
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_predict": self.max_tokens,
                "stream": True
            }

            response = self.session.post(url, json=payload, timeout=self.timeout, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    data = response.json()
                    token = data.get("response", "")
                    if token:
                        yield token

        except Exception as e:
            logger.error(f"Ollama stream failed: {e}")
            raise

    def health_check(self) -> bool:
        """Check Ollama health using /api/tags"""
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    def list_models(self) -> List[str]:
        """List available models from Ollama"""
        try:
            url = f"{self.base_url}/api/tags"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list Ollama models: {e}")
            return []


class OpenAICompatibleProvider(BaseProvider):
    """OpenAI-compatible provider (LM Studio, vLLM, etc.)"""

    def __init__(self, model_name: str = "gpt-3.5-turbo", base_url: str = "http://localhost:1235/v1",
                 temperature: float = 0.3, max_tokens: int = 1024, top_p: float = 0.9):
        super().__init__(model_name, base_url, temperature, max_tokens, top_p)
        logger.info(f"Initialized OpenAICompatibleProvider: {model_name} at {base_url}")

    def generate(self, prompt: str) -> str:
        """Generate text using /v1/chat/completions endpoint"""
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "stream": False
            }

            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"OpenAI-compatible generation failed: {e}")
            raise

    def stream(self, prompt: str) -> Iterator[str]:
        """Stream text generation using OpenAI-compatible API"""
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "stream": True
            }

            response = self.session.post(url, json=payload, timeout=self.timeout, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line and line.startswith(b"data:"):
                    data_str = line.decode().replace("data:", "").strip()
                    if data_str and data_str != "[DONE]":
                        try:
                            import json
                            data = json.loads(data_str)
                            token = data["choices"][0]["delta"].get("content", "")
                            if token:
                                yield token
                        except Exception:
                            pass

        except Exception as e:
            logger.error(f"OpenAI-compatible stream failed: {e}")
            raise

    def health_check(self) -> bool:
        """Check health using /v1/models endpoint"""
        try:
            url = f"{self.base_url}/models"
            response = self.session.get(url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"OpenAI-compatible health check failed: {e}")
            return False

    def list_models(self) -> List[str]:
        """List available models from OpenAI-compatible server"""
        try:
            url = f"{self.base_url}/models"
            response = self.session.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []


def get_provider(provider_type: str, model_name: str, base_url: str,
                 temperature: float = 0.3, max_tokens: int = 1024,
                 top_p: float = 0.9) -> BaseProvider:
    """Factory function to create provider instances"""
    provider_type = provider_type.lower()

    if provider_type == "ollama":
        return OllamaProvider(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
    elif provider_type == "openai_compatible":
        return OpenAICompatibleProvider(
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
