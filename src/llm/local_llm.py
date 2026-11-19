from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from loguru import logger
from config.config import get_config
import time


class LocalLLM:
    """Interface with local LLM via Ollama or HuggingFace with connection pooling and retries"""

    def __init__(self, model_name: str = None, base_url: str = None):
        self.config = get_config()
        self.model_name = model_name or self.config.llm_model
        self.base_url = base_url or self.config.llm_base_url
        self.temperature = self.config.llm_temperature
        self.top_k = self.config.llm_top_k
        self.top_p = self.config.llm_top_p
        self.max_tokens = self.config.llm_max_tokens

        self.session = self._create_session()
        logger.info(f"LocalLLM initialized with model: {self.model_name}")

    def _create_session(self) -> requests.Session:
        """Create session with connection pooling and retry strategy"""
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

    def generate(self, prompt: str, temperature: Optional[float] = None,
                max_tokens: Optional[int] = None) -> str:
        """Generate text using local LLM with exponential backoff"""
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Check if using OpenAI-compatible API (LM Studio) or Ollama
                is_openai_compatible = "/v1" in self.base_url

                if is_openai_compatible:
                    # OpenAI-compatible format (LM Studio, vLLM, etc)
                    response = self.session.post(
                        f"{self.base_url}/chat/completions",
                        json={
                            "model": self.model_name,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": temperature,
                            "top_p": self.top_p,
                            "max_tokens": max_tokens,
                            "stream": False
                        },
                        timeout=60
                    )
                else:
                    # Ollama format
                    response = self.session.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": self.model_name,
                            "prompt": prompt,
                            "temperature": temperature,
                            "top_k": self.top_k,
                            "top_p": self.top_p,
                            "num_predict": max_tokens,
                            "stream": False
                        },
                        timeout=60
                    )

                if response.status_code == 200:
                    result = response.json()

                    # Parse response based on API type
                    if is_openai_compatible:
                        # OpenAI-compatible format - try chat first, then completions
                        choice = result.get("choices", [{}])[0]
                        generated_text = choice.get("message", {}).get("content") or choice.get("text", "")
                    else:
                        # Ollama returns {"response": "..."}
                        generated_text = result.get("response", "")

                    logger.debug(f"Generated {len(generated_text)} characters")
                    return generated_text
                elif response.status_code in [429, 500, 502, 503, 504]:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        logger.warning(f"LLM returned {response.status_code}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise Exception(f"LLM API returned status {response.status_code} after {max_retries} attempts")
                else:
                    logger.error(f"LLM API error: {response.status_code}")
                    error_detail = response.text if response.text else f"HTTP {response.status_code}"
                    logger.error(f"Response: {error_detail}")

                    if response.status_code == 400:
                        raise Exception(f"LLM API error 400: Bad request. Check model name '{self.model_name}' or prompt format. Response: {error_detail[:100]}")
                    elif response.status_code == 404:
                        raise Exception(f"LLM endpoint not found. Is Ollama running at {self.base_url}? Model '{self.model_name}' may not be loaded.")
                    else:
                        raise Exception(f"LLM API returned status {response.status_code}")

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"LLM request timeout, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"LLM request timeout after {max_retries} attempts")
                    raise Exception(f"LLM connection timeout. Is Ollama running at {self.base_url}?")

            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Cannot connect to LLM, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Cannot connect to LLM after {max_retries} attempts")
                    raise Exception(f"LLM connection error. Is Ollama running at {self.base_url}?")

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Error generating text: {e}, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Error generating text after {max_retries} attempts: {e}")
                    raise

        raise Exception("LLM generation failed after all retries")

    def is_available(self) -> bool:
        """Check if LLM is available"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"LLM availability check failed: {e}")
            return False
