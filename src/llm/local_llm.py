from typing import Optional
import requests
from loguru import logger
from config.config import get_config

class LocalLLM:
    """Interface with local LLM via Ollama or HuggingFace"""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        self.config = get_config()
        self.model_name = model_name or self.config.llm_model
        self.base_url = base_url or self.config.llm_base_url
        self.temperature = self.config.llm_temperature
        self.top_k = self.config.llm_top_k
        self.top_p = self.config.llm_top_p
        self.max_tokens = self.config.llm_max_tokens
        
        logger.info(f"LocalLLM initialized with model: {self.model_name}")
    
    def generate(self, prompt: str, temperature: Optional[float] = None, 
                max_tokens: Optional[int] = None) -> str:
        """Generate text using local LLM"""
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        try:
            # Call Ollama API
            response = requests.post(
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
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                logger.debug(f"Generated {len(generated_text)} characters")
                return generated_text
            else:
                logger.error(f"LLM API error: {response.status_code}")
                raise Exception(f"LLM API returned status {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to LLM at {self.base_url}")
            raise Exception(f"LLM connection error. Is Ollama running at {self.base_url}?")
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def is_available(self) -> bool:
        """Check if LLM is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
