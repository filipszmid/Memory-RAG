"""
src/providers/litellm_provider.py
============================================
Unified provider using LiteLLM and Instructor for structured output.
Supports caching, traceability, and multiple backends (OpenAI, Gemini, Ollama).
"""

import litellm
import instructor
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger
from pydantic import BaseModel

from src.config.settings import settings
from src.core.models import FactsResponse
from src.core.prompts import SYSTEM_PROMPT
from src.providers.base import LLMProvider, calculate_telemetry

# Set up LiteLLM caching
if settings.litellm_cache:
    litellm.cache = litellm.Cache()

class LiteLLMProvider(LLMProvider):
    """
    Unified provider implementation using LiteLLM.
    """

    def __init__(self, model: str, provider: str = "openai", timeout: int = 60):
        super().__init__(model, timeout=timeout)
        self.provider = provider.lower()
        
        # Configure LiteLLM model string
        if self.provider == "openai":
            self.litellm_model = f"openai/{model}"
            self.api_key = settings.openai_api_key
        elif self.provider == "gemini":
            self.litellm_model = f"gemini/{model}"
            self.api_key = settings.gemini_api_key
        elif self.provider == "ollama":
            self.litellm_model = f"ollama/{model}"
            self.api_key = "not-needed"
            self.base_url = settings.ollama_base_url
        else:
            self.litellm_model = model
            self.api_key = None

        # Price settings (can be further refined)
        if self.provider == "openai":
            self.price_in_1m = settings.openai_price_in_1m
            self.price_out_1m = settings.openai_price_out_1m
        elif self.provider == "gemini":
            self.price_in_1m = settings.gemini_price_in_1m
            self.price_out_1m = settings.gemini_price_out_1m
        else:
            self.price_in_1m = 0.0
            self.price_out_1m = 0.0

        # Initialize Instructor client
        # instructor.patch() for LiteLLM
        self.instructor_client = instructor.from_litellm(litellm.completion)

    @staticmethod
    def _extract_tokens(raw_resp: Any) -> Tuple[int, int]:
        # Instructor returns the parsed object directly, but we can access the raw response 
        # if we use the right return mode. For simplicity, we'll extract from usage if available.
        if hasattr(raw_resp, "_raw_response") and hasattr(raw_resp._raw_response, "usage"):
            usage = raw_resp._raw_response.usage
            return usage.prompt_tokens, usage.completion_tokens
        # Fallback if LiteLLM provides it elsewhere
        return 0, 0

    @staticmethod
    def _extract_parsed(raw_resp: Any) -> FactsResponse:
        # Instructor already returns the parsed object
        if isinstance(raw_resp, FactsResponse):
            return raw_resp
        return raw_resp

    @calculate_telemetry
    def generate_facts(self, prompt: str) -> Any:
        """
        Generates facts using LiteLLM + Instructor for structured output.
        """
        logger.info(f"Using LiteLLM for extraction: {self.litellm_model}")
        
        # instructor.from_litellm allows us to use response_model
        try:
            # We use validation_context or similar if needed, 
            # but mainly we want the response_model.
            # Instructor patch for litellm.completion
            resp = self.instructor_client.chat.completions.create(
                model=self.litellm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_model=FactsResponse,
                api_key=self.api_key,
                base_url=getattr(self, "base_url", None),
                timeout=self.timeout
            )
            return resp
        except Exception as e:
            logger.error(f"LiteLLM call failed: {e}")
            raise
