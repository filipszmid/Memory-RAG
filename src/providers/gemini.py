"""
src/providers/gemini.py
==================================
Google Gemini implementation of the LLMProvider.

Uses the ``google-genai`` SDK and supports structured output via
the ``response_schema`` configuration.
"""

from google import genai
from google.genai import types
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config.settings import settings
from src.core.models import FactsResponse
from src.core.prompts import SYSTEM_PROMPT
from src.providers.base import LLMProvider, calculate_telemetry


class GeminiProvider(LLMProvider):
    """
    Google Gemini provider implementation.
    """
    price_in_1m: float = settings.gemini_price_in_1m
    price_out_1m: float = settings.gemini_price_out_1m

    def __init__(self, model: str, timeout: int = 60):
        super().__init__(model, timeout=timeout)
        if not settings.gemini_api_key:
            message = "GEMINI_API_KEY not set in environment or .env"
            logger.error(message)
            raise RuntimeError(message)
        self.client = genai.Client(api_key=settings.gemini_api_key)

    @staticmethod
    def _extract_tokens(raw_resp: any) -> tuple[int, int]:
        if hasattr(raw_resp, "usage_metadata") and raw_resp.usage_metadata:
            return (
                getattr(raw_resp.usage_metadata, "prompt_token_count", 0),
                getattr(raw_resp.usage_metadata, "candidates_token_count", 0),
            )
        return 0, 0

    @staticmethod
    def _extract_parsed(raw_resp: any) -> FactsResponse:
        return raw_resp.parsed

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(settings.max_retries),
        retry=retry_if_exception_type(genai.errors.APIError),
        reraise=True,
    )
    @calculate_telemetry
    def generate_facts(self, prompt: str) -> any:
        """
        Generates facts using the Gemini API.
        """
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=FactsResponse,
            ),
        )
        return resp
