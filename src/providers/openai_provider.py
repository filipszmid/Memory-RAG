"""
src/providers/openai_provider.py
===========================================
OpenAI implementation of the LLMProvider.

Uses the ``openai`` SDK and supports structured output via the ``parse`` method.
Named ``openai_provider`` to avoid conflict with the ``openai`` package name.
"""

import openai
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


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider implementation.
    """
    price_in_1m: float = settings.openai_price_in_1m
    price_out_1m: float = settings.openai_price_out_1m

    def __init__(self, model: str, timeout: int = 60):
        super().__init__(model, timeout=timeout)
        if not settings.openai_api_key:
            message = "OPENAI_API_KEY not set in environment or .env"
            logger.error(message)
            raise RuntimeError(message)
        self.client = openai.OpenAI(api_key=settings.openai_api_key)

    @staticmethod
    def _extract_tokens(raw_resp: any) -> tuple[int, int]:
        if hasattr(raw_resp, "usage") and raw_resp.usage:
            return raw_resp.usage.prompt_tokens, raw_resp.usage.completion_tokens
        return 0, 0

    @staticmethod
    def _extract_parsed(raw_resp: any) -> FactsResponse:
        return raw_resp.choices[0].message.parsed

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(settings.max_retries),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
                openai.APITimeoutError,
            )
        ),
        reraise=True,
    )
    @calculate_telemetry
    def generate_facts(self, prompt: str) -> any:
        """
        Generates facts using the OpenAI API.
        """
        resp = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format=FactsResponse,
        )
        return resp
