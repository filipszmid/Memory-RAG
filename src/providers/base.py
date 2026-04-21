"""
src/providers/base.py
================================
Abstract base class and telemetry decorator for all LLM providers.

New providers (e.g., Anthropic, Mistral) must:
1. Subclass ``LLMProvider``.
2. Implement ``_extract_tokens``, ``_extract_parsed``, and ``generate_facts``.
3. Apply the ``@calculate_telemetry`` decorator to ``generate_facts`` AFTER
   any retry decorator so that latency measurement wraps the full retry cycle.
4. Register themselves in ``ProviderFactory``.
"""

import functools
import time
from abc import ABC, abstractmethod
from typing import Any

from src.core.models import FactsResponse


def calculate_telemetry(func):
    """
    Decorator that wraps an LLM provider's ``generate_facts`` method to:

    - Measure wall-clock latency of the full API call (including retries).
    - Extract prompt / completion token counts via the provider's
      ``_extract_tokens`` static method.
    - Compute an estimated USD cost using the provider's ``price_in_1m`` /
      ``price_out_1m`` class attributes.
    - Return a ``(FactsResponse, metrics_dict)`` tuple instead of the raw
      SDK response, giving the pipeline a uniform interface regardless of
      which provider is active.

    Args:
        func: The ``generate_facts`` method of an ``LLMProvider`` subclass.

    Returns:
        Callable: Wrapped function returning ``(FactsResponse, dict)``.
    """

    @functools.wraps(func)
    def wrapper(self, prompt: str) -> tuple[FactsResponse, dict[str, Any]]:
        start_time = time.perf_counter()

        raw_resp = func(self, prompt)

        latency = time.perf_counter() - start_time
        prompt_t, comp_t = self._extract_tokens(raw_resp)
        parsed_data = self._extract_parsed(raw_resp)

        cost = (prompt_t / 1_000_000 * self.price_in_1m) + (
            comp_t / 1_000_000 * self.price_out_1m
        )

        metrics = {
            "api_latency_sec": round(latency, 2),
            "prompt_tokens": prompt_t,
            "completion_tokens": comp_t,
            "estimated_cost_usd": round(cost, 6),
        }
        return parsed_data, metrics

    return wrapper


class LLMProvider(ABC):
    """
    Abstract base class for all LLM providers used in fact extraction.

    Subclasses define provider-specific SDK calls while the pipeline
    interacts only with this uniform interface.

    Class Attributes:
        price_in_1m (float): Cost in USD per 1 M prompt tokens.
        price_out_1m (float): Cost in USD per 1 M completion tokens.
    """

    price_in_1m: float = 0.0
    price_out_1m: float = 0.0

    def __init__(self, model: str, timeout: int = 60) -> None:
        """
        Initializes the LLM provider with a specific model identifier and timeout.

        Args:
            model (str): The name/ID of the LLM model to use.
            timeout (int): API timeout in seconds.
        """
        self.model = model
        self.timeout = timeout

    @staticmethod
    @abstractmethod
    def _extract_tokens(raw_resp: Any) -> tuple[int, int]:
        """
        Extracts prompt and completion token counts from the raw SDK response.

        Args:
            raw_resp (Any): The raw response object returned by the provider SDK.

        Returns:
            tuple[int, int]: ``(prompt_tokens, completion_tokens)``.
        """

    @staticmethod
    @abstractmethod
    def _extract_parsed(raw_resp: Any) -> FactsResponse:
        """
        Extracts the typed Pydantic model from the raw SDK response.

        Args:
            raw_resp (Any): The raw response object returned by the provider SDK.

        Returns:
            FactsResponse: Structured object containing the extracted facts.
        """

    @abstractmethod
    def generate_facts(self, prompt: str) -> Any:
        """
        Calls the underlying LLM API and returns the native SDK response.

        The ``@calculate_telemetry`` decorator must be applied by the subclass
        to wrap the return value into a ``(FactsResponse, metrics)`` tuple.

        Args:
            prompt (str): Full user context prompt to send to the LLM.

        Returns:
            Any: Raw SDK response (decorated into a tuple by ``calculate_telemetry``).
        """
