"""Core sub-package: Pydantic models and system prompt."""

from src.core.models import Fact, FactsResponse
from src.core.prompts import SYSTEM_PROMPT

__all__ = ["Fact", "FactsResponse", "SYSTEM_PROMPT"]
