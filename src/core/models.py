"""
src/core/models.py
=============================
Pydantic domain models for the fact extraction pipeline.

Two models are defined here:
- ``Fact``          — a single atomic, permanent user fact with category.
- ``FactsResponse`` — the top-level wrapper holding a list of facts,
                      used as the structured response format for both
                      Gemini (response_schema) and OpenAI (response_format).

Both models apply validation at field level:
- Category must be one of the keys in ``Settings.valid_categories``.
- Fact text is checked against PII regex patterns and must be non-empty.
"""

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from src.config.settings import PII_PATTERNS, settings


class Fact(BaseModel):
    """
    Represents a single atomic fact extracted from a user's conversation.

    Attributes:
        fact (str): The actual fact extracted (e.g., "User has a Toyota Yaris").
        category (str): The category the fact belongs to (one of valid_categories).
    """

    fact: str = Field(description="The atomic permanent fact about the user")
    category: str = Field(description="Category of the fact")
    fact_key: str | None = Field(
        default=None,
        description="Unique identifier for mutually exclusive traits (e.g. location, marital_status). Leave null if not exclusive.",
    )
    confidence: float = Field(
        default=1.0,
        description="Reliability score of the extraction (0.0 to 1.0)",
    )

    @field_validator("category")
    @classmethod
    def check_category(cls, value: str) -> str:
        """
        Validates that the provided category is within the predefined valid categories.

        Args:
            value (str): The category string to validate.

        Returns:
            str: The stripped and lowercased category string.

        Raises:
            ValueError: If the category is not found in settings.valid_categories.
        """
        value = value.strip().lower()
        if value not in settings.valid_categories:
            logger.warning(f"Invalid category '{value}' detected. Skipping.")
            raise ValueError(f"Category '{value}' is not valid.")
        return value

    @field_validator("fact")
    @classmethod
    def check_pii(cls, value: str) -> str:
        """
        Filters out sensitive PII data like credit cards, SSNs, or obvious passwords.

        Args:
            value (str): The fact text to check for PII.

        Returns:
            str: The original fact text if no PII is found.

        Raises:
            ValueError: If PII patterns are detected in the fact text.
        """
        value = value.strip()
        if not value:
            raise ValueError("Fact cannot be empty.")
        if any(p.search(value) for p in PII_PATTERNS):
            logger.warning(f"[PII FILTER] Removed: {value!r}")
            raise ValueError("PII detected in fact.")
        return value


class FactsResponse(BaseModel):
    """
    Root structure containing a list of extracted facts.

    Used as the structured output schema for both LLM providers so that
    the API returns valid, typed JSON that maps directly to this model.

    Attributes:
        facts (list[Fact]): A list of extracted user facts.
    """

    facts: list[Fact]
