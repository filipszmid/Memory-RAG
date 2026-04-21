"""
src/config/settings.py
=================================
Central configuration for the Personal Memory Module.

All tuneable values — API keys, model names, pricing, rate-limit helpers,
and PII regex patterns — live here and are loaded once at import time.
Other modules import from this file rather than from the environment directly.
"""

import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env relative to the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """
    Pydantic-settings model that reads configuration from environment variables
    or a `.env` file located at the project root.

    Attributes:
        gemini_api_key: API key for Google Gemini.
        openai_api_key: API key for OpenAI.
        gemini_default_model: Default Gemini model identifier.
        gemini_price_in_1m: Cost in USD per 1 M prompt tokens (Gemini).
        gemini_price_out_1m: Cost in USD per 1 M completion tokens (Gemini).
        openai_default_model: Default OpenAI model identifier.
        openai_price_in_1m: Cost in USD per 1 M prompt tokens (OpenAI).
        openai_price_out_1m: Cost in USD per 1 M completion tokens (OpenAI).
        valid_categories: Mapping of category names to their descriptions.
        default_delay: Seconds to sleep between concurrent API calls.
        max_retries: Maximum retry attempts for transient API errors.
        max_workers: Thread pool size for concurrent conversation processing.
    """

    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    gemini_default_model: str = "gemini-3.1-pro-preview"
    gemini_price_in_1m: float = 2.00
    gemini_price_out_1m: float = 12.00

    openai_default_model: str = "gpt-5.1-2025-11-13"
    openai_price_in_1m: float = 1.25
    openai_price_out_1m: float = 10.00

    valid_categories: dict[str, str] = {
        "preferences": "brands, tastes, styles, hobbies, foods, products the user regularly uses",
        "identity": "first name, last name (ONLY if explicitly stated by the user themselves)",
        "household": "children (names/ages), pets (name/breed), type of dwelling, location (city only)",
        "demographics": "gender, marital status, age (only if explicitly mentioned)",
    }
    api_timeout: int = 60
    default_delay: int = 1
    max_retries: int = 3
    max_workers: int = 5

    # ElasticSearch Settings
    es_host: str = "http://localhost:9200"
    es_index: str = "user_facts"

    # Similarity Thresholds
    dup_certainty: float = 0.92
    dup_uncertainty: float = 0.85

    # LiteLLM & External Providers
    ollama_base_url: str = "http://localhost:11434"
    litellm_cache: bool = True
    
    # RAG Settings
    rag_threshold: int = 20
    rag_top_k: int = 5

    model_config = SettingsConfigDict(env_file=str(_PROJECT_ROOT / ".env"), extra="ignore")


# Singleton instance used throughout the package
settings = Settings()

# ---------------------------------------------------------------------------
# PII regex patterns — applied as a second-pass safety net after the LLM
# prompt has already been instructed to omit sensitive data.
# ---------------------------------------------------------------------------
PII_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b\d{11}\b"),                                           # PESEL (Polish national ID)
    re.compile(
        r"\bPL[\s]?\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}\b",
        re.I,
    ),                                                                    # Polish IBAN
    re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"),     # Card numbers
    re.compile(
        r"\b(ul\.|ulica|al\.|aleja|pl\.|plac)\s+\w[\w\s]+\d+[a-z]?\b",
        re.I,
    ),                                                                    # Street addresses
]
