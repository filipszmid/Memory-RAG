import pytest
from src.providers.factory import ProviderFactory
from src.providers.litellm_provider import LiteLLMProvider

def test_provider_factory_returns_litellm():
    provider = ProviderFactory.get_provider("openai", "gpt-4o")
    assert isinstance(provider, LiteLLMProvider)
    assert provider.provider == "openai"
    assert provider.model == "gpt-4o"

def test_provider_factory_case_insensitivity():
    provider = ProviderFactory.get_provider("GEMINI", "gemini-1.5-pro")
    assert provider.provider == "gemini"
