from src.providers.base import LLMProvider
from src.providers.litellm_provider import LiteLLMProvider


class ProviderFactory:
    """
    Factory for instantiating LLM providers.
    """

    @staticmethod
    def get_provider(provider_name: str, model: str, timeout: int = 60) -> LLMProvider:
        """
        Instantiates and returns the unified LiteLLMProvider.
        """
        provider_name = provider_name.lower()
        return LiteLLMProvider(model, provider=provider_name, timeout=timeout)
