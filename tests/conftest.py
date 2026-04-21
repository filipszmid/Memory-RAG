import pytest
import os
from unittest.mock import MagicMock

@pytest.fixture
def sample_user_id():
    return "test-user-123"

@pytest.fixture
def sample_messages():
    return [
        {"role": "user", "content": "I have a cat named Luna."},
        {"role": "assistant", "content": "That's a nice name for a cat!"},
        {"role": "user", "content": "I live in Berlin."}
    ]

@pytest.fixture
def sample_facts():
    return [
        {"id": "1", "fact": "User has a cat named Luna", "user_id": "test-user-123"},
        {"id": "2", "fact": "User lives in Berlin", "user_id": "test-user-123"}
    ]

@pytest.fixture
def mock_litellm(monkeypatch):
    mock = MagicMock()
    # Mocking completion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"facts": ["User has a cat named Luna"]}'
    mock.completion.return_value = mock_response
    
    # Mocking embedding response
    mock_embed = MagicMock()
    mock_embed.data = [{"embedding": [0.1] * 1536}]
    mock.embedding.return_value = mock_embed
    
    monkeypatch.setattr("litellm.completion", mock.completion)
    monkeypatch.setattr("litellm.embedding", mock.embedding)
    return mock

@pytest.fixture
def mock_es_client():
    mock = MagicMock()
    # Ensure indices.exists returns something
    mock.indices.exists.return_value = True
    return mock

@pytest.fixture
def env_setup(monkeypatch):
    monkeypatch.setattr(os, "environ", {
        **os.environ,
        "LITELLM_MASTER_KEY": "sk-1234",
        "REDIS_HOST": "localhost",
        "REDIS_PORT": "6379",
        "ES_HOST": "http://localhost:9200"
    })
