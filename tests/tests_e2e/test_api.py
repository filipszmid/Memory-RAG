import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from interface.api.main import app

client = TestClient(app)

@pytest.fixture
def mock_dependencies():
    with patch("interface.api.main.q") as mock_q, \
         patch("interface.api.main.redis_conn") as mock_redis, \
         patch("interface.api.main.store") as mock_store, \
         patch("litellm.completion") as mock_completion, \
         patch("litellm.embedding") as mock_embedding:
         
        # Setup mock ES
        mock_store.get_active_facts.return_value = []
        
        # Setup mock LiteLLM
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Mocked AI Response"
        mock_completion.return_value = mock_resp
        
        mock_embed = {"data": [{"embedding": [0.1] * 1536}]}
        mock_embedding.return_value = mock_embed
        
        yield {
            "q": mock_q,
            "redis": mock_redis,
            "store": mock_store,
            "litellm": mock_completion
        }

def test_api_root(mock_dependencies):
    response = client.get("/api")
    assert response.status_code == 200
    assert response.json() == {"message": "Personal Memory RAG API is running"}

def test_api_extract_endpoint(mock_dependencies, sample_user_id, sample_messages):
    payload = {
        "user_id": sample_user_id,
        "messages": sample_messages,
        "provider": "openai"
    }
    
    # Setup mock job ID
    mock_job = MagicMock()
    mock_job.get_id.return_value = "job-123"
    mock_dependencies["q"].enqueue.return_value = mock_job
    
    response = client.post("/api/extract", json=payload)
    
    assert response.status_code == 200
    assert response.json()["status"] == "queued"
    assert response.json()["job_id"] == "job-123"
    mock_dependencies["q"].enqueue.assert_called_once()

def test_api_chat_endpoint(mock_dependencies, sample_user_id):
    payload = {
        "user_id": sample_user_id,
        "message": "Hello memory engine",
        "provider": "openai",
        "model": "gpt-4o"
    }
    
    response = client.post("/api/chat", json=payload)
    
    assert response.status_code == 200
    assert "response" in response.json()
    assert response.json()["response"] == "Mocked AI Response"
    assert "telemetry" in response.json()
