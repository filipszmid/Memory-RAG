import pytest
from pydantic import ValidationError
from interface.api.models import ExtractRequest, ChatRequest, MessageBody
from src.core.models import Fact, FactsResponse

def test_valid_fact():
    fact = Fact(fact="User has a Toyota Yaris", category="preferences", confidence=0.9, fact_key="car", created_at="2024-01-01")
    assert fact.fact == "User has a Toyota Yaris"
    assert fact.category == "preferences"
    assert fact.fact_key == "car"
    assert fact.confidence == 0.9

def test_invalid_category():
    with pytest.raises(ValidationError) as excinfo:
        Fact(fact="Valid fact", category="non_existent_category", confidence=1.0, created_at="2024-01-01")
    assert "Category 'non_existent_category' is not valid" in str(excinfo.value)

def test_empty_fact():
    with pytest.raises(ValidationError) as excinfo:
        Fact(fact="", category="preferences", confidence=1.0, created_at="2024-01-01")
    assert "Fact cannot be empty" in str(excinfo.value)

def test_facts_response_model():
    data = {
        "facts": [
            {"fact": "Fact 1", "category": "preferences", "confidence": 1.0, "created_at": "2024-01-01"},
            {"fact": "Fact 2", "category": "identity", "fact_key": "name", "created_at": "2024-01-01"}
        ]
    }
    response = FactsResponse(**data)
    assert len(response.facts) == 2
    assert response.facts[0].fact == "Fact 1"
    assert response.facts[1].category == "identity"

def test_message_model_valid():
    msg = MessageBody(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"

def test_message_model_invalid_role():
    with pytest.raises(ValidationError):
        MessageBody(role=None, content="Hello")

def test_extract_request_valid(sample_user_id, sample_messages):
    req = ExtractRequest(
        user_id=sample_user_id,
        messages=sample_messages,
        provider="openai"
    )
    assert req.user_id == sample_user_id
    assert len(req.messages) == 3

def test_chat_request_defaults(sample_user_id):
    req = ChatRequest(
        user_id=sample_user_id,
        message="What is my cat's name?"
    )
    assert req.provider == "openai" # Default value
    assert req.temperature == 0.7 # Default value
