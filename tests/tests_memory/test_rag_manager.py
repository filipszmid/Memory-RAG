import pytest
from unittest.mock import MagicMock
from src.memory.rag_manager import RAGManager, RAGConfig

@pytest.fixture
def mock_store():
    return MagicMock()

def test_rag_manager_get_context_short_message(mock_store, mock_litellm):
    manager = RAGManager(mock_store)
    context, facts = manager.get_context_for_session("user-1", "Hi")
    assert context == ""
    assert facts == []

def test_rag_manager_get_context_below_threshold(mock_store, mock_litellm):
    # If total facts are below threshold (default 20), it should return all active facts
    mock_store.get_active_facts.return_value = [
        {"id": "f1", "fact": "User is a coder"},
        {"id": "f2", "fact": "User lives in space"}
    ]
    
    manager = RAGManager(mock_store)
    context, facts = manager.get_context_for_session("user-1", "What do I do for a living?")
    
    assert "User is a coder" in context
    assert "User lives in space" in context
    assert len(facts) == 2

def test_rag_manager_get_context_vector_strategy(mock_store, mock_litellm):
    # Force use of vector strategy by providing more than 20 facts 
    # Or by passing a config
    config = RAGConfig(strategy="vector", threshold=1)
    
    mock_store.get_active_facts.return_value = [
        {"id": "f1", "fact": "Fact 1"},
        {"id": "f2", "fact": "Fact 2"}
    ]
    mock_store.knn_search.return_value = [
        {"id": "f1", "fact": "Fact 1", "score": 0.9}
    ]
    
    manager = RAGManager(mock_store)
    context, facts = manager.get_context_for_session("user-1", "Checking fact 1 relevance", config=config)
    
    assert "Fact 1" in context
    assert "Fact 2" not in context
    mock_store.knn_search.assert_called_once()
    mock_litellm.embedding.assert_called_once()
