import pytest
from unittest.mock import MagicMock, patch
from src.memory.session_injector import SessionInjector

@pytest.fixture
def mock_store():
    return MagicMock()

def test_session_injector_skips_short_message(mock_store):
    injector = SessionInjector(mock_store, "user-123")
    segment, facts = injector.inject_memory("Hi")
    assert segment == ""
    assert facts == []
    mock_store.get_active_facts.assert_not_called()

def test_session_injector_first_message_priority(mock_store):
    # Setup mock facts for first message
    mock_store.get_active_facts.side_effect = [
        [{"id": "id1", "fact": "User is John"}], # identity
        [{"id": "h1", "fact": "User has a dog"}], # household
        [] # demographics
    ]
    
    # RAGManager starts with an empty search result to isolate "First Message" logic
    with patch("src.memory.rag_manager.RAGManager.get_context_for_session") as mock_rag:
        mock_rag.return_value = ("", [])
        
        injector = SessionInjector(mock_store, "user-123")
        segment, facts = injector.inject_memory("Hello, I need some help with my account.")
        
        assert "User is John" in segment
        assert "User has a dog" in segment
        assert len(facts) == 2
        assert injector.is_first_message is False
        
        # Verify subsequent call doesn't pull profile facts again
        mock_store.reset_mock()
        injector.inject_memory("And also my dog is hungry")
        mock_store.get_active_facts.assert_not_called()

def test_session_injector_state_persistence(mock_store):
    injector = SessionInjector(mock_store, "user-123", already_injected_ids={"id1"})
    
    # Since we passed id1, it shouldn't be the first message anymore 
    # (or it should treat it as already grounded)
    assert injector.is_first_message is False
    
    mock_store.get_active_facts.return_value = [{"id": "id1", "fact": "Already known"}]
    
    with patch("src.memory.rag_manager.RAGManager.get_context_for_session") as mock_rag:
        mock_rag.return_value = ("", [])
        segment, facts = injector.inject_memory("Some long enough message to trigger RAG")
        
        assert "Already known" not in segment
        assert facts == []
