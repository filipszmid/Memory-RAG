import pytest
from unittest.mock import MagicMock, patch
from src.deduplication.dedup import DeduplicationEngine

@pytest.fixture
def mock_store():
    return MagicMock()

def test_dedup_get_embedding(mock_store, mock_litellm):
    engine = DeduplicationEngine(mock_store)
    vector = engine._get_embedding("I have a cat")
    assert len(vector) == 1536
    assert vector[0] == 0.1
    mock_litellm.embedding.assert_called_once()

@patch("src.deduplication.dedup.datetime")
def test_process_new_fact_no_match(mock_dt, mock_store, mock_litellm):
    mock_dt.datetime.now.return_value.isoformat.return_value = "2024-01-01"
    mock_store.knn_search.return_value = []
    mock_store.save_fact.return_value = "new-id"
    
    engine = DeduplicationEngine(mock_store)
    fact = {"fact": "New unique fact", "category": "personal"}
    
    res_id = engine.process_new_fact("user-1", fact, "conv-1")
    
    assert res_id == "new-id"
    mock_store.save_fact.assert_called_once()

def test_process_new_fact_duplicate_match(mock_store, mock_litellm):
    # Mock return values for deduplication
    mock_store.knn_search.return_value = [
        {"id": "old-id", "fact": "Existing fact", "score": 0.99}
    ]
    
    engine = DeduplicationEngine(mock_store)
    # Settings default dup_certainty is 0.92
    fact = {"fact": "Existing fact", "category": "personal"}
    
    res_id = engine.process_new_fact("user-1", fact, "conv-1")
    
    assert res_id == "old-id"
    mock_store.update_fact.assert_called_once()
