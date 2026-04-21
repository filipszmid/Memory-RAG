import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.pipeline.extraction import FactExtractionPipeline

@pytest.fixture
def mock_pipeline_deps():
    with patch("src.pipeline.extraction.ProviderFactory.get_provider") as mock_factory, \
         patch("src.pipeline.extraction.ESFactStore") as mock_store, \
         patch("src.pipeline.extraction.DeduplicationEngine") as mock_dedup:
        
        mock_provider = MagicMock()
        mock_factory.return_value = mock_provider
        
        yield {
            "provider": mock_provider,
            "store": mock_store.return_value,
            "dedup": mock_dedup.return_value
        }

def test_pipeline_initialization(tmp_path, mock_pipeline_deps):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    
    pipeline = FactExtractionPipeline(input_dir, output_dir, "openai", "gpt-4o")
    
    assert pipeline.provider_name == "openai"
    assert pipeline.model == "gpt-4o"
    assert pipeline.input_dir == input_dir

def test_extract_single_conversation_success(tmp_path, mock_pipeline_deps):
    # Setup mock data
    mock_facts_resp = MagicMock()
    mock_facts_resp.model_dump.return_value = {
        "facts": [{"fact": "Test fact", "category": "test"}]
    }
    mock_pipeline_deps["provider"].generate_facts.return_value = (mock_facts_resp, {"latency": 0.5})
    
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    pipeline = FactExtractionPipeline(input_dir, output_dir, "openai", "gpt-4o")
    
    conversation = {
        "conversation_id": "conv-1",
        "user_id": "user-1",
        "messages": [{"role": "user", "content": "Hello"}]
    }
    
    result = pipeline._extract_single_conversation(conversation)
    
    assert result["conversation_id"] == "conv-1"
    assert len(result["facts"]) == 1
    assert result["telemetry"]["success"] is True
    mock_pipeline_deps["dedup"].process_new_fact.assert_called_once()

def test_load_conversation_valid(tmp_path, mock_pipeline_deps):
    conv_file = tmp_path / "test.json"
    data = {"messages": [{"role": "user", "content": "test"}], "conversation_id": "c1"}
    conv_file.write_text(json.dumps(data))
    
    pipeline = FactExtractionPipeline(tmp_path, tmp_path, "openai", "gpt-4o")
    loaded = pipeline._load_conversation(conv_file)
    
    assert loaded["conversation_id"] == "c1"
    assert len(loaded["messages"]) == 1
