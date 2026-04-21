from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class RAGConfigModel(BaseModel):
    strategy: str = "hybrid"
    top_k: int = 5
    threshold: int = 20
    alpha: float = 0.5
    rerank: bool = False
    rerank_model: str = "gpt-4o-mini"

class ChatRequest(BaseModel):
    user_id: str
    message: str
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.7
    top_p: float = 1.0
    config: Optional[RAGConfigModel] = None

class ChatResponse(BaseModel):
    response: str
    context_used: List[str]
    telemetry: Dict[str, Any]

class Fact(BaseModel):
    id: str
    fact: str
    category: str
    confidence: float
    created_at: str

class SettingsResponse(BaseModel):
    rag_config: RAGConfigModel

class MessageBody(BaseModel):
    role: str
    content: str

class ExtractRequest(BaseModel):
    user_id: str
    messages: List[MessageBody]
    provider: str = "openai"
    model: str = "gpt-4o"
    conversation_id: Optional[str] = None

class ExtractResponse(BaseModel):
    job_id: str
    status: str
    new_facts: Optional[List[Any]] = None
    stored_ids: Optional[List[str]] = None
    telemetry: Optional[Dict[str, Any]] = None
