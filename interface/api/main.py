import time
import uuid
from fastapi import FastAPI, HTTPException
from loguru import logger
import os
import redis
from rq import Queue
from rq.job import Job


from src.config.settings import settings
from src.providers.factory import ProviderFactory
from src.memory.elasticsearch_store import ESFactStore
from src.memory.rag_manager import RAGManager, RAGConfig
from src.pipeline.extraction import FactExtractionPipeline
from interface.api.models import ChatRequest, ChatResponse, RAGConfigModel, SettingsResponse, ExtractRequest, ExtractResponse
from tenacity import retry, stop_after_attempt, wait_exponential

from pathlib import Path

app = FastAPI(
    title="Personal Memory RAG API",
    openapi_url="/openapi.json",
    docs_url="/docs"
)

# Global instances
store = ESFactStore()

# RQ initialization
redis_host = os.getenv("REDIS_HOST", "redis-master")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_conn = redis.Redis(host=redis_host, port=redis_port)
q = Queue("extraction", connection=redis_conn)

@retry(stop=stop_after_attempt(15), wait=wait_exponential(multiplier=1, min=2, max=10))
def initialize_store():
    logger.info("Connecting to ElasticSearch and initializing index...")
    store.create_index()

@app.on_event("startup")
async def startup_event():
    try:
        initialize_store()
        logger.info(f"Connected to Redis for Job Queue at {redis_host}:{redis_port}")
    except Exception as e:
        logger.error(f"Failed to connect to required services: {e}")
        # We don't raise here to allow the pod to stay 'Running' but Unhealthy if needed, 
        # or we could raise to let K8s restart it. Given the retry above, if it fails after 15 times, it's a real issue.
        raise e

@app.get("/api")
async def root():
    return {"message": "Personal Memory RAG API is running"}

@app.post("/api/extract", response_model=ExtractResponse)
async def extract_facts(request: ExtractRequest):
    """
    Enqueue fact extraction as a background job.
    """
    try:
        from src.worker import process_extraction
        
        messages_dicts = [m.model_dump() for m in request.messages]
        
        job = q.enqueue(
            process_extraction,
            user_id=request.user_id,
            messages=messages_dicts,
            provider=request.provider,
            model=request.model,
            conversation_id=request.conversation_id,
            job_timeout='5m'
        )
        
        return ExtractResponse(
            job_id=job.get_id(),
            status="queued"
        )
    except Exception as e:
        logger.error(f"Extraction trigger failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/extract/{job_id}", response_model=ExtractResponse)
async def get_extraction_status(job_id: str):
    """
    Poll for the status and results of a background extraction job.
    """
    try:
        job = Job.fetch(job_id, connection=redis_conn)
        
        if job.is_finished:
            res = job.result
            return ExtractResponse(
                job_id=job_id,
                status="finished",
                new_facts=res.get("new_facts"),
                stored_ids=res.get("stored_ids"),
                telemetry=res.get("telemetry")
            )
        elif job.is_failed:
            return ExtractResponse(job_id=job_id, status="failed")
        else:
            return ExtractResponse(job_id=job_id, status="processing")
            
    except Exception as e:
        logger.error(f"Failed to fetch job {job_id}: {e}")
        raise HTTPException(status_code=404, detail="Job not found or error fetching status")

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    
    # 1. Initialize RAG Config
    config_data = request.config or RAGConfigModel()
    rag_config = RAGConfig(
        strategy=config_data.strategy,
        top_k=config_data.top_k,
        threshold=config_data.threshold,
        alpha=config_data.alpha,
        rerank=config_data.rerank,
        rerank_model=config_data.rerank_model
    )
    
    # 2. Get Context via Per-Request RAGManager
    req_rag = RAGManager(store, provider=request.provider, model=request.model)
    context, retrieved_facts = req_rag.get_context_for_session(
        user_id=request.user_id,
        current_message=request.message,
        config=rag_config
    )
    
    # 3. Construct Final Prompt
    system_instruction = (
        "You are a personalized AI assistant. Use the provided user context to tailor your response. "
        "Stay helpful and grounded in the facts. STRICT RULE: DO NOT use any emojis in your response."
    )
    
    full_prompt = f"{context}\n\nUSER MESSAGE: {request.message}" if context else request.message
    
    # 4. Generate LLM Response using LiteLLM
    try:
        import litellm
        
        # Use LiteLLM Proxy if available (Production/K8s mode)
        proxy_url = os.getenv("LITELLM_PROXY_URL")
        if proxy_url:
            litellm.api_base = proxy_url
            litellm.api_key = "sk-1234"
            # In proxy mode, we use the simple model name defined in config.yaml
            litellm_model = request.model
            api_base = None 
        else:
            litellm_model = f"{request.provider}/{request.model}" if request.provider != "ollama" else f"ollama/{request.model}"
            api_base = settings.ollama_base_url if request.provider == "ollama" else None
        
        resp = litellm.completion(
            model=litellm_model,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": full_prompt},
            ],
            temperature=request.temperature,
            top_p=request.top_p,
            api_base=api_base,
            caching=settings.litellm_cache
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    latency = time.time() - start_time
    
    # Prepare retrieval trace for telemetry
    retrieval_trace = []
    for f in retrieved_facts:
        retrieval_trace.append({
            "fact": f["fact"],
            "score": f.get("rrf_score", 0),
            "details": f.get("trace", {})
        })
    
    return ChatResponse(
        response=answer,
        context_used=[line[2:] for line in context.splitlines() if line.startswith("- ")],
        telemetry={
            "latency_sec": round(latency, 3),
            "model": request.model,
            "provider": request.provider,
            "strategy": rag_config.strategy,
            "retrieval_trace": retrieval_trace
        }
    )

@app.post("/api/reset-index")
async def reset_index():
    """
    Deletes the existing user_facts index and recreates it with new mappings.
    USE WITH CAUTION: This will delete all stored memory.
    """
    try:
        if store.client.indices.exists(index=store.index_name):
            store.client.indices.delete(index=store.index_name)
        store.create_index()
        return {"status": "success", "message": "Index successfully reset and recreated with 1536 dims."}
    except Exception as e:
        logger.error(f"Failed to reset index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/facts/{user_id}")
async def get_facts(user_id: str):
    try:
        facts = store.get_active_facts(user_id)
        return {"facts": facts}
    except Exception as e:
        logger.error(f"Failed to fetch facts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/facts/{fact_id}")
async def delete_fact(fact_id: str):
    try:
        store.soft_delete(fact_id)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"Failed to delete fact: {e}")
        raise HTTPException(status_code=500, detail=str(e))
