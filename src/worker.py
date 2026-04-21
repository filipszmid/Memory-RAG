import os
import redis
from rq import Worker, Queue, Connection
from loguru import logger
from pathlib import Path
from tenacity import retry, stop_after_attempt, wait_exponential

# Important: ensure src is available in path
import sys
sys.path.append(os.getcwd())

from src.pipeline.extraction import FactExtractionPipeline
from src.config.settings import settings

# --- Job definitions ---
def process_extraction(user_id: str, messages: list, provider: str, model: str, conversation_id: str = None):
    """
    Background job function to run fact extraction.
    """
    logger.info(f"Starting background extraction for user {user_id}")
    try:
        pipeline = FactExtractionPipeline(
            input_dir=Path("data/raw"),
            output_dir=Path("outputs"),
            provider_name=provider,
            model=model
        )
        
        result = pipeline.extract_facts_from_messages(
            user_id=user_id,
            messages=messages,
            conversation_id=conversation_id
        )
        
        logger.info(f"Finished background extraction for user {user_id}. Facts found: {len(result.get('facts', []))}")
        return {
            "status": "success",
            "new_facts": result.get("facts", []),
            "stored_ids": result.get("stored_fact_ids", []),
            "telemetry": result.get("telemetry", {})
        }
    except Exception as e:
        logger.error(f"Worker extraction failed: {e}")
        return {"status": "error", "message": str(e)}

# --- Worker entry point ---
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=2, max=10))
def run_worker(redis_host, redis_port):
    conn = redis.Redis(host=redis_host, port=redis_port)
    with Connection(conn):
        logger.info(f"Worker connecting to Redis at {redis_host}:{redis_port}")
        worker = Worker(['extraction'])
        worker.work()

if __name__ == '__main__':
    redis_host = os.getenv("REDIS_HOST", "redis-master")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    
    try:
        run_worker(redis_host, redis_port)
    except Exception as e:
        logger.error(f"Worker failed to start after retries: {e}")
        exit(1)
