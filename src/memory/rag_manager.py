import litellm
import json
import os
from pydantic import BaseModel
from typing import Any, Dict, List, Set, Optional, Tuple

from loguru import logger
from src.config.settings import settings
from src.memory.elasticsearch_store import ESFactStore


class RAGConfig(BaseModel):
    """
    Dynamic configuration for the RAG process.
    """
    strategy: str = "hybrid"  # "vector", "bm25", "hybrid"
    top_k: int = 5
    threshold: int = 20
    alpha: float = 0.5  # New: Weighted balance between vector and keyword
    rerank: bool = False
    rerank_model: str = "gpt-4o-mini" # Default rerank model


class RAGManager:
    """
    Manages how facts are retrieved and injected into the conversation context via LiteLLM.
    """

    def __init__(self, store: ESFactStore, provider: str = "openai", model: str = "gpt-4o"):
        self.store = store
        self.provider = provider
        self.model = model
        self.injected_fact_ids: Set[str] = set()
        
        # Mapping for LiteLLM embedding models
        if provider == "openai":
            self.embedding_model = "openai/text-embedding-3-small"
        elif provider == "gemini":
            self.embedding_model = "gemini/text-embedding-004"
        elif provider == "ollama":
            self.embedding_model = "ollama/mxbai-embed-large"
        else:
            self.embedding_model = "text-embedding-3-small"

    def _get_embedding(self, text: str) -> List[float]:
        """
        Pads embedding to 1536 dimensions for ES compatibility.
        """
        try:
            # Use LiteLLM Proxy if available
            proxy_url = os.getenv("LITELLM_PROXY_URL")
            a_base = proxy_url if proxy_url else (settings.ollama_base_url if self.provider == "ollama" else None)
            a_key = "sk-1234" if proxy_url else None

            resp = litellm.embedding(
                model=self.embedding_model,
                input=[text],
                api_base=a_base,
                api_key=a_key
            )
            embedding = resp.data[0]["embedding"]
            
            # Pad with zeros to 1536 if needed
            if len(embedding) < 1536:
                embedding.extend([0.0] * (1536 - len(embedding)))
            elif len(embedding) > 1536:
                embedding = embedding[:1536]
                
            return embedding
        except Exception as e:
            logger.error(f"Embedding failed for RAG query: {e}")
            return []

    def _rerank_facts(self, query: str, facts: List[Dict[str, Any]], config: RAGConfig) -> List[Dict[str, Any]]:
        """
        Reranks facts using LiteLLM for premium relevance.
        """
        if not facts or not config.rerank:
            return facts

        logger.info(f"Reranking {len(facts)} facts using {config.rerank_model}")
        
        facts_text = "\n".join([f"ID: {f['id']} | Fact: {f['fact']}" for f in facts])
        prompt = (
            f"Role: Sophisticated Reranker. User Message: '{query}'\n\n"
            f"Facts:\n{facts_text}\n\n"
            "Respond ONLY with a JSON list of IDs in order of relevance. Example: ['id1', 'id2']"
        )
        
        try:
            # Use LiteLLM Proxy if available
            proxy_url = os.getenv("LITELLM_PROXY_URL")
            if proxy_url:
                a_base = proxy_url
                a_key = "sk-1234"
                model_to_use = config.rerank_model
            else:
                a_base = settings.ollama_base_url if self.provider == "ollama" else None
                a_key = None
                model_to_use = config.rerank_model
                if self.provider == "ollama" and not model_to_use.startswith("ollama/"):
                    model_to_use = f"ollama/{model_to_use}"

            resp = litellm.completion(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                api_base=a_base,
                api_key=a_key,
                caching=settings.litellm_cache
            )
            
            text = resp.choices[0].message.content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            
            ordered_ids = json.loads(text)
            facts_map = {f["id"]: f for f in facts}
            reranked = [facts_map[fid] for fid in ordered_ids if fid in facts_map]
            missed = [f for f in facts if f["id"] not in ordered_ids]
            return reranked + missed
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return facts

    def get_context_for_session(
        self, 
        user_id: str, 
        current_message: str, 
        config: Optional[RAGConfig] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Determines context injection logic. Returns (prompt_segment, retrieved_facts).
        """
        if config is None:
            config = RAGConfig()

        if len(current_message.split()) < 3:
            return "", []

        all_active = self.store.get_active_facts(user_id)
        
        if len(all_active) <= config.threshold:
            useful_facts = all_active
        else:
            if config.strategy == "vector":
                query_vector = self._get_embedding(current_message)
                useful_facts = self.store.knn_search(user_id, query_vector, top_k=config.top_k) if query_vector else []
            elif config.strategy == "bm25":
                useful_facts = self.store.text_search(user_id, current_message, top_k=config.top_k)
            else: # hybrid
                query_vector = self._get_embedding(current_message)
                if query_vector:
                    useful_facts = self.store.hybrid_search(
                        user_id, current_message, query_vector, 
                        top_k=config.top_k, alpha=config.alpha
                    )
                else:
                    useful_facts = self.store.text_search(user_id, current_message, top_k=config.top_k)

        useful_facts = self._rerank_facts(current_message, useful_facts, config)
        new_facts = [f for f in useful_facts if f["id"] not in self.injected_fact_ids]
        
        if not new_facts:
            return "", []

        context_lines = []
        for f in new_facts:
            context_lines.append(f"- {f['fact']}")
            self.injected_fact_ids.add(f["id"])

        prompt_segment = "RELEVANT USER CONTEXT:\n" + "\n".join(context_lines)
        return prompt_segment, useful_facts
