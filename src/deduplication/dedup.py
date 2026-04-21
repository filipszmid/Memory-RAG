import litellm
from typing import Any, Dict, List, Optional, Tuple
import datetime

from loguru import logger
from src.config.settings import settings
from src.memory.elasticsearch_store import ESFactStore


class DeduplicationEngine:
    """
    Handles semantic deduplication and conflict resolution for extracted facts via LiteLLM.
    """

    def __init__(self, store: ESFactStore, provider: str = "openai", model: str = "text-embedding-3-small"):
        self.store = store
        self.provider = provider
        self.model = model
        
        # Mapping for LiteLLM embedding models
        if provider == "openai":
            self.embedding_model = "openai/text-embedding-3-small"
        elif provider == "gemini":
            self.embedding_model = "gemini/text-embedding-004"
        elif provider == "ollama":
            self.embedding_model = "ollama/mxbai-embed-large" # Common Ollama embedding model
        else:
            self.embedding_model = model

    def _get_embedding(self, text: str) -> List[float]:
        """
        Generates embedding for a given text using LiteLLM.
        Pads to 1536 dimensions for ES compatibility.
        """
        try:
            resp = litellm.embedding(
                model=self.embedding_model,
                input=[text],
                api_base=settings.ollama_base_url if self.provider == "ollama" else None
            )
            embedding = resp.data[0]["embedding"]
            
            # Pad with zeros to 1536 if needed
            if len(embedding) < 1536:
                embedding.extend([0.0] * (1536 - len(embedding)))
            elif len(embedding) > 1536:
                embedding = embedding[:1536]
                
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

    def _verify_with_llm(self, fact_a: str, fact_b: str) -> str:
        """
        Uses LiteLLM to verify relationship between facts.
        Returns: 'DUPLICATE', 'CONTRADICT', or 'NONE'
        """
        prompt = (
            "You are a memory deduplication assistant. compare two user facts:\n"
            f"Old Fact: {fact_a}\nNew Fact: {fact_b}\n\n"
            "Respond ONLY with one of these three labels:\n"
            "- 'DUPLICATE': The facts convey the same information.\n"
            "- 'CONTRADICT': The newer fact invalidates, updates, or contradicts the old one (e.g., 'lives in Paris' vs 'moved to Berlin', or 'has a cat' vs 'cat passed away').\n"
            "- 'NONE': The facts are unrelated.\n"
        )
        try:
            verify_model = "gpt-4o-mini" if self.provider == "openai" else "gemini-1.5-flash"
            if self.provider == "ollama":
                 verify_model = "ollama/llama3"
            
            resp = litellm.completion(
                model=verify_model,
                messages=[{"role": "user", "content": prompt}],
                api_base=settings.ollama_base_url if self.provider == "ollama" else None,
                caching=settings.litellm_cache
            )
            content = resp.choices[0].message.content.upper()
            if "DUPLICATE" in content: return "DUPLICATE"
            if "CONTRADICT" in content: return "CONTRADICT"
            return "NONE"
        except Exception as e:
            logger.error(f"LLM verification failed: {e}")
            return False

    def process_new_fact(self, user_id: str, fact: Dict[str, Any], conversation_id: str) -> str:
        """
        Processes a single extracted fact with persistence to ES.
        """
        fact_text = fact["fact"]
        category = fact["category"]
        fact_key = fact.get("fact_key")
        
        embedding = self._get_embedding(fact_text)
        fact["embedding"] = embedding
        fact["source_conversation_id"] = conversation_id
        fact["created_at"] = datetime.datetime.now().isoformat()

        # 1. Conflict Resolution (Fact Key)
        if fact_key:
            existing_key_fact = self.store.get_fact_by_key(user_id, fact_key)
            if existing_key_fact:
                logger.info(f"Conflict detected for key '{fact_key}'. Superseding.")
                new_id = self.store.save_fact(user_id, fact)
                self.store.soft_delete(existing_key_fact["id"], superseded_by=new_id)
                return new_id

        # 2. Semantic Deduplication
        if not embedding:
            return self.store.save_fact(user_id, fact)

        matches = self.store.knn_search(user_id, embedding, category=category, top_k=1)
        if matches:
            match = matches[0]
            score = match["score"]
            if score >= settings.dup_certainty:
                self.store.update_fact(match["id"], {
                    "last_confirmed_at": datetime.datetime.now().isoformat(),
                    "confirmation_count": match.get("confirmation_count", 1) + 1
                })
                return match["id"]

            if score >= settings.dup_uncertainty:
                relationship = self._verify_with_llm(match["fact"], fact_text)
                if relationship == "DUPLICATE":
                    self.store.update_fact(match["id"], {
                        "last_confirmed_at": datetime.datetime.now().isoformat(),
                        "confirmation_count": match.get("confirmation_count", 1) + 1
                    })
                    return match["id"]
                elif relationship == "CONTRADICT":
                    logger.info(f"Semantic contradiction detected. Superseding '{match['fact']}' with '{fact_text}'")
                    new_id = self.store.save_fact(user_id, fact)
                    self.store.soft_delete(match["id"], superseded_by=new_id)
                    return new_id

        # 3. New Fact
        logger.info(f"Saving new fact: {fact_text}")
        return self.store.save_fact(user_id, fact)

import datetime
