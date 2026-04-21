"""
src/memory/session_injector.py
=========================================
Adaptive session memory injector.
Handles profile-first grounding and contextual search.
"""

from typing import Any, Dict, List, Set, Optional, Tuple
from loguru import logger
from src.memory.rag_manager import RAGManager, RAGConfig
from src.memory.elasticsearch_store import ESFactStore

class SessionInjector:
    """
    Orchestrates hybrid memory injection for conversations.
    Ensures the LLM is grounded in core user identity initially, 
    then switches to high-precision contextual search.
    """

    def __init__(
        self, 
        store: ESFactStore, 
        user_id: str,
        provider: str = "openai",
        model: str = "gpt-4o",
        already_injected_ids: Optional[Set[str]] = None,
        max_injections: int = 10
    ):
        self.store = store
        self.user_id = user_id
        self.rag_manager = RAGManager(store, provider=provider, model=model)
        self.max_injections = max_injections
        
        # Track state across the session
        self.injected_ids = already_injected_ids or set()
        self.rag_manager.injected_fact_ids = self.injected_ids
        
        self.is_first_message = len(self.injected_ids) == 0

    def _get_key_facts(self) -> List[Dict[str, Any]]:
        """
        Retrieves core profile facts (Identity, Household, Demographics).
        """
        key_categories = ["identity", "household", "demographics"]
        all_key_facts = []
        for cat in key_categories:
            cat_facts = self.store.get_active_facts(self.user_id, category=cat)
            all_key_facts.extend(cat_facts)
        return all_key_facts

    def inject_memory(
        self, 
        user_message: str, 
        rag_config: Optional[RAGConfig] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Determines what context to inject based on session state and message content.
        
        Returns:
            Tuple [prompt_segment (formatted string), retrieved_facts (list of dicts)]
        """
        # 1. Optimization: Skip short/noise messages
        msg_len = len(user_message.split())
        if msg_len < 5:
            logger.info(f"Skipping memory injection for short message: '{user_message}'")
            return "", []

        # 1.5 Cost Control: Enforce injection cap (per memory_utilization_en.md)
        if len(self.injected_ids) >= self.max_injections:
            logger.warning(f"Injection cap reached ({self.max_injections}). Skipping further updates.")
            return "", []

        combined_facts = []

        # 2. First Message Logic: Inject Key Profile Facts
        if self.is_first_message:
            logger.info(f"First message detected for {self.user_id}. Injecting key profile facts.")
            key_facts = self._get_key_facts()
            # Filter out any that might already be tracked
            new_key_facts = [f for f in key_facts if f["id"] not in self.injected_ids]
            combined_facts.extend(new_key_facts)
            self.is_first_message = False

        # 3. Contextual RAG Logic
        # RAGManager handles its own deduplication via injected_fact_ids
        rag_prompt, rag_facts = self.rag_manager.get_context_for_session(
            user_id=self.user_id,
            current_message=user_message,
            config=rag_config
        )
        
        # Merge RAG facts with key facts (ensuring no duplicates in the returned list)
        current_ids = {f["id"] for f in combined_facts}
        for f in rag_facts:
            if f["id"] not in current_ids:
                combined_facts.append(f)
                current_ids.add(f["id"])

        if not combined_facts:
            return "", []

        # 4. Format the final context block
        context_lines = []
        # Update our global state tracking
        for f in combined_facts:
            context_lines.append(f"- {f['fact']}")
            self.injected_ids.add(f["id"])
            
        final_segment = "USER BACKGROUND & CONTEXT:\n" + "\n".join(context_lines)
        
        return final_segment, combined_facts

    def get_injected_state(self) -> Set[str]:
        """Returns the current set of facts IDs injected in this session."""
        return self.injected_ids
