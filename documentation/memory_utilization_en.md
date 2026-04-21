# Memory Utilization: Retrieval Mechanisms and Session Optimization

This document outlines the strategy for integrating a persistent knowledge base into an LLM agent's context. The focus is on maintaining low latency, cost-efficiency, and high factual accuracy.

---

## 1. Contextual Feeding Strategies

### Approach A: Full Profile Injection
Inyecting the entire user profile directly into the system prompt.

* **Performance**: Lowest latency (zero additional lookups).
* **Reliability**: Highest (LLM has absolute visibility).
* **Scaling**: Limited to small profiles (< 20-30 facts). Costs increase linearly with context size per turn.

### Approach B: Adaptive RAG (Retrieval-Augmented Generation)
Dynamically fetching the most relevant facts based on the current conversation turn.

* **Efficiency**: Optimized cost; only relevant tokens are processed.
* **Latency**: Moderate (+50-150ms for vector search).
* **Implementation**: Facts are indexed in a vector store (e.g., ElasticSearch with `dense_vector`). Each turn generates an embedding for K-NN retrieval.

### Approach C: Narrative Profile
Using an LLM to pre-compile a descriptive summary of the user (e.g., "A tech-savvy enthusiast living in Berlin").

* **Engagement**: High naturalness for companion AI.
* **Risk**: High hallucination potential. Models may "hallucinate" traits not present in the raw data.

---

## 2. Recommended Hybrid Architecture

For production environments (E-commerce, SaaS Support), a hybrid approach is implemented:

1. **Initial Context**: Key demographic traits (age, preferred language, core filters) are always injected into the primary prompt.
2. **Context-Dependent Retrieval**:
    - Profiles < 20 facts: Full injection for maximum precision.
    - Profiles > 20 facts: RAG-based injection for cost/token efficiency.
3. **Unified Storage**: Utilizing **ElasticSearch** as a unified engine for both Key-Value retrieval (via `term query` on `user_id`) and Vector Search (via `knn search`).

---

## 3. Dynamic Memory in Multi-turn Dialogues

Memory injection is treated as a reactive process, not a one-time setup:

1. **Turn-based Vectorization**: Each user message is vectorized in the background.
2. **Exclusion Logic**: The system maintains an `already_injected_fact_ids` list to prevent redundant context padding within the same session.
3. **System Interjections**: Retrieved facts are inserted as system "reminders" immediately preceding the user's latest message, rather than rewriting the static system prompt.

### Cost Control Mechanisms

* **Message Filtering**: Skip retrieval for short, low-signal inputs (e.g., "Ok", "Yes", "Thanks").
* **Injection Caps**: Enforce a maximum number of new fact injections per session (e.g., limit to 10 context updates).
