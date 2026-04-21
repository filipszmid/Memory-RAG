# Data Consistency: Deduplication and Conflict Resolution

As the knowledge base grows across multiple sessions, maintaining high-quality data requires addressing two fundamental challenges:
* **Redundancy** — identifying the same fact described in different words.
* **Knowledge Decay** — managing information that has been superseded by newer events.

The following architecture provides a production-grade mechanism for managing these challenges.

---

## 1. Fact Deduplication

To avoid redundant storage and processing (e.g., "I own a dog" vs "I have a canine pet"), the system employs a multi-stage deduplication strategy.

### Semantic Similarity (Embeddings)

1. **Category Clustering**: Comparisons are restricted to facts within the same category (e.g., `household`) to minimize computational overhead.
2. **Vectorization**: New candidate facts are converted into high-dimensional embeddings (e.g., `text-embedding-004`).
3. **Similarity Assessment**:
   - **High Certainty (≥ 0.92)**: Automatically treated as a duplicate. The system updates the existing entry's `last_confirmed_at` timestamp and increments the `confirmation_count`.
   - **Ambiguity Zone (0.89 – 0.92)**: Triggers a high-reasoning LLM verification step with a targeted prompt: *"Do Sentence A and Sentence B represent a semantically identical user trait?"*

### Efficiency Optimizations

For large-scale deployments, a two-step pre-filter is recommended:
1. **Structural Hash**: Fast O(1) detection for identical normalized strings.
2. **Textual Overlap**: Jaccard similarity or token set comparison to prune candidate pairs before expensive vector calculations.

---

## 2. Conflict Resolution

User preferences and life circumstances are dynamic. The system must recognize when new information invalidates historical records.

### Fact Key Mapping

Facts identified as "transiently unique" (e.g., `marital_status`, `working_location`) are assigned a **Functional Key (`fact_key`)**. When a new fact targeting an existing key is extracted:

1. **Chronological Authority**: The most recent extraction is prioritized as the "Active" state.
2. **Soft Deletion & Lineage**: Historical records remain in the database but are flagged with `is_outdated = True`.
3. **Supersession Chain**: Outdated records include a `superseded_by` pointer to the new record ID, preserving a historical audit trail.

This approach allows the agent to maintain context over time (e.g., "I recall you used to live in London, but now you've settled in Munich").

---

## 3. Operational Considerations

* **Intra-message Logic**: The extraction pipeline must prioritize the user's final stated intent within a single turn (e.g., "I'm looking for a car... actually, I've decided to wait").
* **Confidence Decay**: Facts that haven't been re-confirmed over a long period (e.g., 2 years) should undergo a confidence score reduction.
* **Narrative Perspective**: Distinguishing between individual traits and household traits (e.g., "We have a cat") to prevent incorrect singular attribution.
* **Intent vs. Fact**: Strict filtering of hypothetical statements (e.g., "I wish I had a car") is handled at the prompt engineering level.

---

## 4. Extended Metadata Schema

| Field | Description |
|---|---|
| `fact_key` | Identifier for mutually exclusive traits. |
| `confidence` | LLM-assigned score (0.0 - 1.0) for extraction reliability. |
| `last_confirmed_at` | Timestamp of the most recent duplicate detection. |
| `is_outdated` | Boolean flag for superseded knowledge. |
| `source_conversation_id` | Provenance link back to the raw conversation log. |
