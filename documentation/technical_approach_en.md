# Technical Approach: Personal Memory Module

This document details the architectural decisions and implementation strategies used in the Personal Memory Module.

## Core Philosophy

The system is designed to transform unstructured conversation logs into a structured, permanent knowledge base while maintaining strict privacy standards (PII filtering) and operational scalability.

## 1. Layered Package Architecture

The refactoring move from a single script to a structured package (`src/`) follows enterprise Python patterns:

- **Config Layer**: Centralized management of settings and sensitive regex patterns. This allows for environment-specific configurations (Dev/Staging/Prod) without code changes.
- **Provider Abstraction**: A common interface (`LLMProvider`) allows switching between Google Gemini and OpenAI mid-operation. This "Multi-LLM" strategy prevents vendor lock-in and allows for cost/performance optimization.
- **Pydantic Validation**: All data flowing in and out of the LLMs is strictly typed. Validation happens at the edge, ensuring that no malformed facts or PII-laden strings enter the downstream pipeline.

## 2. Data Safety & PII Filtering

Privacy is implemented as a "Defense in Depth" strategy:

1. **Prompt Engineering**: The primary filter. LLMs are instructed with high-priority rules to ignore specific categories (Bank accounts, Medical data, exact addresses).
2. **Regex Guardrails**: A secondary, deterministic filter. Pre-defined regex patterns (PESEL, IBAN, Credit Cards) process every candidate fact. If a pattern matches, the fact is rejected before reaching the database.
3. **Category Constraints**: Only facts fitting pre-approved categories are accepted. This prevents "context leakage" where the model might extract trivia that complicates the user profile.

## 3. High-Performance Extraction Pipeline

The pipeline is built for high throughput:

- **Concurrency**: Utilizing `ThreadPoolExecutor` to handle multiple conversations in parallel.
- **Resilience**: Integrated retry logic (exponential backoff) via `tenacity` handles transient API errors and rate limits.
- **Telemetry**: Detailed metrics (latency, token usage, estimated cost) are tracked for every operation, allowing for precise ROI analysis of different LLM models.

## 4. Scaling for Production

The architecture anticipates the next phases of growth:

- **ElasticSearch Readiness**: The structure supports moving from local JSON files to a unified vector and key-value store.
- **Kubernetes Compatibility**: Individual modules are stateless and designed to run as horizontally scalable workers in a K8s cluster.
- **Extension Points**: Dedicated stubs for Deduplication and Memory Management allow for rapid development of advanced features without refactoring core logic.
