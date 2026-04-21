"""
Personal Memory Module
======================

A production-ready, horizontally-scalable pipeline for extracting,
deduplicating, and serving permanent user facts from LLM conversations.

Package layout
--------------
src/
├── config/         — Settings, PII patterns
├── core/           — Pydantic models, system prompt
├── providers/      — LLM provider abstraction (Gemini, OpenAI)
├── pipeline/       — Concurrent extraction orchestration
├── deduplication/  — [Task 2] Semantic dedup & conflict resolution stubs
├── memory/         — [Task 3] ElasticSearch RAG & session injection stubs
└── cli.py          — Click CLI entry point
"""

__version__ = "0.1.0"
