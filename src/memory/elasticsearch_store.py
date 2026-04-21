"""
src/memory/elasticsearch_store.py
===========================================
ElasticSearch implementation for persistent fact storage.
Supports vector search (K-NN) and keyword-based retrieval.
"""

import datetime
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch, NotFoundError
from loguru import logger

from src.config.settings import settings


class ESFactStore:
    """
    ElasticSearch-backed store for user facts.
    """

    def __init__(self, host: Optional[str] = None, index_name: Optional[str] = None):
        self.host = host or settings.es_host
        self.index_name = index_name or settings.es_index
        self.client = Elasticsearch([self.host])

    def create_index(self):
        """
        Creates the index with specialized mapping for vectors and metadata.
        """
        mapping = {
            "mappings": {
                "properties": {
                    "user_id": {"type": "keyword"},
                    "fact": {"type": "text"},
                    "category": {"type": "keyword"},
                    "fact_key": {"type": "keyword"},
                    "confidence": {"type": "float"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 1536,  # Standardize on OpenAI size, pad smaller ones
                        "index": True,
                        "similarity": "cosine",
                    },
                    "last_confirmed_at": {"type": "date"},
                    "confirmation_count": {"type": "integer"},
                    "is_outdated": {"type": "boolean"},
                    "superseded_by": {"type": "keyword"},
                    "source_conversation_id": {"type": "keyword"},
                    "created_at": {"type": "date"},
                }
            }
        }
        if not self.client.indices.exists(index=self.index_name):
            logger.info(f"Creating index: {self.index_name}")
            self.client.indices.create(index=self.index_name, body=mapping)
        else:
            logger.info(f"Index {self.index_name} already exists.")

    def save_fact(self, user_id: str, fact_data: Dict[str, Any]) -> str:
        """
        Saves or updates a fact in ElasticSearch.
        """
        doc = {
            "user_id": user_id,
            "created_at": datetime.datetime.now().isoformat(),
            "last_confirmed_at": datetime.datetime.now().isoformat(),
            "confirmation_count": 1,
            "is_outdated": False,
            **fact_data,
        }
        resp = self.client.index(index=self.index_name, document=doc, refresh="wait_for")
        return resp["_id"]

    def update_fact(self, fact_id: str, updates: Dict[str, Any]):
        """
        Updates specific fields of an existing fact.
        """
        self.client.update(index=self.index_name, id=fact_id, doc=updates, refresh="wait_for")

    def soft_delete(self, fact_id: str, superseded_by: Optional[str] = None):
        """
        Marks a fact as outdated.
        """
        updates = {"is_outdated": True}
        if superseded_by:
            updates["superseded_by"] = superseded_by
        self.update_fact(fact_id, updates)

    def get_active_facts(self, user_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieves all non-outdated facts for a specific user.
        """
        query = {
            "bool": {
                "must": [
                    {"term": {"user_id": user_id}},
                    {"term": {"is_outdated": False}}
                ]
            }
        }
        if category:
            query["bool"]["must"].append({"term": {"category": category}})

        try:
            resp = self.client.search(index=self.index_name, query=query, size=1000)
            return [{"id": hit["_id"], **hit["_source"]} for hit in resp["hits"]["hits"]]
        except NotFoundError:
            return []

    def knn_search(self, user_id: str, vector: List[float], category: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search.
        """
        knn = {
            "field": "embedding",
            "query_vector": vector,
            "k": top_k,
            "num_candidates": 100,
            "filter": {
                "bool": {
                    "must": [
                        {"term": {"user_id": user_id}},
                        {"term": {"is_outdated": False}}
                    ]
                }
            }
        }
        if category:
            knn["filter"]["bool"]["must"].append({"term": {"category": category}})

        resp = self.client.search(index=self.index_name, knn=knn, size=top_k)
        return [{"id": hit["_id"], "score": hit["_score"], **hit["_source"]} for hit in resp["hits"]["hits"]]

    def text_search(self, user_id: str, query_text: str, category: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs BM25 keyword search.
        """
        query = {
            "bool": {
                "must": [
                    {"term": {"user_id": user_id}},
                    {"term": {"is_outdated": False}},
                    {"match": {"fact": query_text}}
                ]
            }
        }
        if category:
            query["bool"]["must"].append({"term": {"category": category}})

        resp = self.client.search(index=self.index_name, query=query, size=top_k)
        return [{"id": hit["_id"], "score": hit["_score"], **hit["_source"]} for hit in resp["hits"]["hits"]]

    def hybrid_search(self, user_id: str, query_text: str, vector: List[float], category: Optional[str] = None, top_k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """
        Combines KNN and BM25 search results using weighted reciprocal rank fusion (RRF).
        Alpha controls the balance: 1.0 = Vector only, 0.0 = Keyword only.
        """
        knn_results = self.knn_search(user_id, vector, category, top_k=top_k * 3)
        text_results = self.text_search(user_id, query_text, category, top_k=top_k * 3)

        # Weighted RRF implementation
        rrf_scores = {}
        
        # We also want to keep track of the original scores/ranks for tracing
        trace_data = {}

        for rank, res in enumerate(knn_results):
            fact_id = res["id"]
            score = (1.0 / (rank + 60)) * alpha
            rrf_scores[fact_id] = rrf_scores.get(fact_id, 0) + score
            trace_data.setdefault(fact_id, {})["vector_score"] = round(res["score"], 4)
            trace_data[fact_id]["vector_rank"] = rank + 1
            
        for rank, res in enumerate(text_results):
            fact_id = res["id"]
            score = (1.0 / (rank + 60)) * (1.0 - alpha)
            rrf_scores[fact_id] = rrf_scores.get(fact_id, 0) + score
            trace_data.setdefault(fact_id, {})["keyword_score"] = round(res["score"], 4)
            trace_data[fact_id]["keyword_rank"] = rank + 1

        # Merge result dictionaries
        all_hits = {res["id"]: res for res in knn_results + text_results}
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        
        final_results = []
        for fid in sorted_ids:
            hit = all_hits[fid]
            hit["rrf_score"] = round(rrf_scores[fid], 6)
            hit["trace"] = trace_data.get(fid, {})
            final_results.append(hit)
            
        return final_results

    def get_fact_by_key(self, user_id: str, fact_key: str) -> Optional[Dict[str, Any]]:
        """
        Finds the current active fact for a given key.
        """
        query = {
            "bool": {
                "must": [
                    {"term": {"user_id": user_id}},
                    {"term": {"fact_key": fact_key}},
                    {"term": {"is_outdated": False}}
                ]
            }
        }
        resp = self.client.search(index=self.index_name, query=query, size=1)
        hits = resp["hits"]["hits"]
        if hits:
            return {"id": hits[0]["_id"], **hits[0]["_source"]}
        return None
