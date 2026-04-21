"""
src/pipeline/extraction.py
====================================
The core orchestration logic for the fact extraction process.

Handles directory scanning, concurrent conversation processing,
error handling, and telemetry aggregation.
"""

import datetime
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import ValidationError

from src.config.settings import settings
from src.providers.factory import ProviderFactory
from src.memory.elasticsearch_store import ESFactStore
from src.deduplication.dedup import DeduplicationEngine



class FactExtractionPipeline:
    """
    Orchestrates reading conversations, coordinating concurrent extraction via the LLM provider,
    handling errors, and aggregating the telemetry and JSON results to disk.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        provider_name: str,
        model: str,
        delay: int = 1,
    ):
        """
        Initializes the fact extraction pipeline.
        """
        self.input_dir = Path(input_dir)
        token = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_dir) / f"extraction_pipeline_{token}"

        self.provider_name = provider_name
        self.model = model
        self.delay = delay
        self.max_workers = settings.max_workers
        self.provider = ProviderFactory.get_provider(
            provider_name, model, timeout=settings.api_timeout
        )

        # Initialize ES Store and Deduplication Engine (Provider-agnostic)
        self.store = ESFactStore()
        self.deduplicator = DeduplicationEngine(self.store, provider=provider_name, model=model)

    @staticmethod
    def _build_user_prompt(conversation: Dict[str, Any]) -> str:
        lines = [
            f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}"
            for msg in conversation.get("messages", [])
        ]
        return "Extract facts from this conversation:\n\n" + "\n".join(lines)

    def _create_error_result(
        self, conv_id: str, error: str, pii_count: int = 0, validation_count: int = 0
    ) -> Dict[str, Any]:
        return {
            "conversation_id": conv_id,
            "facts": [],
            "pii_filtered_count": pii_count,
            "validation_errors_count": validation_count,
            "model": self.model,
            "provider": self.provider_name,
            "error": error,
            "telemetry": {
                "api_latency_sec": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "estimated_cost_usd": 0.0,
                "success": False,
            },
        }

    def _extract_single_conversation(
        self, conversation: Dict[str, Any]
    ) -> Dict[str, Any]:
        conv_id = conversation.get("conversation_id", "unknown")
        logger.info(f"Processing: {conv_id} [{self.provider_name}/{self.model}]")
        user_prompt = self._build_user_prompt(conversation)

        try:
            # The calculate_telemetry decorator ensures generate_facts returns (FactsResponse, telemetry)
            facts_response, telemetry = self.provider.generate_facts(user_prompt)
            telemetry["success"] = True
        except ValidationError as e:
            blocked_count = sum(
                1 for err in e.errors() if "PII detected" in err.get("msg", "")
            )
            validation_error_count = len(e.errors()) - blocked_count
            logger.warning(
                f"Validation failed for {conv_id}: {blocked_count} PII, {validation_error_count} structured errors."
            )

            return self._create_error_result(
                conv_id=conv_id,
                error=str(e),
                pii_count=blocked_count,
                validation_count=validation_error_count,
            )
        except Exception as e:
            logger.error(f"API call or Parsing failed for {conv_id}: {e}")
            return self._create_error_result(conv_id, str(e))

        clean_facts = facts_response.model_dump()["facts"]

        # Process facts through the deduplication/conflict engine
        # For simulation, we'll try to find a user_id or use conversation_id
        user_id = conversation.get("user_id", conversation.get("conversation_id", "default_user"))
        
        final_fact_ids = []
        for fact_dict in clean_facts:
            try:
                fact_id = self.deduplicator.process_new_fact(user_id, fact_dict, conv_id)
                final_fact_ids.append(fact_id)
                logger.success(f"[{fact_dict['category']}] {fact_dict['fact']} (ID: {fact_id})")
            except Exception as e:
                logger.error(f"Failed to process fact '{fact_dict['fact']}': {e}")

        if self.delay > 0:
            time.sleep(self.delay)

        return {
            "conversation_id": conv_id,
            "user_id": user_id,
            "facts": clean_facts,
            "stored_fact_ids": final_fact_ids,
            "pii_filtered_count": 0,
            "validation_errors_count": 0,
            "model": self.model,
            "provider": self.provider_name,
            "telemetry": telemetry,
        }

    def _load_conversation(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to read/parse {path.name}: {e}. Skipping entirely.")
            return None

        if "messages" not in data:
            logger.warning(f"{path.name}: no 'messages' key - skipping")
            return None
        if "conversation_id" not in data:
            data["conversation_id"] = path.stem
        return data

    def extract_facts_from_messages(self, user_id: str, messages: List[Dict[str, str]], conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Extracts facts from a list of messages directly.
        """
        conv_id = conversation_id or f"direct_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        conversation = {
            "user_id": user_id,
            "conversation_id": conv_id,
            "messages": messages
        }
        return self._extract_single_conversation(conversation)

    def run(self) -> None:
        """
        Executes the concurrent extraction pipeline.
        """
        logger.info("Personal Memory Module - Fact Extraction Pipeline Started")
        logger.info(f"Provider: {self.provider_name} | Model: {self.model}")
        logger.info(f"Input: {self.input_dir} | Output: {self.output_dir}")
        logger.info(f"Concurrency: {self.max_workers} workers")

        json_files = sorted(self.input_dir.glob("*.json"))
        if not json_files:
            logger.error(f"No .json files found in {self.input_dir}")
            return

        logger.info(f"Found {len(json_files)} conversation file(s)")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure ES index exists before starting
        try:
            self.store.create_index()
        except Exception as e:
            logger.error(f"Failed to initialize ElasticSearch index: {e}")
            logger.warning("Pipeline will continue without persistence if ES is unreachable.")

        all_results = {}
        total_facts = 0
        total_pii = 0
        total_validation_errors = 0
        total_errors = 0
        agg_telemetry = {
            "total_api_latency_sec": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_estimated_cost_usd": 0.0,
            "successful_calls": 0,
            "failed_calls": 0,
        }

        conversations = []
        for path in json_files:
            conv = self._load_conversation(path)
            if conv is not None:
                conversations.append(conv)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_conv = {
                executor.submit(self._extract_single_conversation, conv): conv
                for conv in conversations
            }

            for future in as_completed(future_to_conv):
                try:
                    result = future.result()
                    conv_id = result["conversation_id"]
                    all_results[conv_id] = result
                    total_facts += len(result["facts"])
                    total_pii += result.get("pii_filtered_count", 0)
                    total_validation_errors += result.get("validation_errors_count", 0)

                    if "error" in result:
                        total_errors += 1
                        agg_telemetry["failed_calls"] += 1
                    else:
                        agg_telemetry["successful_calls"] += 1

                    tele = result.get("telemetry", {})
                    agg_telemetry["total_api_latency_sec"] += tele.get(
                        "api_latency_sec", 0.0
                    )
                    agg_telemetry["total_prompt_tokens"] += tele.get("prompt_tokens", 0)
                    agg_telemetry["total_completion_tokens"] += tele.get(
                        "completion_tokens", 0
                    )
                    agg_telemetry["total_estimated_cost_usd"] += tele.get(
                        "estimated_cost_usd", 0.0
                    )

                    out_path = self.output_dir / f"{conv_id}_facts.json"
                    out_path.write_text(
                        json.dumps(result, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    logger.info(f"Saved: {out_path}")
                except Exception as exc:
                    logger.error(
                        f"Conversation processing generated an exception: {exc}"
                    )

        total_convs = len(all_results)
        success_rate = (
            (agg_telemetry["successful_calls"] / total_convs * 100)
            if total_convs > 0
            else 0.0
        )
        agg_telemetry["total_api_latency_sec"] = round(
            agg_telemetry["total_api_latency_sec"], 2
        )
        agg_telemetry["total_estimated_cost_usd"] = round(
            agg_telemetry["total_estimated_cost_usd"], 6
        )
        agg_telemetry["avg_latency_per_call_sec"] = (
            round(
                agg_telemetry["total_api_latency_sec"]
                / agg_telemetry["successful_calls"],
                2,
            )
            if agg_telemetry["successful_calls"] > 0
            else 0.0
        )
        agg_telemetry["extraction_success_rate_pct"] = round(success_rate, 2)

        aggregate = {
            "summary": {
                "total_conversations": total_convs,
                "total_facts_extracted": total_facts,
                "total_pii_filtered": total_pii,
                "total_validation_errors": total_validation_errors,
                "total_pipeline_errors": total_errors,
                "provider": self.provider_name,
                "model": self.model,
                "telemetry": agg_telemetry,
            },
            "conversations": all_results,
        }
        agg_path = self.output_dir / "all_facts.json"
        agg_path.write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        logger.info("Pipeline Execution Complete!")
        logger.info(f"Aggregate output        : {agg_path}")
