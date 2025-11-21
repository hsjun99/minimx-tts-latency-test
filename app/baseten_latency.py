import asyncio
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
from baseten_performance_client import PerformanceClient


@dataclass
class BasetenLatencyConfig:
    api_key: str
    model_id: str
    site_url: str = "https://example.com"
    site_name: str = "LatencyTest"

    @property
    def endpoint_url(self) -> str:
        return (
            f"https://model-{self.model_id}.api.baseten.co/environments/production/sync"
        )


class BasetenLatencyError(RuntimeError):
    """Raised when latency measurement cannot be completed."""


def format_qwen_query(task_description: str, query: str) -> str:
    """Formats the query for Qwen-3-embedding if a task description is provided."""
    if not task_description:
        return query
    return f"Instruct: {task_description}\nQuery:{query}"


async def measure_baseten_embedding_latency(
    *,
    session: aiohttp.ClientSession,  # Kept for signature compatibility but unused by PerformanceClient
    cfg: BasetenLatencyConfig,
    input_text: str,
    task_description: str = "",
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    if not cfg.api_key:
        raise BasetenLatencyError("BASETEN_API_KEY is required")
    if not cfg.model_id:
        raise BasetenLatencyError("BASETEN_MODEL_ID is required")

    if not input_text:
        raise BasetenLatencyError("Input text must be non-empty")

    # Format if task description is present
    final_input = format_qwen_query(task_description, input_text)

    client = PerformanceClient(base_url=cfg.endpoint_url, api_key=cfg.api_key)
    texts = [final_input]

    def _run_embed():
        return client.embed(
            input=texts,
            model="my_model",
            batch_size=16,  # Kept from user snippet
            max_concurrent_requests=32,  # Kept from user snippet
        )

    loop = asyncio.get_running_loop()
    start = time.perf_counter()

    try:
        # Run the synchronous client in an executor
        await loop.run_in_executor(None, _run_embed)
        end = time.perf_counter()
    except Exception as e:
        raise BasetenLatencyError(f"Baseten request failed: {str(e)}")

    latency_ms = (end - start) * 1000.0

    return {
        "latency_ms": latency_ms,
        "model_id": cfg.model_id,
        "input_text": input_text,
        "formatted_input": final_input if task_description else None,
        "endpoint_url": cfg.endpoint_url,
    }
