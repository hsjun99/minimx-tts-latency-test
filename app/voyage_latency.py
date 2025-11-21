import logging
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class VoyageConfig:
    api_key: str
    base_url: str
    model: str = "voyage-3.5-lite"
    dimension: int = 512


class VoyageLatencyError(RuntimeError):
    """Raised when Voyage embedding measurement cannot be completed."""


async def measure_voyage_embedding_latency(
    *,
    session: aiohttp.ClientSession,
    cfg: VoyageConfig,
    texts: list[str],
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    if not cfg.api_key:
        raise VoyageLatencyError("VOYAGE_AI_API_KEY is required")

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "input": texts,
        "model": cfg.model,
        "output_dimension": cfg.dimension,
    }

    start_time = time.perf_counter()

    try:
        async with session.post(
            cfg.base_url, headers=headers, json=payload, timeout=timeout_s
        ) as response:
            if response.status != 200:
                response_text = await response.text()
                raise VoyageLatencyError(
                    f"API request failed with status {response.status}: {response_text}"
                )

            result = await response.json()
            elapsed = time.perf_counter() - start_time

            # Validate response structure roughly
            if "data" not in result:
                raise VoyageLatencyError(
                    "Unexpected API response format: missing 'data' field"
                )

            return {
                "model": cfg.model,
                # "embeddings": [res["embedding"] for res in result["data"]],
                "latency_s": elapsed,
                "base_url": cfg.base_url,
            }

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.warning(f"Voyage embedding failed in {elapsed:.3f} seconds: {e}")
        if isinstance(e, VoyageLatencyError):
            raise e
        raise VoyageLatencyError(f"Voyage API request failed: {e}") from e
