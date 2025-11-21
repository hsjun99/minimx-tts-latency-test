import time
from dataclasses import dataclass
from typing import Any

import aiohttp


@dataclass
class OpenRouterLatencyConfig:
    api_key: str
    model: str
    base_url: str
    site_url: str = "https://example.com"
    site_name: str = "LatencyTest"

    def embeddings_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/embeddings"):
            return base
        return f"{base}/embeddings"


class OpenRouterLatencyError(RuntimeError):
    """Raised when latency measurement cannot be completed."""


async def measure_openrouter_embedding_latency(
    *,
    session: aiohttp.ClientSession,
    cfg: OpenRouterLatencyConfig,
    input_text: str,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    if not cfg.api_key:
        raise OpenRouterLatencyError("OPENROUTER_API_KEY is required")
    if not cfg.model.strip():
        raise OpenRouterLatencyError("Model is required")

    model = cfg.model.strip()
    if not input_text:
        raise OpenRouterLatencyError("Input text must be non-empty")

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": cfg.site_url,
        "X-Title": cfg.site_name,
    }

    payload = {
        "model": f"{model}:latency",
        "input": input_text,
        "encoding_format": "float",
    }

    url = cfg.embeddings_url()
    start = time.perf_counter()

    async with session.post(
        url, headers=headers, json=payload, timeout=timeout_s
    ) as resp:
        if resp.status >= 400:
            body = await resp.text()
            raise OpenRouterLatencyError(
                f"OpenRouter request failed with status {resp.status}: {body}"
            )
        # Read the response to ensure request is complete
        await resp.json()
        end = time.perf_counter()

    latency_ms = (end - start) * 1000.0

    return {
        "latency_ms": latency_ms,
        "model": model,
        "input_text": input_text,
        "base_url": cfg.embeddings_url(),
    }
