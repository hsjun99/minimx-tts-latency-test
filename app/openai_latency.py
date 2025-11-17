import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp


@dataclass
class OpenAILatencyConfig:
    api_key: str
    model: str
    base_url: str
    system_prompt_path: str

    def completion_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions") or base.endswith("/responses"):
            return base
        return f"{base}/chat/completions"


class OpenAILatencyError(RuntimeError):
    """Raised when TTFT measurement cannot be completed."""


def _load_system_prompt(path: str) -> str:
    try:
        prompt = Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise OpenAILatencyError(f"System prompt file not found: {path}") from exc

    if not prompt:
        raise OpenAILatencyError(f"System prompt file is empty: {path}")
    return prompt


async def _await_ttft(resp: aiohttp.ClientResponse, start_time: float) -> float:
    buffer = ""
    async for chunk in resp.content.iter_chunked(1024):
        if not chunk:
            continue
        buffer += chunk.decode("utf-8", errors="ignore")
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            for line in block.splitlines():
                if not line.startswith("data:"):
                    continue
                payload = line.split("data:", 1)[1].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choice = next(iter(data.get("choices", [])), None)
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    continue
                content_piece = delta.get("content")
                if isinstance(content_piece, str) and content_piece:
                    return (time.perf_counter() - start_time) * 1000.0
    raise OpenAILatencyError("Stream ended before any content was received")


async def measure_openai_ttft(
    *,
    session: aiohttp.ClientSession,
    cfg: OpenAILatencyConfig,
    user_prompt: str,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    if not cfg.api_key:
        raise OpenAILatencyError("OPENAI_API_KEY is required")
    if not cfg.model.strip():
        raise OpenAILatencyError("Model is required")
    model = cfg.model.strip()
    if not user_prompt.strip():
        raise OpenAILatencyError("User prompt must be a non-empty string")
    user_prompt = user_prompt.strip()
    system_prompt = _load_system_prompt(cfg.system_prompt_path)

    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
    }

    url = cfg.completion_url()
    start = time.perf_counter()
    async with session.post(
        url, headers=headers, json=payload, timeout=timeout_s
    ) as resp:
        if resp.status >= 400:
            body = await resp.text()
            raise OpenAILatencyError(
                f"OpenAI request failed with status {resp.status}: {body}"
            )
        ttft_ms = await _await_ttft(resp, start)

    return {
        "ttft_ms": ttft_ms,
        "model": model,
        "system_prompt_path": cfg.system_prompt_path,
        "user_prompt": user_prompt,
        "base_url": cfg.completion_url(),
    }
