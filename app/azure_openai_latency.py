import time
from dataclasses import dataclass
from typing import Any

import aiohttp

from .openai_latency import (
    OpenAILatencyError,
    _await_ttft,
    _load_system_prompt,
)


@dataclass
class AzureOpenAIConfig:
    api_key: str
    endpoint: str
    api_version: str
    system_prompt_path: str
    deployment: str | None = None
    model: str | None = None

    def completion_url(self) -> str:
        base = self.endpoint.rstrip("/")
        if self.deployment:
            return (
                f"{base}/openai/deployments/{self.deployment}/chat/completions"
                f"?api-version={self.api_version}"
            )
        return f"{base}/openai/chat/completions?api-version={self.api_version}"


async def measure_azure_openai_ttft(
    *,
    session: aiohttp.ClientSession,
    cfg: AzureOpenAIConfig,
    user_prompt: str,
    timeout_s: float = 20.0,
) -> dict[str, Any]:
    if not cfg.api_key:
        raise OpenAILatencyError("AZURE_OPENAI_API_KEY is required")
    if not cfg.endpoint.strip():
        raise OpenAILatencyError("AZURE_OPENAI_ENDPOINT is required")
    deployment = cfg.deployment.strip() if cfg.deployment else ""
    if not cfg.api_version.strip():
        raise OpenAILatencyError("AZURE_OPENAI_API_VERSION is required")
    if not user_prompt.strip():
        raise OpenAILatencyError("User prompt must be a non-empty string")

    endpoint = cfg.endpoint.strip()
    api_version = cfg.api_version.strip()
    user_prompt = user_prompt.strip()
    system_prompt = _load_system_prompt(cfg.system_prompt_path)

    headers = {
        "api-key": cfg.api_key,
        "Content-Type": "application/json",
    }
    selected_model: str | None = None

    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": True,
    }
    if not deployment:
        selected_model = (cfg.model or "").strip()
        if not selected_model:
            raise OpenAILatencyError(
                "AZURE_OPENAI_MODEL is required when deployment is not specified"
            )
        payload["model"] = selected_model

    url = cfg.completion_url()
    start = time.perf_counter()
    async with session.post(
        url, headers=headers, json=payload, timeout=timeout_s
    ) as resp:
        if resp.status >= 400:
            body = await resp.text()
            raise OpenAILatencyError(
                f"Azure OpenAI request failed with status {resp.status}: {body}"
            )
        ttft_ms = await _await_ttft(resp, start)

    return {
        "ttft_ms": ttft_ms,
        "deployment": deployment or None,
        "model": selected_model,
        "api_version": api_version,
        "system_prompt_path": cfg.system_prompt_path,
        "user_prompt": user_prompt,
        "base_url": url,
    }
