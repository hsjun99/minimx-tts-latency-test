import logging

from fastapi import FastAPI, HTTPException, Query
import aiohttp

from .azure_openai_latency import AzureOpenAIConfig, measure_azure_openai_ttft
from .config import get_settings
from .minimax_latency import measure_minimax_tts_latency, MiniMaxConfig
from .openai_latency import (
    OpenAILatencyConfig,
    OpenAILatencyError,
    measure_openai_ttft,
)

app = FastAPI(title="MiniMax TTS Latency Test")
logger = logging.getLogger(__name__)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/minimax/tts-latency")
async def minimax_tts_latency(
    sentence: str = Query(
        default="안녕하세요. 오늘도 좋은 하루 보내세요.",
        description="Text to synthesize for measuring TTFB.",
    ),
    timeout_s: float = Query(
        default=10.0, ge=1.0, le=60.0, description="Overall timeout in seconds"
    ),
):
    settings = get_settings()
    if not settings.minimax_api_key:
        raise HTTPException(status_code=500, detail="MINIMAX_API_KEY is not configured")

    async with aiohttp.ClientSession() as session:
        try:
            result = await measure_minimax_tts_latency(
                session=session,
                api_key=settings.minimax_api_key,
                base_url=settings.minimax_base_url,
                sentence=sentence,
                timeout_s=timeout_s,
                cfg=MiniMaxConfig(
                    api_key=settings.minimax_api_key, base_url=settings.minimax_base_url
                ),
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=str(e))

    return result


@app.get("/openai/ttft")
async def openai_ttft_latency(
    prompt: str = Query(
        ...,
        min_length=1,
        description="User prompt text to send to the OpenAI model",
    ),
    model: str | None = Query(
        default="gpt-5.1",
        description="OpenAI model identifier to use for this request (default: gpt-5.1)",
    ),
    timeout_s: float = Query(
        default=10.0, ge=1.0, le=60.0, description="Overall timeout in seconds"
    ),
):
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")

    selected_model = (model or settings.openai_model).strip()
    if not selected_model:
        raise HTTPException(status_code=400, detail="Model parameter cannot be blank")

    cfg = OpenAILatencyConfig(
        api_key=settings.openai_api_key,
        model=selected_model,
        base_url=settings.openai_base_url,
        system_prompt_path=settings.openai_system_prompt_path,
    )

    async with aiohttp.ClientSession() as session:
        try:
            result = await measure_openai_ttft(
                session=session,
                cfg=cfg,
                user_prompt=prompt,
                timeout_s=timeout_s,
            )
        except OpenAILatencyError as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    return result


@app.get("/azure-openai/ttft")
async def azure_openai_ttft_latency(
    prompt: str = Query(
        ...,
        min_length=1,
        description="User prompt text to send to the Azure OpenAI deployment",
    ),
    deployment: str | None = Query(
        default=None,
        description="Azure OpenAI deployment name (defaults to config value)",
    ),
    model: str | None = Query(
        default="gpt-5.1",
        description="Azure OpenAI model/deployment identifier (required if no deployment path is used)",
    ),
    api_version: str | None = Query(
        default=None,
        description="Azure OpenAI API version (defaults to config value)",
    ),
    timeout_s: float = Query(
        default=10.0, ge=1.0, le=60.0, description="Overall timeout in seconds"
    ),
):
    settings = get_settings()
    if not settings.azure_openai_api_key:
        raise HTTPException(
            status_code=500, detail="AZURE_OPENAI_API_KEY is not configured"
        )
    if not settings.azure_openai_endpoint:
        raise HTTPException(
            status_code=500, detail="AZURE_OPENAI_ENDPOINT is not configured"
        )
    default_deployment = settings.azure_openai_deployment.strip()
    selected_deployment = (deployment or default_deployment).strip()
    selected_model = (model or settings.azure_openai_model).strip()
    if not selected_deployment and not selected_model:
        preview_prompt = prompt[:80] + ("..." if len(prompt) > 80 else "")
        detail = "Provide either deployment query param (or AZURE_OPENAI_DEPLOYMENT) or model/ AZURE_OPENAI_MODEL"
        logger.warning(
            "Azure TTFT request rejected: missing deployment/model (prompt=%s)",
            preview_prompt,
        )
        raise HTTPException(
            status_code=400,
            detail=detail,
        )

    default_api_version = settings.azure_openai_api_version.strip()
    selected_api_version = (api_version or default_api_version).strip()
    if not selected_api_version:
        raise HTTPException(
            status_code=400, detail="API version parameter cannot be blank"
        )

    cfg = AzureOpenAIConfig(
        api_key=settings.azure_openai_api_key,
        endpoint=settings.azure_openai_endpoint,
        deployment=selected_deployment or selected_model,
        model=selected_model or None,
        api_version=selected_api_version,
        system_prompt_path=settings.azure_openai_system_prompt_path,
    )

    async with aiohttp.ClientSession() as session:
        try:
            result = await measure_azure_openai_ttft(
                session=session,
                cfg=cfg,
                user_prompt=prompt,
                timeout_s=timeout_s,
            )
        except OpenAILatencyError as exc:
            logger.exception("Azure OpenAI TTFT failed: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc))
        except Exception as exc:
            logger.exception("Azure OpenAI TTFT unexpected error: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc))

    return result
