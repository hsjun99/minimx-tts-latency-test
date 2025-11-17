from fastapi import FastAPI, HTTPException, Query
import aiohttp

from .config import get_settings
from .minimax_latency import measure_minimax_tts_latency, MiniMaxConfig
from .openai_latency import (
    OpenAILatencyConfig,
    OpenAILatencyError,
    measure_openai_ttft,
)

app = FastAPI(title="MiniMax TTS Latency Test")


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
        default=None,
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
