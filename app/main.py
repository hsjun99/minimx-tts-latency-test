from fastapi import FastAPI, HTTPException, Query
import aiohttp

from .config import get_settings
from .minimax_latency import measure_minimax_tts_latency, MiniMaxConfig

app = FastAPI(title="MiniMax TTS Latency Test")


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
