import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal, Optional

import aiohttp

DEFAULT_BASE_URL = "https://api.minimax.io"

TTSModel = Literal[
    "speech-2.5-hd-preview",
    "speech-2.5-turbo-preview",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
]

TTSVoice = Literal[
    "voice_agent_Female_Phone_4",
    "voice_agent_Male_Phone_1",
    "English_StressedLady",
    "English_SentimentalLady",
    "English_WiseScholar",
    "English_radiant_girl",
]

TTSEmotion = Literal[
    "happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"
]

TTSAudioFormat = Literal["pcm", "mp3", "flac", "wav"]
TTSSampleRate = Literal[8000, 16000, 22050, 24000, 32000, 44100]
TTSBitRate = Literal[32000, 64000, 128000, 256000]


@dataclass
class MiniMaxConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: TTSModel | str = "speech-02-turbo"
    voice_id: TTSVoice | str = "English_radiant_girl"
    sample_rate: TTSSampleRate = 24000
    bitrate: TTSBitRate = 128000
    audio_format: TTSAudioFormat = "pcm"
    speed: float = 1.0
    vol: float = 1.0
    pitch: int = 0
    emotion: Optional[TTSEmotion] = None


def _ws_url(base_url: str) -> str:
    url = base_url
    if url.startswith("http"):
        url = url.replace("http", "ws", 1)
    return f"{url}/ws/v1/t2a_v2"


def build_task_start(cfg: MiniMaxConfig) -> dict[str, Any]:
    msg: dict[str, Any] = {
        "event": "task_start",
        "model": cfg.model,
        "voice_setting": {
            "voice_id": cfg.voice_id,
            "speed": cfg.speed,
            "vol": cfg.vol,
            "pitch": cfg.pitch,
        },
        "audio_setting": {
            "sample_rate": cfg.sample_rate,
            "bitrate": cfg.bitrate,
            "format": cfg.audio_format,
            "channel": 1,
        },
    }
    if cfg.emotion is not None:
        msg["voice_setting"]["emotion"] = cfg.emotion
    return msg


async def measure_minimax_tts_latency(
    *,
    session: aiohttp.ClientSession,
    api_key: str,
    base_url: str | None = None,
    sentence: str = "안녕하세요. 오늘도 좋은 하루 보내세요.",
    timeout_s: float = 10.0,
    cfg: MiniMaxConfig | None = None,
) -> dict[str, Any]:
    """Connect to MiniMax TTS over WebSocket, send a sentence and measure latencies.

    Returns a dict containing ws_connect_ms, connected_success_ms, ttfb_ms and echo of config.
    """
    if not api_key:
        raise ValueError("MINIMAX_API_KEY is required")

    cfg = cfg or MiniMaxConfig(api_key=api_key, base_url=base_url or DEFAULT_BASE_URL)
    if base_url:
        cfg.base_url = base_url

    headers = {"Authorization": f"Bearer {api_key}"}
    ws_uri = _ws_url(cfg.base_url)

    connect_start = time.perf_counter()
    try:
        ws = await asyncio.wait_for(
            session.ws_connect(ws_uri, headers=headers), timeout_s
        )
    except Exception as e:
        raise RuntimeError(f"Failed to connect MiniMax WS: {e}") from e
    connect_end = time.perf_counter()

    ws_connect_ms = (connect_end - connect_start) * 1000.0

    connected_success_ms: Optional[float] = None

    async def recv_until(predicate):
        while True:
            msg = await asyncio.wait_for(ws.receive(), timeout=timeout_s)
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if predicate(data):
                    return data
                # surface non-zero status
                status = data.get("base_resp", {}).get("status_code")
                if status not in (None, 0):
                    raise RuntimeError(f"MiniMax error status_code={status}: {data}")
            elif msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                raise RuntimeError("MiniMax WS closed unexpectedly")

    # Wait for connected_success (optional but informative)
    try:
        await recv_until(lambda d: d.get("event") == "connected_success")
        connected_success_ms = (time.perf_counter() - connect_end) * 1000.0
    except Exception:
        # best-effort; continue even if not seen
        connected_success_ms = None

    # Send task_start and wait for task_started
    await ws.send_str(json.dumps(build_task_start(cfg)))
    await recv_until(lambda d: d.get("event") == "task_started")

    # Send sentence and measure TTFB (time to first audio)
    t_send = time.perf_counter()
    await ws.send_str(json.dumps({"event": "task_continue", "text": sentence}))

    async def first_audio_predicate(d: dict[str, Any]) -> bool:
        if d.get("event") == "task_continued":
            audio_hex = d.get("data", {}).get("audio")
            return bool(audio_hex)
        # propagate errors early
        status = d.get("base_resp", {}).get("status_code")
        if status not in (None, 0):
            raise RuntimeError(f"MiniMax error status_code={status}: {d}")
        return False

    await recv_until(first_audio_predicate)
    ttfb_ms = (time.perf_counter() - t_send) * 1000.0

    # Graceful finish
    try:
        await ws.send_str(json.dumps({"event": "task_finish"}))
        # drain until finished or timeout, best-effort
        try:
            await recv_until(
                lambda d: d.get("event") in {"task_finished", "task_failed"}
            )
        except Exception:
            pass
    finally:
        await ws.close()

    return {
        "ws_connect_ms": ws_connect_ms,
        "connected_success_ms": connected_success_ms,
        "ttfb_ms": ttfb_ms,
        "model": cfg.model,
        "voice_id": cfg.voice_id,
        "audio_format": cfg.audio_format,
        "sample_rate": cfg.sample_rate,
        "sentence": sentence,
        "base_url": cfg.base_url,
    }
