# MiniMax TTS Latency Test (FastAPI + Poetry)

This service measures MiniMax TTS WebSocket latencies:
- WS connect latency
- TTFB (from sending text to first audio chunk)

## Requirements
- Python 3.11–3.12
- MiniMax API key

## Setup
```bash
poetry install
cp .env.example .env
# edit .env and set MINIMAX_API_KEY=<your key>
```

## Run
```bash
poetry run uvicorn app.main:app --reload --port 8000
```

## Endpoint
`GET /minimax/tts-latency`

Query params:
- `sentence` (optional): text to synthesize. Default: `안녕하세요. 오늘도 좋은 하루 보내세요.`
- `timeout_s` (optional): overall timeout in seconds. Default: `10.0`

Response example:
```json
{
  "ws_connect_ms": 85.2,
  "connected_success_ms": 12.7,
  "ttfb_ms": 210.4,
  "model": "speech-02-turbo",
  "voice_id": "English_radiant_girl",
  "audio_format": "mp3",
  "sample_rate": 24000,
  "sentence": "안녕하세요. 오늘도 좋은 하루 보내세요.",
  "base_url": "https://api.minimax.io"
}
```

## Notes
- Auth via `Authorization: Bearer <MINIMAX_API_KEY>` header.
- WS URL is derived from `MINIMAX_BASE_URL` by replacing http→ws and appending `/ws/v1/t2a_v2`.
- We stop at the first audio chunk to compute TTFB, then send `task_finish`.
