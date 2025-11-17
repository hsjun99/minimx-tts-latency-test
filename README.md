# MiniMax TTS Latency Test (FastAPI + Poetry)

This service measures:

- MiniMax TTS WebSocket latencies (connect + TTFB)
- OpenAI chat TTFT (time-to-first-token)

## Requirements

- Python 3.11–3.12
- MiniMax API key

## Setup

```bash
poetry install
cp .env.example .env  # if you have one, otherwise create a new .env
# edit .env and set the required keys
```

### Environment variables

| Name                        | Required      | Description                                                                          |
| --------------------------- | ------------- | ------------------------------------------------------------------------------------ |
| `MINIMAX_API_KEY`           | ✅            | MiniMax API key for TTS latency.                                                     |
| `MINIMAX_BASE_URL`          | ⛔️           | MiniMax base URL (defaults to `https://api.minimax.io`).                             |
| `OPENAI_API_KEY`            | ✅ (for TTFT) | OpenAI (or compatible) API key.                                                      |
| `OPENAI_BASE_URL`           | ⛔️           | Base URL for OpenAI-compatible endpoint (defaults to `https://api.openai.com/v1`).   |
| `OPENAI_MODEL`              | ⛔️           | Chat completion model (defaults to `gpt-5.1`).                                       |
| `OPENAI_SYSTEM_PROMPT_PATH` | ⛔️           | Path to the system prompt file (defaults to `app/prompts/openai_system_prompt.txt`). |

## Run

```bash
poetry run uvicorn app.main:app --reload --port 8000
```

## Endpoint

### MiniMax

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

### OpenAI TTFT

`GET /openai/ttft`

Query params:

- `prompt` (required): user prompt text to send to the model.
- `model` (optional): OpenAI-compatible model name. Default: `gpt-5.1`.
- `timeout_s` (optional): total timeout in seconds. Default: `10.0`

Response example:

```json
{
  "ttft_ms": 134.2,
  "model": "gpt-5.1",
  "system_prompt_path": "app/prompts/openai_system_prompt.txt",
  "user_prompt": "Explain how rain forms.",
  "base_url": "https://api.openai.com/v1/chat/completions"
}
```

## Notes

- Auth via `Authorization: Bearer <MINIMAX_API_KEY>` header.
- WS URL is derived from `MINIMAX_BASE_URL` by replacing http→ws and appending `/ws/v1/t2a_v2`.
- We stop at the first audio chunk to compute TTFB, then send `task_finish`.
- OpenAI TTFT uses streaming chat completions; TTFT is captured when the first non-empty `delta.content` arrives.
