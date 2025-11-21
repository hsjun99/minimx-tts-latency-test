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

| Name                              | Required            | Description                                                                                         |
| --------------------------------- | ------------------- | --------------------------------------------------------------------------------------------------- |
| `MINIMAX_API_KEY`                 | ✅                  | MiniMax API key for TTS latency.                                                                    |
| `MINIMAX_BASE_URL`                | ⛔️                 | MiniMax base URL (defaults to `https://api.minimax.io`).                                            |
| `OPENAI_API_KEY`                  | ✅ (for TTFT)       | OpenAI (or compatible) API key.                                                                     |
| `OPENAI_BASE_URL`                 | ⛔️                 | Base URL for OpenAI-compatible endpoint (defaults to `https://api.openai.com/v1`).                  |
| `OPENAI_MODEL`                    | ⛔️                 | Chat completion model (defaults to `gpt-5.1`).                                                      |
| `OPENAI_SYSTEM_PROMPT_PATH`       | ⛔️                 | Path to the system prompt file (defaults to `app/prompts/openai_system_prompt.txt`).                |
| `AZURE_OPENAI_API_KEY`            | ✅ (for Azure TTFT) | Azure OpenAI API key.                                                                               |
| `AZURE_OPENAI_ENDPOINT`           | ✅ (for Azure TTFT) | Azure OpenAI endpoint base URL (e.g., `https://my-resource.openai.azure.com`).                      |
| `AZURE_OPENAI_DEPLOYMENT`         | ⛔️ (Azure TTFT)    | Default Azure OpenAI deployment name (optional if you instead set `AZURE_OPENAI_MODEL`).            |
| `AZURE_OPENAI_MODEL`              | ⛔️ (Azure TTFT)    | Azure OpenAI model/deployment identifier for request body (optional if deployment path is used).    |
| `AZURE_OPENAI_API_VERSION`        | ⛔️                 | API version for Azure OpenAI (defaults to `2024-02-15-preview`).                                    |
| `AZURE_OPENAI_SYSTEM_PROMPT_PATH` | ⛔️                 | Path to the system prompt file for Azure TTFT (defaults to `app/prompts/openai_system_prompt.txt`). |
| `OPENROUTER_API_KEY`              | ✅ (for OpenRouter) | OpenRouter API key.                                                                                 |
| `OPENROUTER_BASE_URL`             | ⛔️                 | OpenRouter base URL (defaults to `https://openrouter.ai/api/v1`).                                   |

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

### Azure OpenAI TTFT

`GET /azure-openai/ttft`

Query params:

- `prompt` (required): user prompt text to send to the deployment.
- `deployment` (optional): override the configured deployment name (required if no default is set via env and no `model` provided).
- `model` (optional): Azure OpenAI model/deployment identifier to send in the request body (required if no deployment path is used).
- `api_version` (optional): override the configured API version.
- `timeout_s` (optional): total timeout in seconds. Default: `10.0`

Response example:

```json
{
  "ttft_ms": 152.8,
  "deployment": "gpt-4o-mini-live",
  "model": null,
  "api_version": "2024-02-15-preview",
  "system_prompt_path": "app/prompts/openai_system_prompt.txt",
  "user_prompt": "Explain how rain forms.",
  "base_url": "https://my-resource.openai.azure.com/openai/deployments/gpt-4o-mini-live/chat/completions?api-version=2024-02-15-preview"
}
```

### OpenRouter Embedding Latency

`GET /openrouter/embedding-latency`

Query params:

- `text` (required): text input to embed.
- `model` (optional): OpenRouter model identifier. Default: `qwen/qwen3-embedding-8b`.
- `timeout_s` (optional): total timeout in seconds. Default: `10.0`

Response example:

```json
{
  "latency_ms": 45.6,
  "model": "qwen/qwen3-embedding-8b",
  "input_text": "Your text string goes here",
  "base_url": "https://openrouter.ai/api/v1/embeddings"
}
```

## Notes

- Auth via `Authorization: Bearer <MINIMAX_API_KEY>` header.
- WS URL is derived from `MINIMAX_BASE_URL` by replacing http→ws and appending `/ws/v1/t2a_v2`.
- We stop at the first audio chunk to compute TTFB, then send `task_finish`.
- OpenAI TTFT uses streaming chat completions; TTFT is captured when the first non-empty `delta.content` arrives.
- Azure OpenAI TTFT uses the `api-key` header and the configured deployment/API version.
