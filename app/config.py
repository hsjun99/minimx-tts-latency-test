import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SYSTEM_PROMPT_PATH = str(BASE_DIR / "prompts" / "openai_system_prompt.txt")


@dataclass
class Settings:
    minimax_api_key: str
    minimax_base_url: str
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    openai_system_prompt_path: str


def get_settings() -> Settings:
    api_key = os.getenv("MINIMAX_API_KEY", "").strip()
    base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-5.1").strip()
    openai_system_prompt_path = os.getenv(
        "OPENAI_SYSTEM_PROMPT_PATH", DEFAULT_SYSTEM_PROMPT_PATH
    ).strip()

    return Settings(
        minimax_api_key=api_key,
        minimax_base_url=base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        openai_system_prompt_path=openai_system_prompt_path,
    )
