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
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str
    azure_openai_model: str
    azure_openai_api_version: str
    azure_openai_system_prompt_path: str


def get_settings() -> Settings:
    api_key = os.getenv("MINIMAX_API_KEY", "").strip()
    base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io").strip()
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    openai_model = os.getenv("OPENAI_MODEL", "gpt-5.1").strip()
    openai_system_prompt_path = os.getenv(
        "OPENAI_SYSTEM_PROMPT_PATH", DEFAULT_SYSTEM_PROMPT_PATH
    ).strip()
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
    azure_model = os.getenv("AZURE_OPENAI_MODEL", "").strip()
    azure_api_version = os.getenv(
        "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
    ).strip()
    azure_system_prompt_path = os.getenv(
        "AZURE_OPENAI_SYSTEM_PROMPT_PATH", DEFAULT_SYSTEM_PROMPT_PATH
    ).strip()

    return Settings(
        minimax_api_key=api_key,
        minimax_base_url=base_url,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        openai_system_prompt_path=openai_system_prompt_path,
        azure_openai_api_key=azure_api_key,
        azure_openai_endpoint=azure_endpoint,
        azure_openai_deployment=azure_deployment,
        azure_openai_model=azure_model,
        azure_openai_api_version=azure_api_version,
        azure_openai_system_prompt_path=azure_system_prompt_path,
    )
