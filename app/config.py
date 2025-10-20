import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Load .env if present
load_dotenv()


@dataclass
class Settings:
    minimax_api_key: str
    minimax_base_url: str


def get_settings() -> Settings:
    api_key = os.getenv("MINIMAX_API_KEY", "").strip()
    base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimax.io").strip()
    return Settings(minimax_api_key=api_key, minimax_base_url=base_url)
