from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
LOGS_DIR = PROJECT_ROOT / "logs"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:12b"
    tavily_api_key: str = ""
    google_sheets_id: str = ""
    google_service_account_json: str = "credentials/service_account.json"
    google_sheets_worksheet: str = "research_results"
    eval_quality_threshold: float = 0.6


@lru_cache
def get_settings() -> Settings:
    return Settings()


def check_ollama(base_url: str) -> tuple[bool, str]:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=5.0)
        response.raise_for_status()
        models = [m.get("name", "") for m in response.json().get("models", [])]
        return True, f"Ollama OK ({len(models)} models available)"
    except Exception as exc:
        return False, f"Ollama unreachable: {exc}"


def check_tavily(api_key: str) -> tuple[bool, str]:
    if not api_key.strip():
        return False, "TAVILY_API_KEY is not set"
    return True, "Tavily API key configured"


def check_google_sheets(sheets_id: str, credentials_path: str) -> tuple[bool, str]:
    if not sheets_id.strip():
        return False, "GOOGLE_SHEETS_ID is not set"
    path = PROJECT_ROOT / credentials_path
    if not path.exists():
        return False, f"Service account JSON not found: {path}"
    return True, f"Google Sheets credentials found: {path}"


def run_health_checks(settings: Settings | None = None) -> list[tuple[str, bool, str]]:
    settings = settings or get_settings()
    checks = [
        ("Ollama", *check_ollama(settings.ollama_base_url)),
        ("Tavily", *check_tavily(settings.tavily_api_key)),
        (
            "Google Sheets",
            *check_google_sheets(
                settings.google_sheets_id,
                settings.google_service_account_json,
            ),
        ),
    ]
    return [(name, ok, message) for name, ok, message in checks]
