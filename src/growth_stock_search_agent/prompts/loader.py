from __future__ import annotations

import json
from pathlib import Path

from growth_stock_search_agent.config import PROMPTS_DIR

BASE_PROMPT_PATH = PROMPTS_DIR / "base_prompt.txt"
OPTIMIZED_PROMPT_PATH = PROMPTS_DIR / "optimized_prompt.json"


def load_base_prompt() -> str:
    return BASE_PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_research_prompt(use_base: bool = False) -> str:
    if use_base or not OPTIMIZED_PROMPT_PATH.exists():
        return load_base_prompt()

    payload = json.loads(OPTIMIZED_PROMPT_PATH.read_text(encoding="utf-8"))
    prompt = payload.get("prompt", "").strip()
    return prompt or load_base_prompt()


def save_optimized_prompt(prompt: str, score: float, char_count: int) -> Path:
    from datetime import datetime, timezone

    payload = {
        "prompt": prompt,
        "score": score,
        "char_count": char_count,
        "optimized_at": datetime.now(timezone.utc).isoformat(),
    }
    OPTIMIZED_PROMPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OPTIMIZED_PROMPT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return OPTIMIZED_PROMPT_PATH
