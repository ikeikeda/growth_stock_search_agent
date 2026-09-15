from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import time

from crewai import Crew, Process

from growth_stock_search_agent.config import LOGS_DIR, get_settings
from growth_stock_search_agent.crew.report_parse import parse_crew_result
from growth_stock_search_agent.crew.tasks import build_tasks
from growth_stock_search_agent.models import ResearchReport
from growth_stock_search_agent.output.enrichment import prepare_report


def _is_empty_llm_response(exc: BaseException) -> bool:
    return "Invalid response from LLM call" in str(exc)


def _dump_parse_failure(result: Any, error: Exception | None) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = LOGS_DIR / f"parse_failure_{timestamp}.txt"
    chunks: list[str] = [f"error: {error}", ""]
    raw = getattr(result, "raw", None)
    if raw:
        chunks.append("=== crew.raw ===")
        chunks.append(str(raw)[:200_000])
        chunks.append("")
    for index, task in enumerate(getattr(result, "tasks_output", None) or []):
        chunks.append(f"=== task[{index}] {getattr(task, 'agent', '')} ===")
        chunks.append(str(getattr(task, "raw", ""))[:200_000])
        chunks.append("")
    path.write_text("\n".join(chunks), encoding="utf-8")
    print(f"パース失敗時の生出力を保存しました: {path}")


def run_research_crew(research_prompt: str, retry_on_parse_error: bool = True) -> ResearchReport:
    tasks = build_tasks(research_prompt)
    crew = Crew(
        agents=[task.agent for task in tasks],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=False,
    )

    last_error: Exception | None = None
    last_result: Any = None
    attempts = 2 if retry_on_parse_error else 1

    for attempt in range(attempts):
        try:
            result = crew.kickoff()
            last_result = result
            return prepare_report(parse_crew_result(result))
        except Exception as exc:
            last_error = exc
            if not _is_empty_llm_response(exc):
                # Parse / other ValueErrors: retry when attempts remain.
                if isinstance(exc, ValueError) and attempt + 1 < attempts:
                    continue
                if isinstance(exc, ValueError):
                    break
                raise

            settings = get_settings()
            if attempt + 1 < attempts:
                # Transient empty content (e.g. thinking budget edge cases).
                time.sleep(2 ** attempt)
                continue
            raise ValueError(
                "Ollama から空の応答が返されました。"
                " gemma4 は content を空のまま thinking だけ返すことがあります。"
                " OLLAMA_DISABLE_THINKING=false のまま、OLLAMA_MAX_TOKENS を"
                " OLLAMA_NUM_CTX より小さくしてください。"
                f" (model={settings.ollama_model},"
                f" max_tokens={settings.ollama_max_tokens},"
                f" num_ctx={settings.ollama_num_ctx},"
                f" attempt={attempt + 1}/{attempts})"
            ) from exc

    _dump_parse_failure(last_result, last_error)
    raise ValueError(f"Failed to parse crew output as ResearchReport: {last_error}")
