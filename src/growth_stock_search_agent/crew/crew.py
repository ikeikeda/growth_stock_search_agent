from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from crewai import Crew, Process

from growth_stock_search_agent.config import LOGS_DIR, get_settings
from growth_stock_search_agent.crew.tasks import build_tasks
from growth_stock_search_agent.models import (
    RankerOutput,
    ResearchReport,
    parse_ranker_output,
    parse_research_report,
    report_from_unaudited_ranker,
)
from growth_stock_search_agent.output.enrichment import enrich_report


def _task_raw(task_output: object) -> str:
    return str(getattr(task_output, "raw", task_output) or "")


def _save_crew_raw(result: object, extra: str = "") -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = LOGS_DIR / f"crew_raw_{timestamp}.txt"
    parts = [f"=== final raw ===\n{getattr(result, 'raw', result)}\n"]
    tasks_output = getattr(result, "tasks_output", None) or []
    for index, task_output in enumerate(tasks_output):
        parts.append(f"=== task[{index}] ===\n{_task_raw(task_output)}\n")
    if extra:
        parts.append(f"=== extra ===\n{extra}\n")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _extract_ranker_output(result: object) -> RankerOutput | None:
    chunks: list[str] = []
    tasks_output = getattr(result, "tasks_output", None) or []
    chunks.extend(_task_raw(task_output) for task_output in tasks_output)
    chunks.append(str(getattr(result, "raw", result)))

    found: RankerOutput | None = None
    for raw in chunks:
        try:
            found = parse_ranker_output(raw)
        except Exception:
            continue
    return found


def _raise_empty_llm(exc: Exception) -> None:
    raise ValueError(
        "Ollama から空の応答が返されました。"
        " Ollama が起動しているか、モデル名が正しいか確認してください。"
        " gemma4 等の thinking モデルでは think=False が必要です。"
        f" (model={get_settings().ollama_model})"
    ) from exc


def run_research_crew(research_prompt: str, retry_on_parse_error: bool = True) -> ResearchReport:
    tasks = build_tasks(research_prompt)
    crew = Crew(
        agents=[task.agent for task in tasks],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=False,
    )

    try:
        result = crew.kickoff()
    except Exception as exc:
        if "Invalid response from LLM call" in str(exc):
            _raise_empty_llm(exc)
        raise

    raw_output = str(result.raw if hasattr(result, "raw") else result)
    try:
        return enrich_report(parse_research_report(raw_output))
    except Exception as parse_error:
        log_path = _save_crew_raw(result, extra=str(parse_error))
        if retry_on_parse_error:
            ranker = _extract_ranker_output(result)
            if ranker is not None:
                print(
                    "警告: 最終出力を ResearchReport JSON として解析できませんでした。"
                    f" Ranker 結果から未監査レポートを組み立てます。 raw={log_path}"
                )
                return enrich_report(report_from_unaudited_ranker(ranker))
        raise ValueError(
            f"Failed to parse crew output as ResearchReport: {parse_error}. "
            f"raw saved to {log_path}"
        ) from parse_error
