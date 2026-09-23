from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
import time
from dataclasses import dataclass

from crewai import Crew, Process

from growth_stock_search_agent.config import LOGS_DIR, get_settings
from growth_stock_search_agent.crew.report_parse import parse_crew_result
from growth_stock_search_agent.crew.tasks import build_tasks
from growth_stock_search_agent.models import (
    RANKER_RECOVERY_NOTE,
    ResearchReport,
    resolve_crew_report,
)
from growth_stock_search_agent.models import ResearchReport
from growth_stock_search_agent.output.enrichment import prepare_report
from growth_stock_search_agent.run_summary import (
    RunSummary,
    collect_task_raws,
    collect_task_snapshots,
    is_empty_llm_response,
    summary_from_failure,
    summary_from_report,
    write_task_output_log,
)


@dataclass
class CrewRunResult:
    report: ResearchReport | None
    summary: RunSummary
    raw_output: str = ""


def _is_retryable_parse_error(exc: BaseException) -> bool:
    if is_empty_llm_response(exc):
        return True
    if isinstance(exc, ValueError):
        return True
    return type(exc).__name__ in {"ValidationError", "JSONDecodeError"}


def _rejected_llm_outputs(crew: Crew) -> list[str]:
    remembered: list[str] = []
    for task in crew.tasks:
        llm = getattr(task.agent, "llm", None)
        items = getattr(llm, "rejected_outputs", None) or []
        for item in items:
            text = str(item)
            if text and text not in remembered:
                remembered.append(text)
    return remembered


def _annotate_ranker_recovery(summary: RunSummary, report: ResearchReport) -> None:
    summary.headline = (
        f"Ranker JSONから復旧 — 身元確認後 {len(report.candidates)} 件"
        "（ルーブリック評価は未実施）"
    )
    summary.diagnosis = f"{RANKER_RECOVERY_NOTE}\n{summary.diagnosis}"


def _success_result(
    *,
    report: ResearchReport,
    recovery_note: str,
    attempt: int,
    attempts: int,
    snapshots: list,
    raw_output: str,
    raws: list[str],
    error: BaseException | None,
    rejected_outputs: list[str],
) -> CrewRunResult:
    summary = summary_from_report(
        report,
        attempts=attempt,
        max_attempts=attempts,
        tasks=snapshots,
        raw_output=raw_output,
    )
    if recovery_note:
        _annotate_ranker_recovery(summary, report)
        log_path = write_task_output_log(
            tasks=snapshots,
            raws=raws,
            final_output=raw_output,
            error=error,
            rejected_outputs=rejected_outputs,
        )
        summary.log_path = str(log_path)
    return CrewRunResult(report=report, summary=summary, raw_output=raw_output)


def run_research_crew(
    research_prompt: str,
    retry_on_parse_error: bool = True,
    identity_lessons: str = "",
) -> CrewRunResult:
    tasks = build_tasks(research_prompt, identity_lessons=identity_lessons)
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
    last_raw = ""
    last_snapshots = []
    last_raws: list[str] = []
    last_result: Any = None
    attempts = 2 if retry_on_parse_error else 1

    for attempt in range(attempts):
        result = None
        try:
            result = crew.kickoff()
            last_raw = str(result.raw if hasattr(result, "raw") else result)
            last_snapshots = collect_task_snapshots(crew, result)
            last_raws = collect_task_raws(crew, result)
            report, recovery_note = resolve_crew_report(last_raw, last_raws)
            report = prepare_report(report)
            return _success_result(
                report=report,
                recovery_note=recovery_note,
                attempt=attempt + 1,
                attempts=attempts,
                snapshots=last_snapshots,
                raw_output=last_raw,
                raws=last_raws,
                error=None,
                rejected_outputs=_rejected_llm_outputs(crew),
            )
            last_result = result
            return prepare_report(parse_crew_result(result))
        except Exception as exc:
            last_error = exc
            last_snapshots = collect_task_snapshots(crew, result)
            last_raws = collect_task_raws(crew, result)
            if result is not None and not last_raw:
                last_raw = str(getattr(result, "raw", "") or "")
            if _is_retryable_parse_error(exc):
                try:
                    report, recovery_note = resolve_crew_report(last_raw, last_raws)
                except (ValueError, json.JSONDecodeError):
                    report = None
                    recovery_note = ""
                if report is not None and recovery_note:
                    try:
                        prepared = prepare_report(report)
                    except Exception as prepare_exc:
                        last_error = prepare_exc
                    else:
                        return _success_result(
                            report=prepared,
                            recovery_note=recovery_note,
                            attempt=attempt + 1,
                            attempts=attempts,
                            snapshots=last_snapshots,
                            raw_output=last_raw,
                            raws=last_raws,
                            error=exc,
                            rejected_outputs=_rejected_llm_outputs(crew),
                        )
            retryable = _is_retryable_parse_error(exc)
            if retryable and attempt + 1 < attempts:
                if is_empty_llm_response(exc):
                    time.sleep(2**attempt)
                continue
            if is_empty_llm_response(exc):
                settings = get_settings()
                last_error = ValueError(
                    "Ollama から空の応答が返されました。"
                    " gemma4 は content を空のまま thinking だけ返すことがあります。"
                    " OLLAMA_DISABLE_THINKING=false のまま、OLLAMA_MAX_TOKENS を"
                    " OLLAMA_NUM_CTX より小さくしてください。"
                    f" (model={settings.ollama_model},"
                    f" max_tokens={settings.ollama_max_tokens},"
                    f" num_ctx={settings.ollama_num_ctx},"
                    f" attempt={attempt + 1}/{attempts})"
                )
            break

    error = last_error or ValueError("Crew run failed without a captured exception")
    summary = summary_from_failure(
        error,
        attempts=attempts,
        max_attempts=attempts,
        tasks=last_snapshots,
        raw_output=last_raw,
    )
    log_path = write_task_output_log(
        tasks=last_snapshots,
        raws=last_raws,
        final_output=last_raw,
        error=error,
        rejected_outputs=_rejected_llm_outputs(crew),
    )
    summary.log_path = str(log_path)
    return CrewRunResult(
        report=None,
        summary=summary,
        raw_output=last_raw,
    )
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
