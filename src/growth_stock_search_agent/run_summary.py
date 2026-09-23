from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from growth_stock_search_agent.models import ResearchReport

STAGE_LABELS = (
    "Researcher（候補収集）",
    "Analyst（絞り込み）",
    "Ranker（順位付けJSON）",
    "Evaluator（最終レポートJSON）",
)

_PREVIEW_LIMIT = 280
_FINAL_PREVIEW_LIMIT = 600


class RunStatus(str, Enum):
    success = "success"
    no_pass = "no_pass"
    quality_skip = "quality_skip"
    parse_error = "parse_error"
    empty_llm = "empty_llm"
    error = "error"


@dataclass
class TaskSnapshot:
    index: int
    stage: str
    agent: str
    chars: int
    has_json: bool
    preview: str


@dataclass
class RunSummary:
    status: RunStatus
    headline: str
    diagnosis: str
    attempts: int = 1
    max_attempts: int = 1
    error_type: str = ""
    error_message: str = ""
    tasks: list[TaskSnapshot] = field(default_factory=list)
    final_output_preview: str = ""
    candidate_count: int | None = None
    passing_codes: list[str] = field(default_factory=list)
    rejected_codes: list[str] = field(default_factory=list)
    quality_score: float | None = None
    sheets_note: str = ""
    feedback_note: str = ""
    log_path: str = ""


def has_json_object(text: str) -> bool:
    start = text.find("{")
    end = text.rfind("}")
    return start != -1 and end != -1 and end > start


def preview_text(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    collapsed = " ".join((text or "").split())
    if not collapsed:
        return "（空）"
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "…"


def is_empty_llm_response(exc: BaseException) -> bool:
    return "Invalid response from LLM call" in str(exc)


def collect_task_raws(crew: object | None, result: object | None) -> list[str]:
    """Full text of each task, in pipeline order."""
    return [_task_raw(item) for item in _task_items(crew, result)]


def collect_task_snapshots(crew: object | None, result: object | None) -> list[TaskSnapshot]:
    """Read per-task output from a CrewAI result, falling back to crew.tasks."""
    snapshots: list[TaskSnapshot] = []
    for index, item in enumerate(_task_items(crew, result)):
        raw = _task_raw(item)
        agent = _agent_label(item)
        stage = (
            STAGE_LABELS[index]
            if index < len(STAGE_LABELS)
            else (agent or f"task_{index + 1}")
        )
        snapshots.append(
            TaskSnapshot(
                index=index + 1,
                stage=stage,
                agent=agent,
                chars=len(raw),
                has_json=has_json_object(raw),
                preview=preview_text(raw),
            )
        )
    return snapshots


def write_task_output_log(
    *,
    tasks: list[TaskSnapshot],
    raws: list[str],
    final_output: str,
    error: BaseException | None = None,
    rejected_outputs: list[str] | None = None,
    logs_dir: Path | None = None,
) -> Path:
    """Persist full task text when the final JSON could not be used as-is."""
    if logs_dir is None:
        from growth_stock_search_agent.config import LOGS_DIR

        logs_dir = LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = logs_dir / f"task_outputs_{timestamp}.json"
    recorded = []
    for index, task in enumerate(tasks):
        raw = raws[index] if index < len(raws) else ""
        recorded.append(
            {
                "index": task.index,
                "stage": task.stage,
                "agent": task.agent,
                "chars": len(raw),
                "has_json": has_json_object(raw),
                "raw": raw,
            }
        )
    payload = {
        "error_type": type(error).__name__ if error else "",
        "error_message": str(error) if error else "",
        "final_output": final_output,
        "tasks": recorded,
        "rejected_llm_outputs": list(rejected_outputs or []),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def summary_from_report(
    report: ResearchReport,
    *,
    attempts: int,
    max_attempts: int,
    tasks: list[TaskSnapshot],
    raw_output: str,
) -> RunSummary:
    candidate_codes = {candidate.code.strip() for candidate in report.candidates}
    passing = [
        item.code
        for item in report.evaluation.stock_evaluations
        if item.passes_criteria and item.code.strip() in candidate_codes
    ]
    rejected = list(report.evaluation.rejected_codes)
    if passing:
        headline = f"成功 — 合格銘柄 {len(passing)} 件"
        diagnosis = (
            "Crew の最終出力を ResearchReport として解釈でき、"
            f"合格銘柄が {len(passing)} 件あります。"
        )
        status = RunStatus.success
    else:
        headline = "レポートは生成されたが、合格銘柄は 0 件"
        diagnosis = (
            "最終JSONのパース自体は成功しています。"
            "候補が空、または評価・身元チェックですべて不合格になったため、"
            "Spreadsheet へ追記する銘柄がありません。"
        )
        if not report.candidates:
            diagnosis += " candidates 配列は空でした。"
        status = RunStatus.no_pass
    return RunSummary(
        status=status,
        headline=headline,
        diagnosis=diagnosis,
        attempts=attempts,
        max_attempts=max_attempts,
        tasks=tasks,
        final_output_preview=preview_text(raw_output, _FINAL_PREVIEW_LIMIT),
        candidate_count=len(report.candidates),
        passing_codes=passing,
        rejected_codes=rejected,
        quality_score=report.evaluation.report_quality_score,
    )


def summary_from_failure(
    exc: BaseException,
    *,
    attempts: int,
    max_attempts: int,
    tasks: list[TaskSnapshot],
    raw_output: str,
) -> RunSummary:
    status = _failure_status(exc)
    return RunSummary(
        status=status,
        headline=_failure_headline(status),
        diagnosis=diagnose_failure(exc, tasks=tasks, raw_output=raw_output),
        attempts=attempts,
        max_attempts=max_attempts,
        error_type=type(exc).__name__,
        error_message=str(exc),
        tasks=tasks,
        final_output_preview=preview_text(raw_output, _FINAL_PREVIEW_LIMIT),
        sheets_note="未書き込み（レポート未生成）",
    )


def diagnose_failure(
    exc: BaseException,
    *,
    tasks: list[TaskSnapshot],
    raw_output: str,
) -> str:
    lines: list[str] = []
    message = str(exc)
    last = tasks[-1] if tasks else None

    if is_empty_llm_response(exc) or "空の応答" in message:
        lines.append(
            "Ollama が空の応答を返しました。"
            " gemma4 は thinking だけ出して content が空になることがあります。"
        )
        lines.append("レポートは生成されていません。候補の有無以前の失敗です。")
        return "\n".join(lines)

    if "No JSON object found" in message or "Empty output" in message:
        lines.append(
            "最終出力から ResearchReport の JSON（{ ... }）を取り出せませんでした。"
        )
        lines.append(
            "これは「候補データが0件」ではなく、出力形式の問題です。"
            " 候補ゼロなら空の candidates 配列を含む JSON が返る想定です。"
        )
        if not (raw_output or "").strip():
            lines.append("最終出力自体が空でした。")
        elif last is not None and not last.has_json:
            lines.append(
                f"最終タスク「{last.stage}」は {last.chars} 文字出力しましたが、"
                "JSON オブジェクトが含まれていません。"
            )
        empty_stages = [task.stage for task in tasks if task.chars == 0]
        json_stages = [task.stage for task in tasks if task.has_json]
        if empty_stages:
            lines.append("出力が空だった段階: " + "、".join(empty_stages))
        if json_stages:
            lines.append(
                "JSON があった段階: "
                + "、".join(json_stages)
                + "（最終段階で失われたか、自然文に置き換わった可能性）"
            )
        elif tasks and all(task.chars == 0 for task in tasks):
            lines.append(
                "全タスクの出力が空です。検索または LLM が最初から失敗した可能性があります。"
            )
        elif not tasks:
            lines.append(
                "各タスクの出力を取得できませんでした。"
                " Crew の kickoff が完了前に失敗した可能性があります。"
            )
        return "\n".join(lines)

    if isinstance(exc, json.JSONDecodeError) or type(exc).__name__ == "JSONDecodeError":
        return (
            "JSON らしい文字列はありましたが、構文が壊れていてパースできませんでした。"
            " 出力が途中で切れたか、JSON 以外の文字が混入した可能性があります。"
        )

    if type(exc).__name__ == "ValidationError":
        return (
            "JSON は見つかりましたが、ResearchReport のスキーマと一致しませんでした。"
            f" {message}"
        )

    return f"リサーチ実行中に予期しないエラーが発生しました: {type(exc).__name__}: {message}"


def format_run_summary(summary: RunSummary) -> str:
    status_label = {
        RunStatus.success: "成功",
        RunStatus.no_pass: "完了（合格0件）",
        RunStatus.quality_skip: "完了（品質閾値未満で未書き込み）",
        RunStatus.parse_error: "失敗",
        RunStatus.empty_llm: "失敗",
        RunStatus.error: "失敗",
    }[summary.status]
    lines = [
        "",
        "=" * 60,
        "  実行総括",
        "=" * 60,
        f"結果: {status_label} — {summary.headline}",
        f"試行回数: {summary.attempts}/{summary.max_attempts}",
        "",
        "何が起きたか:",
        *(f"  {line}" if line else "" for line in summary.diagnosis.splitlines()),
    ]

    if summary.candidate_count is not None:
        passing = ", ".join(summary.passing_codes) or "なし"
        rejected = ", ".join(summary.rejected_codes) or "なし"
        lines.append("")
        lines.append(f"候補数: {summary.candidate_count}")
        lines.append(f"合格コード: {passing}")
        lines.append(f"除外コード: {rejected}")
        if summary.quality_score is not None:
            lines.append(f"品質スコア: {summary.quality_score:.2f}")

    lines.append("")
    lines.append("各エージェントの出力:")
    if summary.tasks:
        last_index = summary.tasks[-1].index
        for task in summary.tasks:
            json_mark = "JSONあり" if task.has_json else "JSONなし"
            suffix = "  ← 最終" if task.index == last_index else ""
            lines.append(
                f"  {task.index}. {task.stage}  {task.chars}字  {json_mark}{suffix}"
            )
            if task.preview and task.preview != "（空）":
                lines.append(f"     先頭: {task.preview}")
            elif task.chars == 0:
                lines.append("     先頭: （空）")
    else:
        lines.append("  （タスク出力を取得できませんでした）")

    if summary.status in {RunStatus.parse_error, RunStatus.empty_llm, RunStatus.error}:
        lines.append("")
        lines.append("最終出力（抜粋）:")
        lines.append(f"  {summary.final_output_preview or '（空）'}")

    if summary.error_message:
        lines.append("")
        lines.append(f"例外: {summary.error_type}: {summary.error_message}")

    if summary.sheets_note:
        lines.append("")
        lines.append(f"Sheets: {summary.sheets_note}")
    if summary.feedback_note:
        lines.append(f"フィードバック: {summary.feedback_note}")
    if summary.log_path:
        lines.append(f"ログ: {summary.log_path}")

    lines.append("=" * 60)
    return "\n".join(lines)


def summary_to_dict(summary: RunSummary) -> dict:
    payload = asdict(summary)
    payload["status"] = summary.status.value
    return payload


def _failure_status(exc: BaseException) -> RunStatus:
    message = str(exc)
    if is_empty_llm_response(exc) or "空の応答" in message:
        return RunStatus.empty_llm
    if isinstance(exc, ValueError) or type(exc).__name__ in {
        "ValidationError",
        "JSONDecodeError",
    }:
        return RunStatus.parse_error
    return RunStatus.error


def _failure_headline(status: RunStatus) -> str:
    if status is RunStatus.empty_llm:
        return "LLM が空応答を返し、レポート未生成"
    if status is RunStatus.parse_error:
        return "最終JSONを解釈できず、レポート未生成"
    return "実行エラーのためレポート未生成"


def _task_items(crew: object | None, result: object | None) -> list[object]:
    tasks_output = getattr(result, "tasks_output", None) if result is not None else None
    if tasks_output:
        return list(tasks_output)
    if crew is None:
        return []
    items: list[object] = []
    for task in getattr(crew, "tasks", None) or []:
        output = getattr(task, "output", None)
        items.append(output if output is not None else task)
    return items


def _task_raw(item: object | None) -> str:
    if item is None:
        return ""
    raw = getattr(item, "raw", None)
    if raw is not None:
        return str(raw)
    output = getattr(item, "output", None)
    if output is not None and output is not item:
        return _task_raw(output)
    return ""


def _agent_label(item: object | None) -> str:
    if item is None:
        return ""
    agent = getattr(item, "agent", "") or ""
    role = getattr(agent, "role", None)
    if role:
        return str(role)
    return str(agent)
