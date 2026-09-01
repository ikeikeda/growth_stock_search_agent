from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from growth_stock_search_agent.config import LOGS_DIR, PROJECT_ROOT, get_settings, run_health_checks
from growth_stock_search_agent.crew.crew import run_research_crew
from growth_stock_search_agent.output.sheets_writer import append_new_candidates
from growth_stock_search_agent.prompts.loader import load_research_prompt


def _save_evaluation_log(report) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = LOGS_DIR / f"evaluation_{timestamp}.json"
    path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def run_check() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    settings = get_settings()
    if settings.tavily_api_key:
        os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    all_ok = True
    for name, ok, message in run_health_checks(settings):
        status = "OK" if ok else "NG"
        print(f"[{status}] {name}: {message}")
        all_ok = all_ok and ok
    return 0 if all_ok else 1


def run_research(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="成長株リサーチエージェント")
    parser.add_argument("--dry-run", action="store_true", help="Sheets書き込みをスキップ")
    parser.add_argument("--force-write", action="store_true", help="品質閾値未満でも書き込み")
    parser.add_argument("--use-base", action="store_true", help="最適化前プロンプトを使用")
    args = parser.parse_args(argv)

    settings = get_settings()
    if settings.tavily_api_key:
        os.environ["TAVILY_API_KEY"] = settings.tavily_api_key

    prompt = load_research_prompt(use_base=args.use_base)

    print("リサーチを開始します...")
    report = run_research_crew(prompt)
    log_path = _save_evaluation_log(report)
    print(f"評価ログを保存しました: {log_path}")

    output = json.dumps(report.model_dump(), ensure_ascii=False, indent=2)
    if args.dry_run:
        print(output)
        print(
            f"\nreport_quality_score={report.evaluation.report_quality_score:.2f} "
            f"(threshold={settings.eval_quality_threshold})"
        )
        return 0

    if (
        report.evaluation.report_quality_score < settings.eval_quality_threshold
        and not args.force_write
    ):
        print(
            "警告: report_quality_score が閾値未満のため Sheets 書き込みをスキップしました。"
            f" score={report.evaluation.report_quality_score:.2f}, "
            f"threshold={settings.eval_quality_threshold}. "
            "強制書き込みは --force-write を使用してください。"
        )
        return 2

    appended = append_new_candidates(report)
    if appended:
        print(f"Spreadsheet に {len(appended)} 件の新規銘柄を追記しました: {', '.join(appended)}")
    else:
        print("追記対象の新規銘柄はありませんでした。")

    return 0


if __name__ == "__main__":
    sys.exit(run_research())
