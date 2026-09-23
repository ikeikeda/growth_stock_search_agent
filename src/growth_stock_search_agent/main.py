from __future__ import annotations

import os

# Windows cp932 環境での CrewAI ログ出力エラーを防ぐ
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from growth_stock_search_agent.config import LOGS_DIR, PROJECT_ROOT, get_settings, run_health_checks
from growth_stock_search_agent.crew.crew import run_research_crew
from growth_stock_search_agent.feedback.classifier import should_record_feedback
from growth_stock_search_agent.feedback.loop import record_and_update
from growth_stock_search_agent.feedback.store import (
    format_identity_lessons,
    format_research_lessons,
    load_lessons,
)
from growth_stock_search_agent.models import (
    RANKER_RECOVERY_NOTE,
    EvaluationReport,
    ResearchReport,
    StockCandidate,
    StockEvaluation,
    format_run_context,
    now_run_date,
)
from growth_stock_search_agent.output.sheets_writer import append_new_candidates
from growth_stock_search_agent.prompts.loader import load_research_prompt
from growth_stock_search_agent.run_summary import (
    RunStatus,
    RunSummary,
    format_run_summary,
    summary_to_dict,
)


def _save_evaluation_log(report) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = LOGS_DIR / f"evaluation_{timestamp}.json"
    path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _save_run_summary(summary: RunSummary) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = LOGS_DIR / f"run_summary_{timestamp}.json"
    path.write_text(
        json.dumps(summary_to_dict(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _print_run_summary(summary: RunSummary) -> None:
    print(format_run_summary(summary), flush=True)


def _build_sample_report() -> ResearchReport:
    """Sheets書き込みテスト用のサンプルレポートを生成する。"""
    run_date = now_run_date()
    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    candidates = [
        StockCandidate(
            rank=1,
            name="テスト成長A",
            code=f"T{stamp}1",
            business_description="産業用センサーの製造販売",
            current_price="1,250",
            market_cap="350億円",
            forecast_per="11.2",
            revenue_growth="18%",
            operating_profit_growth="25%",
            undervalued_reason="成長率に対してPERが低い",
            unnoticed_reason="売買代金が小さくテーマ過熱なし",
            growth_drivers="新規事業の寄与拡大",
            risks="競争激化",
            is_top3=True,
        ),
        StockCandidate(
            rank=2,
            name="テスト成長B",
            code=f"T{stamp}2",
            business_description="中小企業向けSaaSの企画・運営",
            current_price="890",
            market_cap="120億円",
            forecast_per="9.8",
            revenue_growth="22%",
            operating_profit_growth="30%",
            undervalued_reason="好決算後もPER低位",
            unnoticed_reason="アナリストカバレッジが薄い",
            growth_drivers="海外売上拡大",
            risks="為替変動",
            is_top3=True,
        ),
        StockCandidate(
            rank=3,
            name="テスト除外C",
            code=f"T{stamp}3",
            business_description="不動産仲介",
            current_price="2,100",
            market_cap="800億円",
            forecast_per="14.5",
            revenue_growth="8%",
            operating_profit_growth="5%",
            undervalued_reason="見かけ上の低PER",
            unnoticed_reason="一時的な反動増の可能性",
            growth_drivers="コスト削減効果",
            risks="成長持続性に疑問",
            is_top3=False,
        ),
    ]
    evaluations = [
        StockEvaluation(
            code=candidates[0].code,
            passes_criteria=True,
            growth_score=0.85,
            valuation_score=0.80,
            unnoticed_score=0.75,
            exclusion_check_passed=True,
            data_freshness_ok=True,
            issues=[],
            overall_score=0.80,
        ),
        StockEvaluation(
            code=candidates[1].code,
            passes_criteria=True,
            growth_score=0.90,
            valuation_score=0.85,
            unnoticed_score=0.80,
            exclusion_check_passed=True,
            data_freshness_ok=True,
            issues=[],
            overall_score=0.85,
        ),
        StockEvaluation(
            code=candidates[2].code,
            passes_criteria=False,
            growth_score=0.40,
            valuation_score=0.50,
            unnoticed_score=0.30,
            exclusion_check_passed=False,
            data_freshness_ok=True,
            issues=["成長率が閾値未満", "反動増の可能性"],
            overall_score=0.35,
        ),
    ]
    return ResearchReport(
        run_date=run_date,
        candidates=candidates,
        top3_comparison="テスト用: Aは成長と割安のバランス、Bは高成長で未注目。",
        evaluation=EvaluationReport(
            stock_evaluations=evaluations,
            report_quality_score=0.75,
            purpose_alignment_summary="Sheets書き込みテスト用サンプル",
            rejected_codes=[candidates[2].code],
            recommendations="本番ではリサーチ結果を使用してください",
        ),
    )


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


def run_test_sheets(argv: list[str] | None = None) -> int:
    """リサーチを実行せず、サンプルデータで Spreadsheet 書き込みだけを試す。"""
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Spreadsheet書き込みテスト（リサーチなし）"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="書き込まずサンプルJSONを表示するだけ",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    report = _build_sample_report()

    print("サンプルレポートを生成しました（リサーチは実行していません）")
    print(
        f"  Pass候補: "
        f"{[c.code for c, e in zip(report.candidates, report.evaluation.stock_evaluations) if e.passes_criteria]}"
    )
    print(
        f"  Fail候補: "
        f"{[c.code for c, e in zip(report.candidates, report.evaluation.stock_evaluations) if not e.passes_criteria]}"
    )

    if args.preview:
        print(json.dumps(report.model_dump(), ensure_ascii=False, indent=2))
        return 0

    if not settings.google_sheets_id.strip():
        print("エラー: GOOGLE_SHEETS_ID が未設定です。.env を確認してください。")
        return 1

    try:
        appended = append_new_candidates(report, verify_identity=False)
    except Exception as exc:
        print(f"Spreadsheet 書き込みに失敗しました: {exc}")
        return 1

    if appended:
        print(
            f"Spreadsheet に {len(appended)} 件を追記しました: {', '.join(appended)}"
        )
        print(
            f"（Fail銘柄 {report.evaluation.rejected_codes} は品質ゲートで除外済み）"
        )
    else:
        print(
            "追記対象がありませんでした。"
            "（同じテストコードが既に存在するか、Pass銘柄がありません）"
        )
    return 0


def run_research(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="成長株リサーチエージェント")
    parser.add_argument("--dry-run", action="store_true", help="Sheets書き込みをスキップ")
    parser.add_argument("--force-write", action="store_true", help="品質閾値未満でも書き込み")
    parser.add_argument("--use-base", action="store_true", help="最適化前プロンプトを使用")
    parser.add_argument(
        "--no-feedback",
        action="store_true",
        help="失敗時の教訓更新と次回向け注入をスキップ",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    if settings.tavily_api_key:
        os.environ["TAVILY_API_KEY"] = settings.tavily_api_key

    lessons_text = ""
    identity_lessons = ""
    feedback_on = settings.feedback_enabled and not args.no_feedback
    if feedback_on:
        lessons = load_lessons()
        lessons_text = format_research_lessons(lessons, settings=settings)
        identity_lessons = format_identity_lessons(lessons)

    prompt = format_run_context(
        load_research_prompt(use_base=args.use_base),
        lessons_text=lessons_text,
    )

    print("リサーチを開始します...")
    crew_result = run_research_crew(prompt, identity_lessons=identity_lessons)
    summary = crew_result.summary
    try:
        report = crew_result.report
        if report is None:
            return 1

        log_path = _save_evaluation_log(report)
        print(f"評価ログを保存しました: {log_path}")
        if summary.log_path:
            summary.log_path = f"{summary.log_path}; {log_path}"
        else:
            summary.log_path = str(log_path)
        if feedback_on and should_record_feedback(report):
            result = record_and_update(report, settings=settings)
            counts = (
                ", ".join(
                    f"{kind}={count}" for kind, count in result.kind_counts.items()
                )
                or "なし"
            )
            print("合格銘柄が0件のため、次回向けの教訓を更新しました。")
            print(f"  分類: {counts}")
            print(f"  教訓: {result.lessons_path}")
            summary.feedback_note = f"教訓を更新しました（{counts}）"
        if report.evaluation.rejected_codes:
            print("銘柄身元チェックで除外:")
            for evaluation in report.evaluation.stock_evaluations:
                if not evaluation.passes_criteria and evaluation.issues:
                    print(f"  {evaluation.code}: {'; '.join(evaluation.issues)}")

        output = json.dumps(report.model_dump(), ensure_ascii=False, indent=2)
        if args.dry_run:
            print(output)
            print(
                f"\nreport_quality_score={report.evaluation.report_quality_score:.2f} "
                f"(threshold={settings.eval_quality_threshold})"
            )
            summary.sheets_note = "スキップ（--dry-run）"
            return 0

        recovered_from_ranker = RANKER_RECOVERY_NOTE in (
            report.evaluation.purpose_alignment_summary
            or report.evaluation.recommendations
    try:
        report = run_research_crew(prompt)
    except ValueError as exc:
        print(f"リサーチ結果のJSONパースに失敗したため Spreadsheet へは書き込みません: {exc}")
        return 1
    log_path = _save_evaluation_log(report)
    print(f"評価ログを保存しました: {log_path}")
    if report.evaluation.rejected_codes:
        print("品質ゲートで除外:")
        for evaluation in report.evaluation.stock_evaluations:
            if not evaluation.passes_criteria and evaluation.issues:
                print(f"  {evaluation.code}: {'; '.join(evaluation.issues)}")

    output = json.dumps(report.model_dump(), ensure_ascii=False, indent=2)
    if args.dry_run:
        print(output)
        print(
            f"\nreport_quality_score={report.evaluation.report_quality_score:.2f} "
            f"(threshold={settings.eval_quality_threshold})"
        )
        if (
            not recovered_from_ranker
            and report.evaluation.report_quality_score < settings.eval_quality_threshold
            and not args.force_write
        ):
            print(
                "警告: report_quality_score が閾値未満のため Sheets 書き込みをスキップしました。"
                f" score={report.evaluation.report_quality_score:.2f}, "
                f"threshold={settings.eval_quality_threshold}. "
                "強制書き込みは --force-write を使用してください。"
            )
            summary.status = RunStatus.quality_skip
            summary.headline = (
                f"品質スコア {report.evaluation.report_quality_score:.2f} が"
                f"閾値 {settings.eval_quality_threshold} 未満のため未書き込み"
            )
            summary.sheets_note = "スキップ（品質閾値未満。強制は --force-write）"
            return 2

        appended = append_new_candidates(report, verify_identity=False)
        if appended:
            print(
                f"Spreadsheet に {len(appended)} 件の新規銘柄を追記しました: "
                f"{', '.join(appended)}"
            )
            summary.sheets_note = f"{len(appended)} 件追記（{', '.join(appended)}）"
        else:
            print("追記対象の新規銘柄はありませんでした。")
            summary.sheets_note = "追記なし（既存または合格銘柄なし）"
        return 0
    except Exception as exc:
        summary.status = RunStatus.error
        summary.headline = "後処理中にエラーが発生し、結果を確定できませんでした"
        summary.diagnosis = (
            f"{summary.diagnosis}\n後処理エラー: {type(exc).__name__}: {exc}"
    if not report.candidates:
        print(
            "合格銘柄が0件のため Spreadsheet に追記しませんでした。"
            f" rejected={report.evaluation.rejected_codes or []},"
            f" score={report.evaluation.report_quality_score:.2f}"
        )
        return 2

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
        summary.error_type = type(exc).__name__
        summary.error_message = str(exc)
        summary.sheets_note = summary.sheets_note or "未書き込み（後処理エラー）"
        return 1
    finally:
        summary_path = _save_run_summary(summary)
        if not summary.log_path:
            summary.log_path = str(summary_path)
        elif str(summary_path) not in summary.log_path:
            summary.log_path = f"{summary.log_path}; {summary_path}"
        _print_run_summary(summary)


if __name__ == "__main__":
    sys.exit(run_research())
