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
from growth_stock_search_agent.models import (
    EvaluationReport,
    ResearchReport,
    StockCandidate,
    StockEvaluation,
)
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


def _build_sample_report() -> ResearchReport:
    """Sheets書き込みテスト用のサンプルレポートを生成する。"""
    run_date = datetime.now(timezone.utc).isoformat()
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
        appended = append_new_candidates(report)
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
