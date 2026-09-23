from __future__ import annotations

from types import SimpleNamespace

from growth_stock_search_agent.models import (
    EvaluationReport,
    ResearchReport,
    StockCandidate,
    StockEvaluation,
)
from growth_stock_search_agent.run_summary import (
    RunStatus,
    collect_task_snapshots,
    diagnose_failure,
    format_run_summary,
    summary_from_failure,
    summary_from_report,
)


def _candidate(code: str = "7203") -> StockCandidate:
    return StockCandidate(
        rank=1,
        name="トヨタ自動車",
        code=code,
        business_description="自動車",
        current_price="3000",
        market_cap="40兆",
        forecast_per="10",
        revenue_growth="12%",
        operating_profit_growth="15%",
        undervalued_reason="PER低位",
        unnoticed_reason="大型株だが今回の条件に合致",
        growth_drivers="海外販売",
        risks="為替",
        is_top3=True,
    )


def _evaluation(code: str, *, passes: bool) -> StockEvaluation:
    return StockEvaluation(
        code=code,
        passes_criteria=passes,
        growth_score=0.8 if passes else 0.2,
        valuation_score=0.8 if passes else 0.2,
        unnoticed_score=0.8 if passes else 0.2,
        exclusion_check_passed=passes,
        data_freshness_ok=True,
        issues=[] if passes else ["成長率が閾値未満"],
        overall_score=0.8 if passes else 0.2,
    )


def test_no_json_diagnosis_is_not_empty_candidates() -> None:
    tasks = collect_task_snapshots(
        crew=None,
        result=SimpleNamespace(
            tasks_output=[
                SimpleNamespace(raw="銘柄を3件集めました", agent="日本株市場リサーチャー"),
                SimpleNamespace(raw="絞り込み完了", agent="財務アナリスト"),
                SimpleNamespace(raw="該当なし", agent="投資レポート作成者"),
                SimpleNamespace(raw="評価できませんでした", agent="リサーチ品質監査者"),
            ]
        ),
    )
    text = diagnose_failure(
        ValueError("No JSON object found in output"),
        tasks=tasks,
        raw_output="評価できませんでした",
    )
    assert "候補データが0件" in text
    assert "出力形式" in text
    assert "Evaluator" in text or "最終タスク" in text


def test_empty_candidates_report_is_parse_success() -> None:
    report = ResearchReport(
        run_date="2026-09-23T08:00:00",
        candidates=[],
        top3_comparison="",
        evaluation=EvaluationReport(
            stock_evaluations=[],
            report_quality_score=0.0,
            purpose_alignment_summary="該当なし",
            rejected_codes=[],
            recommendations="",
        ),
    )
    summary = summary_from_report(
        report,
        attempts=1,
        max_attempts=1,
        tasks=[],
        raw_output='{"candidates":[]}',
    )
    assert summary.status is RunStatus.no_pass
    assert "パース自体は成功" in summary.diagnosis
    assert "candidates 配列は空" in summary.diagnosis
    rendered = format_run_summary(summary)
    assert "実行総括" in rendered
    assert "合格0件" in rendered


def test_success_summary_lists_passing_codes() -> None:
    report = ResearchReport(
        run_date="2026-09-23T08:00:00",
        candidates=[_candidate("7203")],
        top3_comparison="比較",
        evaluation=EvaluationReport(
            stock_evaluations=[_evaluation("7203", passes=True)],
            report_quality_score=0.8,
            purpose_alignment_summary="合格",
            rejected_codes=[],
            recommendations="",
        ),
    )
    summary = summary_from_report(
        report,
        attempts=1,
        max_attempts=2,
        tasks=[],
        raw_output="{}",
    )
    assert summary.status is RunStatus.success
    assert summary.passing_codes == ["7203"]
    assert "7203" in format_run_summary(summary)


def test_empty_llm_failure_summary() -> None:
    summary = summary_from_failure(
        ValueError("Invalid response from LLM call - None or empty"),
        attempts=2,
        max_attempts=2,
        tasks=[],
        raw_output="",
    )
    assert summary.status is RunStatus.empty_llm
    assert "空" in summary.diagnosis
    rendered = format_run_summary(summary)
    assert "実行総括" in rendered
    assert "最終出力" in rendered


def test_collect_task_snapshots_from_crew_fallback() -> None:
    crew = SimpleNamespace(
        tasks=[
            SimpleNamespace(
                output=SimpleNamespace(raw='{"candidates":[]}', agent="投資レポート作成者")
            )
        ]
    )
    snapshots = collect_task_snapshots(crew, result=None)
    assert len(snapshots) == 1
    assert snapshots[0].has_json is True
    assert snapshots[0].stage.startswith("Researcher")
