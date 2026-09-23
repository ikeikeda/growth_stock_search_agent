from __future__ import annotations

from growth_stock_search_agent.config import Settings
from growth_stock_search_agent.feedback.classifier import (
    classify_failures,
    should_record_feedback,
)
from growth_stock_search_agent.feedback.loop import record_and_update
from growth_stock_search_agent.feedback.models import FailureEvent, FailureKind, Lessons
from growth_stock_search_agent.feedback.store import (
    format_identity_lessons,
    format_research_lessons,
    load_failure_records,
    load_lessons,
    merge_lessons,
)
from growth_stock_search_agent.feedback.synthesizer import template_search_hints
from growth_stock_search_agent.models import (
    EvaluationReport,
    ResearchReport,
    StockCandidate,
    StockEvaluation,
    format_run_context,
)


def _settings(**overrides) -> Settings:
    values = {
        "feedback_enabled": True,
        "feedback_unknown_code_threshold": 2,
        "feedback_lessons_max_chars": 1500,
        "feedback_max_banned_pairs": 20,
        "feedback_max_unverified_codes": 20,
        "feedback_max_criteria_notes": 10,
        "feedback_max_search_hints": 10,
    }
    values.update(overrides)
    return Settings.model_construct(**values)


def _evaluation(
    code: str,
    *,
    passes: bool = False,
    issues: list[str] | None = None,
) -> StockEvaluation:
    return StockEvaluation(
        code=code,
        passes_criteria=passes,
        growth_score=0.8 if passes else 0.2,
        valuation_score=0.8 if passes else 0.2,
        unnoticed_score=0.8 if passes else 0.2,
        exclusion_check_passed=passes,
        data_freshness_ok=True,
        issues=issues or [],
        overall_score=0.8 if passes else 0.2,
    )


def _candidate(code: str, name: str = "テスト会社") -> StockCandidate:
    return StockCandidate(
        rank=1,
        name=name,
        code=code,
        current_price="1,000",
        market_cap="100億円",
        forecast_per="10.0",
        revenue_growth="20%",
        operating_profit_growth="25%",
        undervalued_reason="成長率に対してPERが低い",
        unnoticed_reason="カバレッジが薄い",
        growth_drivers="新規事業",
        risks="競争",
    )


def _report(
    *,
    candidates: list[StockCandidate] | None = None,
    evaluations: list[StockEvaluation] | None = None,
    recommendations: str = "一次情報でYoYを確認する",
) -> ResearchReport:
    evaluations = evaluations or []
    passing = any(item.passes_criteria for item in evaluations)
    return ResearchReport(
        run_date="2026-09-16T06:00:00",
        candidates=candidates or [],
        top3_comparison="",
        evaluation=EvaluationReport(
            stock_evaluations=evaluations,
            report_quality_score=0.8 if passing else 0.0,
            purpose_alignment_summary="test",
            rejected_codes=[item.code for item in evaluations if not item.passes_criteria],
            recommendations=recommendations,
        ),
    )


def test_classify_invalid_code_and_mismatch_and_empty() -> None:
    called: list[str] = []

    def lookup(code: str) -> str | None:
        called.append(code)
        raise AssertionError("invalid code must not trigger lookup")

    report = _report(
        candidates=[],
        evaluations=[
            _evaluation("ABC", issues=["銘柄コードが日本株の4〜5桁ではない"]),
            _evaluation(
                "7203",
                issues=["銘柄名とコードが不一致: 報告=架空工業 公式=トヨタ自動車"],
            ),
        ],
    )
    events = classify_failures(report, lookup_name=lookup)
    kinds = [event.kind for event in events]
    assert FailureKind.empty_output in kinds
    assert FailureKind.invalid_code in kinds
    assert FailureKind.name_mismatch in kinds
    mismatch = next(event for event in events if event.kind is FailureKind.name_mismatch)
    assert mismatch.reported_name == "架空工業"
    assert mismatch.official_name == "トヨタ自動車"
    assert called == []


def test_lookup_retry_success_is_outage_not_unknown() -> None:
    report = _report(
        candidates=[],
        evaluations=[
            _evaluation("7203", issues=["上場銘柄として確認できないコード"]),
        ],
    )
    events = classify_failures(report, lookup_name=lambda code: "トヨタ自動車")
    kinds = [event.kind for event in events]
    assert FailureKind.lookup_outage in kinds
    assert FailureKind.unknown_code not in kinds
    outage = next(event for event in events if event.kind is FailureKind.lookup_outage)
    assert outage.official_name == "トヨタ自動車"


def test_lookup_retry_failure_stays_unknown() -> None:
    report = _report(
        candidates=[],
        evaluations=[
            _evaluation("9999", issues=["上場銘柄として確認できないコード"]),
        ],
    )
    events = classify_failures(report, lookup_name=lambda code: None)
    assert any(event.kind is FailureKind.unknown_code for event in events)


def test_passing_candidates_skip_feedback_loop(tmp_path) -> None:
    report = _report(
        candidates=[_candidate("7203", "トヨタ自動車")],
        evaluations=[_evaluation("7203", passes=True)],
    )
    assert should_record_feedback(report) is False
    result = record_and_update(
        report,
        directory=tmp_path,
        lookup_name=lambda code: "トヨタ自動車",
        synthesize_hints=lambda *args, **kwargs: ["should not run"],
        settings=_settings(),
    )
    assert result.recorded is False
    assert not (tmp_path / "failures.jsonl").exists()
    assert not (tmp_path / "lessons.json").exists()


def test_merge_does_not_ban_on_lookup_outage() -> None:
    existing = merge_lessons(
        Lessons(),
        [
            FailureEvent(kind=FailureKind.unknown_code, code="7203"),
        ],
        search_hints=[],
        source_run_date="2026-09-16T06:00:00",
        settings=_settings(),
        now="2026-09-16T06:00:00Z",
    )
    assert existing.unverified_codes[0].banned is False

    updated = merge_lessons(
        existing,
        [
            FailureEvent(
                kind=FailureKind.lookup_outage,
                code="7203",
                official_name="トヨタ自動車",
            ),
        ],
        search_hints=[],
        source_run_date="2026-09-16T07:00:00",
        settings=_settings(),
        now="2026-09-16T07:00:00Z",
    )
    assert updated.unverified_codes == []


def test_unverified_code_banned_after_threshold() -> None:
    settings = _settings(feedback_unknown_code_threshold=2)
    first = merge_lessons(
        Lessons(),
        [FailureEvent(kind=FailureKind.unknown_code, code="9999")],
        search_hints=[],
        source_run_date="2026-09-16T06:00:00",
        settings=settings,
        now="2026-09-16T06:00:00Z",
    )
    assert first.unverified_codes[0].consecutive_failures == 1
    assert first.unverified_codes[0].banned is False
    assert "9999" not in format_research_lessons(first, settings=settings)

    second = merge_lessons(
        first,
        [FailureEvent(kind=FailureKind.unknown_code, code="9999")],
        search_hints=[],
        source_run_date="2026-09-16T07:00:00",
        settings=settings,
        now="2026-09-16T07:00:00Z",
    )
    assert second.unverified_codes[0].banned is True
    text = format_research_lessons(second, settings=settings)
    assert "未確認コード 9999" in text


def test_banned_pairs_are_capped() -> None:
    settings = _settings(feedback_max_banned_pairs=2)
    events = [
        FailureEvent(
            kind=FailureKind.name_mismatch,
            code=f"{1000 + index}",
            reported_name=f"報告{index}",
            official_name=f"公式{index}",
        )
        for index in range(3)
    ]
    lessons = merge_lessons(
        Lessons(),
        events,
        search_hints=[],
        source_run_date="2026-09-16T06:00:00",
        settings=settings,
        now="2026-09-16T06:00:00Z",
    )
    assert len(lessons.banned_pairs) == 2


def test_identity_lessons_omit_search_hints() -> None:
    lessons = Lessons(
        banned_pairs=[],
        unverified_codes=[],
        criteria_notes=[],
        search_hints=["業種の偏りを避ける"],
        updated_at="2026-09-16T06:00:00Z",
        source_run_date="2026-09-16T06:00:00",
    )
    assert format_identity_lessons(lessons) == ""
    research = format_research_lessons(lessons, settings=_settings())
    assert "業種の偏りを避ける" in research


def test_record_and_update_persists_files(tmp_path) -> None:
    report = _report(
        candidates=[],
        evaluations=[
            _evaluation(
                "1234",
                issues=["銘柄名とコードが不一致: 報告=架空工業 公式=実在商事"],
            ),
        ],
        recommendations="検索クエリを変える",
    )
    result = record_and_update(
        report,
        directory=tmp_path,
        lookup_name=lambda code: None,
        synthesize_hints=lambda *args, **kwargs: ["一次情報を増やす"],
        settings=_settings(),
    )
    assert result.recorded is True
    assert result.kind_counts["name_mismatch"] == 1
    records = load_failure_records(tmp_path)
    assert len(records) == 1
    lessons = load_lessons(tmp_path)
    assert lessons.banned_pairs[0].code == "1234"
    assert lessons.banned_pairs[0].reported_name == "架空工業"
    assert "一次情報を増やす" in lessons.search_hints


def test_format_run_context_appends_lessons() -> None:
    text = format_run_context(
        "調査せよ",
        run_date="2026-09-16T06:00:00",
        lessons_text="【前回までの失敗から守ること】\n- 禁止: 報告名「架空」をコード 1234 と組にしない",
    )
    assert "【実行日時】 2026-09-16T06:00:00" in text
    assert "報告名「架空」" in text


def test_template_hints_cover_empty_and_criteria() -> None:
    hints = template_search_hints(
        [
            FailureEvent(kind=FailureKind.empty_output),
            FailureEvent(kind=FailureKind.criteria_fail, issues=["反動増の可能性"]),
        ],
        recommendations="卸売の見かけ低PERに偏らない",
    )
    assert any("検索クエリ" in hint for hint in hints)
    assert "卸売の見かけ低PERに偏らない" in hints
