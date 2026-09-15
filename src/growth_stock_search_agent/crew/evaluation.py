from __future__ import annotations

from growth_stock_search_agent.models import (
    EvaluationReport,
    RankerOutput,
    ResearchReport,
    StockCandidate,
    StockEvaluation,
    parse_first_number,
)

_EXCLUSION_KEYWORDS = ("赤字", "希薄化", "反動増", "反動")


def _growth_score(revenue_growth: str, operating_profit_growth: str) -> tuple[float, bool, list[str]]:
    revenue = parse_first_number(revenue_growth)
    operating = parse_first_number(operating_profit_growth)
    best = max(value for value in (revenue, operating) if value is not None) if any(
        value is not None for value in (revenue, operating)
    ) else None
    issues: list[str] = []
    if best is None:
        issues.append("成長率を数値として解釈できない")
        return 0.0, False, issues
    if best >= 20:
        return 1.0, True, issues
    if best >= 10:
        return 0.7, True, issues
    issues.append(f"成長率 {best}% がYoY10%以上の基準を満たさない")
    return 0.3, False, issues


def _valuation_score(forecast_per: str) -> tuple[float, bool, list[str]]:
    per = parse_first_number(forecast_per)
    issues: list[str] = []
    if per is None:
        issues.append("予想PERを数値として解釈できない")
        return 0.0, False, issues
    if per < 10:
        return 1.0, True, issues
    if per < 15:
        return 0.7, True, issues
    issues.append(f"予想PER {per} が15倍未満の基準を満たさない")
    return 0.0, False, issues


def _unnoticed_score(market_cap: str) -> tuple[float, bool, list[str]]:
    issues: list[str] = []
    if "兆" in (market_cap or ""):
        issues.append("時価総額が兆円規模で未注目とは言えない")
        return 0.2, False, issues
    return 0.6, True, issues


def _exclusion_passed(candidate: StockCandidate) -> tuple[bool, list[str]]:
    blob = " ".join(
        [
            candidate.undervalued_reason,
            candidate.unnoticed_reason,
            candidate.growth_drivers,
            candidate.risks,
        ]
    )
    hits = [word for word in _EXCLUSION_KEYWORDS if word in blob]
    if hits:
        return False, [f"除外条件の言及あり: {', '.join(hits)}"]
    return True, []


def evaluate_candidate(candidate: StockCandidate) -> StockEvaluation:
    growth_score, growth_ok, growth_issues = _growth_score(
        candidate.revenue_growth, candidate.operating_profit_growth
    )
    valuation_score, valuation_ok, valuation_issues = _valuation_score(
        candidate.forecast_per
    )
    unnoticed_score, unnoticed_ok, unnoticed_issues = _unnoticed_score(
        candidate.market_cap
    )
    exclusion_ok, exclusion_issues = _exclusion_passed(candidate)
    issues = growth_issues + valuation_issues + unnoticed_issues + exclusion_issues
    passes = growth_ok and valuation_ok and unnoticed_ok and exclusion_ok
    overall = round((growth_score + valuation_score + unnoticed_score) / 3, 2)
    return StockEvaluation(
        code=candidate.code,
        passes_criteria=passes,
        growth_score=growth_score,
        valuation_score=valuation_score,
        unnoticed_score=unnoticed_score,
        exclusion_check_passed=exclusion_ok,
        data_freshness_ok=True,
        issues=issues,
        overall_score=overall,
    )


def report_from_ranker(
    ranker: RankerOutput,
    *,
    source: str = "programmatic",
) -> ResearchReport:
    """Build a ResearchReport from Ranker JSON when the auditor did not emit JSON."""
    evaluations = [evaluate_candidate(candidate) for candidate in ranker.candidates]
    passing_codes = {
        item.code for item in evaluations if item.passes_criteria
    }
    passing_candidates = [
        candidate.model_copy(update={"is_top3": False})
        for candidate in ranker.candidates
        if candidate.code in passing_codes
    ]
    passing_candidates = [
        candidate.model_copy(
            update={"rank": index, "is_top3": index <= 3}
        )
        for index, candidate in enumerate(passing_candidates, start=1)
    ]
    passed_scores = [item.overall_score for item in evaluations if item.passes_criteria]
    pass_ratio = (len(passed_scores) / len(evaluations)) if evaluations else 0.0
    avg_score = sum(passed_scores) / len(passed_scores) if passed_scores else 0.0
    quality = round(pass_ratio * avg_score, 2) if passed_scores else 0.0
    rejected = [item.code for item in evaluations if not item.passes_criteria]
    top3 = ranker.top3_comparison if passing_candidates else ""
    return ResearchReport(
        run_date=ranker.run_date,
        candidates=passing_candidates,
        top3_comparison=top3,
        evaluation=EvaluationReport(
            stock_evaluations=evaluations,
            report_quality_score=quality,
            purpose_alignment_summary=(
                f"監査者JSONが得られなかったため {source} でルーブリック判定した。"
                f"合格 {len(passing_candidates)} / {len(evaluations)} 銘柄。"
            ),
            rejected_codes=rejected,
            recommendations=(
                "監査者は思考の繰り返しをせず、判定後ただちにJSONのみを出力すること。"
                "PER15倍未満かつ未注目の実在銘柄に絞ること。"
            ),
        ),
    )
