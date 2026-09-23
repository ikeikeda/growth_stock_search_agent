from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

from growth_stock_search_agent.feedback.models import FailureEvent, FailureKind
from growth_stock_search_agent.models import ResearchReport, StockEvaluation, normalize_stock_code

ISSUE_INVALID = "銘柄コードが日本株の4〜5桁ではない"
ISSUE_UNKNOWN = "上場銘柄として確認できないコード"
ISSUE_MISMATCH_PREFIX = "銘柄名とコードが不一致"
ISSUE_DUPLICATE = "銘柄コードが重複している"

_MISMATCH_RE = re.compile(
    r"銘柄名とコードが不一致:\s*報告=(.+?)\s*公式=(.+)$"
)

LookupName = Callable[[str], str | None]


def has_passing_candidates(report: ResearchReport) -> bool:
    """True when identity-verified candidates include at least one evaluator pass."""
    passing_codes = {
        item.code.strip()
        for item in report.evaluation.stock_evaluations
        if item.passes_criteria
    }
    if not passing_codes:
        return False
    return any(candidate.code.strip() in passing_codes for candidate in report.candidates)


def should_record_feedback(report: ResearchReport) -> bool:
    """Record lessons only when the run found zero passing listed stocks."""
    return not has_passing_candidates(report)


def classify_failures(
    report: ResearchReport,
    *,
    lookup_name: LookupName | None = None,
) -> list[FailureEvent]:
    """Turn a failed research report into structured next-run events."""
    recommendations = report.evaluation.recommendations or ""
    events: list[FailureEvent] = []
    lookup = lookup_name

    if not report.candidates:
        events.append(
            FailureEvent(
                kind=FailureKind.empty_output,
                recommendations=recommendations,
            )
        )

    for evaluation in report.evaluation.stock_evaluations:
        if evaluation.passes_criteria:
            continue
        event = _classify_evaluation(
            evaluation,
            recommendations=recommendations,
            lookup_name=lookup,
        )
        events.append(event)

    return events


def kind_counts(events: list[FailureEvent]) -> dict[str, int]:
    counts = Counter(event.kind.value for event in events)
    return dict(counts)


def _classify_evaluation(
    evaluation: StockEvaluation,
    *,
    recommendations: str,
    lookup_name: LookupName | None,
) -> FailureEvent:
    issues = list(evaluation.issues)
    code = evaluation.code.strip()
    base = {
        "code": code,
        "issues": issues,
        "recommendations": recommendations,
    }

    if any(ISSUE_INVALID in issue for issue in issues):
        return FailureEvent(kind=FailureKind.invalid_code, **base)

    if any(ISSUE_DUPLICATE in issue for issue in issues):
        return FailureEvent(kind=FailureKind.duplicate_code, **base)

    mismatch = next((issue for issue in issues if ISSUE_MISMATCH_PREFIX in issue), None)
    if mismatch:
        reported, official = _parse_mismatch(mismatch)
        return FailureEvent(
            kind=FailureKind.name_mismatch,
            reported_name=reported,
            official_name=official,
            **base,
        )

    if any(ISSUE_UNKNOWN in issue for issue in issues):
        digits = normalize_stock_code(code)
        if digits and lookup_name is not None:
            official = lookup_name(digits)
            if official:
                return FailureEvent(
                    kind=FailureKind.lookup_outage,
                    official_name=official,
                    **base,
                )
        return FailureEvent(kind=FailureKind.unknown_code, **base)

    return FailureEvent(kind=FailureKind.criteria_fail, **base)


def _parse_mismatch(issue: str) -> tuple[str, str]:
    match = _MISMATCH_RE.search(issue)
    if not match:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()
