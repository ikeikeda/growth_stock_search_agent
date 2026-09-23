from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from growth_stock_search_agent.config import Settings, get_settings
from growth_stock_search_agent.feedback.classifier import (
    classify_failures,
    kind_counts,
    should_record_feedback,
)
from growth_stock_search_agent.feedback.models import (
    FailureRecord,
    FeedbackUpdateResult,
    Lessons,
)
from growth_stock_search_agent.feedback.store import (
    append_failure_record,
    load_lessons,
    merge_lessons,
    save_lessons,
)
from growth_stock_search_agent.feedback.synthesizer import synthesize_search_hints
from growth_stock_search_agent.models import ResearchReport
from growth_stock_search_agent.output.enrichment import lookup_listed_name

LookupName = Callable[[str], str | None]
SynthesizeHints = Callable[..., list[str]]


def record_and_update(
    report: ResearchReport,
    *,
    directory: Path | None = None,
    lookup_name: LookupName | None = None,
    synthesize_hints: SynthesizeHints | None = None,
    settings: Settings | None = None,
) -> FeedbackUpdateResult:
    """Persist a miss and refresh lessons for the next research run."""
    settings = settings or get_settings()
    if not should_record_feedback(report):
        return FeedbackUpdateResult(
            recorded=False,
            skipped_reason="合格銘柄があるためフィードバック対象外",
        )

    lookup = lookup_name or lookup_listed_name
    events = classify_failures(report, lookup_name=lookup)
    counts = kind_counts(events)
    recorded_at = datetime.now(timezone.utc).isoformat()

    record = FailureRecord(
        run_date=report.run_date,
        recorded_at=recorded_at,
        recommendations=report.evaluation.recommendations,
        purpose_alignment_summary=report.evaluation.purpose_alignment_summary,
        events=events,
        kind_counts=counts,
    )
    append_failure_record(record, directory)

    synthesizer = synthesize_hints or synthesize_search_hints
    hints = synthesizer(
        events,
        report.evaluation.recommendations,
        settings=settings,
    )
    existing = load_lessons(directory)
    updated = merge_lessons(
        existing,
        events,
        search_hints=hints,
        source_run_date=report.run_date,
        settings=settings,
        now=recorded_at,
    )
    path = save_lessons(updated, directory)
    return FeedbackUpdateResult(
        recorded=True,
        event_count=len(events),
        kind_counts=counts,
        lessons_path=str(path),
    )


def current_lessons(directory: Path | None = None) -> Lessons:
    return load_lessons(directory)
