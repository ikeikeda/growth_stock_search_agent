from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from growth_stock_search_agent.config import LOGS_DIR, Settings, get_settings
from growth_stock_search_agent.feedback.models import (
    BannedPair,
    CriteriaNote,
    FailureEvent,
    FailureKind,
    FailureRecord,
    Lessons,
    UnverifiedCode,
)
from growth_stock_search_agent.models import now_run_date

IDENTITY_ISSUE_MARKERS = (
    "銘柄コードが日本株の4〜5桁ではない",
    "上場銘柄として確認できないコード",
    "銘柄名とコードが不一致",
    "銘柄コードが重複している",
)


def feedback_dir(base: Path | None = None) -> Path:
    return (base or LOGS_DIR) / "feedback"


def lessons_path(directory: Path | None = None) -> Path:
    return (directory or feedback_dir()) / "lessons.json"


def load_lessons(directory: Path | None = None) -> Lessons:
    path = lessons_path(directory)
    if not path.exists():
        return Lessons()
    try:
        return Lessons.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError):
        return Lessons()


def save_lessons(lessons: Lessons, directory: Path | None = None) -> Path:
    path = lessons_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(lessons.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def append_failure_record(record: FailureRecord, directory: Path | None = None) -> Path:
    path = (directory or feedback_dir()) / "failures.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(record.model_dump_json() + "\n")
    return path


def load_failure_records(directory: Path | None = None) -> list[FailureRecord]:
    path = (directory or feedback_dir()) / "failures.jsonl"
    if not path.exists():
        return []
    records: list[FailureRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(FailureRecord.model_validate(json.loads(line)))
        except (json.JSONDecodeError, ValueError):
            continue
    return records


def merge_lessons(
    existing: Lessons,
    events: list[FailureEvent],
    *,
    search_hints: list[str],
    source_run_date: str,
    settings: Settings | None = None,
    now: str | None = None,
) -> Lessons:
    """Fold new failure events into the compact next-run lesson store."""
    settings = settings or get_settings()
    stamp = now or datetime.now(timezone.utc).isoformat()
    run_date = source_run_date or now_run_date()

    pairs = list(existing.banned_pairs)
    unverified = list(existing.unverified_codes)
    notes = list(existing.criteria_notes)

    for event in events:
        if event.kind is FailureKind.name_mismatch and event.code:
            pairs = _upsert_pair(pairs, event, stamp)
        elif event.kind is FailureKind.unknown_code and event.code:
            unverified = _bump_unverified(
                unverified,
                event.code,
                stamp,
                threshold=settings.feedback_unknown_code_threshold,
            )
        elif event.kind is FailureKind.lookup_outage and event.code:
            unverified = [item for item in unverified if item.code != event.code]
        elif event.kind is FailureKind.criteria_fail:
            for issue in event.issues:
                cleaned = issue.strip()
                if not cleaned or any(marker in cleaned for marker in IDENTITY_ISSUE_MARKERS):
                    continue
                notes = _upsert_note(notes, cleaned, stamp)
        elif event.kind is FailureKind.invalid_code:
            notes = _upsert_note(notes, "4〜5桁でないコードを候補にしない", stamp)

    merged_hints = _merge_hints(existing.search_hints, search_hints)

    pairs = _cap_by_recency(pairs, settings.feedback_max_banned_pairs)
    unverified = _cap_unverified(unverified, settings.feedback_max_unverified_codes)
    notes = _cap_by_recency(notes, settings.feedback_max_criteria_notes)
    merged_hints = merged_hints[: settings.feedback_max_search_hints]

    return Lessons(
        banned_pairs=pairs,
        unverified_codes=unverified,
        criteria_notes=notes,
        search_hints=merged_hints,
        updated_at=stamp,
        source_run_date=run_date,
    )


def format_research_lessons(
    lessons: Lessons,
    *,
    max_chars: int | None = None,
    settings: Settings | None = None,
) -> str:
    """Full lesson block injected into the researcher run context."""
    settings = settings or get_settings()
    limit = settings.feedback_lessons_max_chars if max_chars is None else max_chars
    lines = _lesson_lines(lessons, include_search=True)
    if not lines:
        return ""
    text = "【前回までの失敗から守ること】\n" + "\n".join(lines)
    if len(text) <= limit:
        return text
    return _truncate_lines(lines, limit, heading="【前回までの失敗から守ること】")


def format_identity_lessons(lessons: Lessons) -> str:
    """Short identity-only block for the Analyst task."""
    lines = _lesson_lines(lessons, include_search=False)
    if not lines:
        return ""
    return "【前回までの身元失敗】\n" + "\n".join(lines)


def _lesson_lines(lessons: Lessons, *, include_search: bool) -> list[str]:
    lines: list[str] = []
    for pair in lessons.banned_pairs:
        official = f"（公式は {pair.official_name}）" if pair.official_name else ""
        lines.append(
            f"- 禁止: 報告名「{pair.reported_name}」をコード {pair.code} と組にしない{official}"
        )
    for item in lessons.unverified_codes:
        if not item.banned:
            continue
        lines.append(
            f"- 未確認コード {item.code} は再提出しない。株探/Yahoo で社名が取れてから出す"
        )
    if include_search:
        for note in lessons.criteria_notes:
            lines.append(f"- 除外: {note.note}")
        for hint in lessons.search_hints:
            cleaned = hint.strip()
            if cleaned:
                prefix = "" if cleaned.startswith("- ") else "- 検索: "
                if cleaned.startswith("- "):
                    lines.append(cleaned)
                else:
                    lines.append(f"{prefix}{cleaned}")
    return lines


def _truncate_lines(lines: list[str], limit: int, *, heading: str) -> str:
    kept: list[str] = []
    prefix = heading + "\n"
    for line in lines:
        candidate = prefix + "\n".join(kept + [line])
        if len(candidate) > limit:
            break
        kept.append(line)
    if not kept:
        return heading
    return prefix + "\n".join(kept)


def _upsert_pair(pairs: list[BannedPair], event: FailureEvent, stamp: str) -> list[BannedPair]:
    for index, item in enumerate(pairs):
        if item.code == event.code and item.reported_name == event.reported_name:
            updated = item.model_copy(
                update={
                    "official_name": event.official_name or item.official_name,
                    "hit_count": item.hit_count + 1,
                    "last_seen": stamp,
                }
            )
            return [updated] + pairs[:index] + pairs[index + 1 :]
    return [
        BannedPair(
            code=event.code,
            reported_name=event.reported_name,
            official_name=event.official_name,
            hit_count=1,
            last_seen=stamp,
        )
    ] + pairs


def _bump_unverified(
    items: list[UnverifiedCode],
    code: str,
    stamp: str,
    *,
    threshold: int,
) -> list[UnverifiedCode]:
    for index, item in enumerate(items):
        if item.code == code:
            consecutive = item.consecutive_failures + 1
            updated = item.model_copy(
                update={
                    "consecutive_failures": consecutive,
                    "last_seen": stamp,
                    "banned": consecutive >= threshold,
                }
            )
            return [updated] + items[:index] + items[index + 1 :]
    return [
        UnverifiedCode(
            code=code,
            consecutive_failures=1,
            last_seen=stamp,
            banned=threshold <= 1,
        )
    ] + items


def _upsert_note(notes: list[CriteriaNote], note: str, stamp: str) -> list[CriteriaNote]:
    for index, item in enumerate(notes):
        if item.note == note:
            updated = item.model_copy(
                update={"hit_count": item.hit_count + 1, "last_seen": stamp}
            )
            return [updated] + notes[:index] + notes[index + 1 :]
    return [CriteriaNote(note=note, hit_count=1, last_seen=stamp)] + notes


def _merge_hints(existing: list[str], incoming: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for hint in list(incoming) + list(existing):
        cleaned = hint.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        merged.append(cleaned)
    return merged


def _cap_by_recency(items: list, limit: int) -> list:
    ordered = sorted(
        items,
        key=lambda item: (item.last_seen, item.hit_count),
        reverse=True,
    )
    return ordered[:limit]


def _cap_unverified(items: list[UnverifiedCode], limit: int) -> list[UnverifiedCode]:
    ordered = sorted(
        items,
        key=lambda item: (item.banned, item.last_seen, item.consecutive_failures),
        reverse=True,
    )
    return ordered[:limit]
