from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FailureKind(str, Enum):
    invalid_code = "invalid_code"
    unknown_code = "unknown_code"
    name_mismatch = "name_mismatch"
    duplicate_code = "duplicate_code"
    criteria_fail = "criteria_fail"
    empty_output = "empty_output"
    lookup_outage = "lookup_outage"


class FailureEvent(BaseModel):
    kind: FailureKind
    code: str = ""
    reported_name: str = ""
    official_name: str = ""
    issues: list[str] = Field(default_factory=list)
    recommendations: str = ""


class FailureRecord(BaseModel):
    run_date: str
    recorded_at: str
    recommendations: str = ""
    purpose_alignment_summary: str = ""
    events: list[FailureEvent] = Field(default_factory=list)
    kind_counts: dict[str, int] = Field(default_factory=dict)


class BannedPair(BaseModel):
    code: str
    reported_name: str
    official_name: str = ""
    hit_count: int = 1
    last_seen: str = ""


class UnverifiedCode(BaseModel):
    code: str
    consecutive_failures: int = 1
    last_seen: str = ""
    banned: bool = False


class CriteriaNote(BaseModel):
    note: str
    hit_count: int = 1
    last_seen: str = ""


class Lessons(BaseModel):
    banned_pairs: list[BannedPair] = Field(default_factory=list)
    unverified_codes: list[UnverifiedCode] = Field(default_factory=list)
    criteria_notes: list[CriteriaNote] = Field(default_factory=list)
    search_hints: list[str] = Field(default_factory=list)
    updated_at: str = ""
    source_run_date: str = ""


class FeedbackUpdateResult(BaseModel):
    recorded: bool
    event_count: int = 0
    kind_counts: dict[str, int] = Field(default_factory=dict)
    lessons_path: str = ""
    skipped_reason: str = ""
