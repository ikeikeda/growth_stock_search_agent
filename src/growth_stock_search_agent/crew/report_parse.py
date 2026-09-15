from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from growth_stock_search_agent.crew.evaluation import report_from_ranker
from growth_stock_search_agent.models import (
    RankerOutput,
    ResearchReport,
    extract_json_payload,
    parse_ranker_output,
    parse_research_report,
)

logger = logging.getLogger(__name__)


def _text_blobs(result: Any) -> list[str]:
    blobs: list[str] = []
    raw = getattr(result, "raw", None)
    if raw:
        blobs.append(str(raw))
    for task in reversed(list(getattr(result, "tasks_output", None) or [])):
        task_raw = getattr(task, "raw", None)
        if task_raw:
            blobs.append(str(task_raw))
    if not blobs and result is not None:
        blobs.append(str(result))
    return blobs


def _pydantic_outputs(result: Any) -> list[Any]:
    found: list[Any] = []
    pydantic = getattr(result, "pydantic", None)
    if pydantic is not None:
        found.append(pydantic)
    for task in reversed(list(getattr(result, "tasks_output", None) or [])):
        task_pydantic = getattr(task, "pydantic", None)
        if task_pydantic is not None:
            found.append(task_pydantic)
    return found


def _from_payload(payload: dict, *, allow_ranker_fallback: bool) -> ResearchReport | None:
    if "evaluation" in payload:
        try:
            return ResearchReport.model_validate(payload)
        except ValidationError:
            logger.warning("JSON has evaluation but is not a valid ResearchReport")
    if not allow_ranker_fallback:
        return None
    try:
        ranker = RankerOutput.model_validate(payload)
    except ValidationError:
        return None
    logger.warning("Using Ranker JSON as ResearchReport via programmatic evaluation")
    return report_from_ranker(ranker, source="ranker_json_fallback")


def parse_crew_result(result: Any) -> ResearchReport:
    """Parse crew output, falling back to Ranker JSON if the auditor skipped JSON."""
    for pydantic in _pydantic_outputs(result):
        if isinstance(pydantic, ResearchReport):
            return pydantic
        if isinstance(pydantic, RankerOutput):
            logger.warning("Crew returned RankerOutput; applying programmatic evaluation")
            return report_from_ranker(pydantic, source="ranker_pydantic_fallback")

    last_error: Exception | None = None
    for blob in _text_blobs(result):
        try:
            return parse_research_report(blob)
        except (ValueError, ValidationError) as exc:
            last_error = exc
        try:
            payload = extract_json_payload(blob)
        except ValueError as exc:
            last_error = exc
            continue
        report = _from_payload(payload, allow_ranker_fallback=True)
        if report is not None:
            return report
        last_error = ValueError("JSON object is neither ResearchReport nor RankerOutput")

    raise ValueError(f"Failed to parse crew output as ResearchReport: {last_error}")


def try_parse_ranker_from_result(result: Any) -> RankerOutput | None:
    for pydantic in _pydantic_outputs(result):
        if isinstance(pydantic, RankerOutput):
            return pydantic
    for blob in _text_blobs(result):
        try:
            return parse_ranker_output(blob)
        except (ValueError, ValidationError):
            continue
    return None
