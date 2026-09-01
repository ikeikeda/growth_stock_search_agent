from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import dspy

from growth_stock_search_agent.models import ResearchReport, extract_json_payload

EVAL_EXAMPLES_PATH = Path(__file__).resolve().parent / "eval_examples.json"


def load_eval_examples() -> list[dict]:
    return json.loads(EVAL_EXAMPLES_PATH.read_text(encoding="utf-8"))


def completeness_score(report: ResearchReport, expected_fields: list[str]) -> float:
    if not report.candidates:
        return 0.0

    total = len(report.candidates) * len(expected_fields)
    if total == 0:
        return 0.0

    hits = 0
    for candidate in report.candidates:
        data = candidate.model_dump()
        for field in expected_fields:
            value = data.get(field, "")
            if str(value).strip():
                hits += 1
    return hits / total


def structure_score(raw_json: str) -> float:
    try:
        payload = extract_json_payload(raw_json)
        ResearchReport.model_validate(payload)
        return 1.0
    except Exception:
        return 0.0


def alignment_score(report: ResearchReport, min_alignment_score: float) -> float:
    score = report.evaluation.report_quality_score
    if score >= min_alignment_score:
        return score
    return score * 0.5


def compute_metric(example: Any, pred: dspy.Prediction, trace=None) -> float:
    del trace

    raw_json = getattr(pred, "stock_report_json", "")
    instruction = getattr(example, "instruction", "")
    expected_fields = getattr(example, "expected_fields", [])
    min_candidates = getattr(example, "min_candidates", 1)
    min_alignment = getattr(example, "min_alignment_score", 0.5)

    struct = structure_score(raw_json)
    if struct == 0.0:
        return 0.0

    try:
        report = ResearchReport.model_validate(extract_json_payload(raw_json))
    except Exception:
        return struct * 0.1

    if len(report.candidates) < min_candidates:
        candidate_factor = len(report.candidates) / max(min_candidates, 1)
    else:
        candidate_factor = 1.0

    complete = completeness_score(report, list(expected_fields))
    align = alignment_score(report, float(min_alignment))
    char_penalty = len(instruction) / 1000.0

    score = (
        complete * 0.4
        + struct * 0.2
        + align * 0.3
        - char_penalty * 0.1
    ) * candidate_factor

    return max(0.0, min(1.0, score))
