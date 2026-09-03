from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class StockCandidate(BaseModel):
    rank: int
    name: str
    code: str
    current_price: str
    market_cap: str
    forecast_per: str
    revenue_growth: str
    operating_profit_growth: str
    undervalued_reason: str
    unnoticed_reason: str
    growth_drivers: str
    risks: str
    is_top3: bool = False


class RankerOutput(BaseModel):
    run_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    candidates: list[StockCandidate]
    top3_comparison: str


class StockEvaluation(BaseModel):
    code: str
    passes_criteria: bool
    growth_score: float = Field(ge=0.0, le=1.0)
    valuation_score: float = Field(ge=0.0, le=1.0)
    unnoticed_score: float = Field(ge=0.0, le=1.0)
    exclusion_check_passed: bool
    data_freshness_ok: bool
    issues: list[str] = Field(default_factory=list)
    overall_score: float = Field(ge=0.0, le=1.0)


class EvaluationReport(BaseModel):
    stock_evaluations: list[StockEvaluation]
    report_quality_score: float = Field(ge=0.0, le=1.0)
    purpose_alignment_summary: str
    rejected_codes: list[str] = Field(default_factory=list)
    recommendations: str


class ResearchReport(BaseModel):
    run_date: str
    candidates: list[StockCandidate]
    top3_comparison: str
    evaluation: EvaluationReport


def extract_json_payload(text: str) -> dict:
    """Extract a JSON object from raw LLM output."""
    text = text.strip()
    if not text:
        raise ValueError("Empty output")

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        return json.loads(fence_match.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in output")
    return json.loads(text[start : end + 1])


def parse_research_report(raw_output: str) -> ResearchReport:
    payload = extract_json_payload(raw_output)
    return ResearchReport.model_validate(payload)


def parse_ranker_output(raw_output: str) -> RankerOutput:
    payload = extract_json_payload(raw_output)
    if "evaluation" in payload:
        raise ValueError("Payload looks like ResearchReport, not RankerOutput")
    return RankerOutput.model_validate(payload)


UNAUDITED_FALLBACK_NOTE = "Evaluator が JSON を返さなかったため未監査"


def report_from_unaudited_ranker(ranker: RankerOutput) -> ResearchReport:
    """Wrap Ranker JSON in a ResearchReport with quality 0 so Sheets writes stay skipped."""
    evaluations = [
        StockEvaluation(
            code=candidate.code,
            passes_criteria=False,
            growth_score=0.0,
            valuation_score=0.0,
            unnoticed_score=0.0,
            exclusion_check_passed=False,
            data_freshness_ok=False,
            issues=[UNAUDITED_FALLBACK_NOTE],
            overall_score=0.0,
        )
        for candidate in ranker.candidates
    ]
    return ResearchReport(
        run_date=ranker.run_date,
        candidates=ranker.candidates,
        top3_comparison=ranker.top3_comparison,
        evaluation=EvaluationReport(
            stock_evaluations=evaluations,
            report_quality_score=0.0,
            purpose_alignment_summary=UNAUDITED_FALLBACK_NOTE,
            rejected_codes=[candidate.code for candidate in ranker.candidates],
            recommendations=UNAUDITED_FALLBACK_NOTE,
        ),
    )
