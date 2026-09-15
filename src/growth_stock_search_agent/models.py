from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

JST = ZoneInfo("Asia/Tokyo")

MISSING_NAME_PLACEHOLDER = "（社名未取得）"
_TICKER_NAME_RE = re.compile(r"^\d{3,5}(?:\.T)?$", re.IGNORECASE)
_MISSING_NAME_LABELS = {"", "-", "不明", "n/a", "na", "none", MISSING_NAME_PLACEHOLDER}
_STOCK_CODE_RE = re.compile(r"^(\d{4,5})(?:\.T)?$", re.IGNORECASE)
_GENERIC_NAME_STEMS = {
    "日本",
    "東京",
    "東洋",
    "大阪",
    "三菱",
    "三井",
    "住友",
    "大和",
    "第一",
    "東北",
    "西日本",
    "東日本",
}
_COMPANY_SUFFIX_RE = re.compile(
    r"株式会社|㈱|（株）|\(株\)|ホールディングス|Holdings|ＨＤ|HD|"
    r"グループ|Group|Corp\.?|Inc\.?|Co\.?,?\s*Ltd\.?",
    re.IGNORECASE,
)


def now_run_date() -> str:
    """Return the actual local run timestamp. Never trust LLM-invented dates."""
    return datetime.now(JST).isoformat(timespec="seconds")


def is_missing_company_name(name: str, code: str) -> bool:
    """True when name is empty, a placeholder, or just the ticker code."""
    normalized_name = (name or "").strip()
    normalized_code = (code or "").strip()
    code_digits = re.sub(r"\.T$", "", normalized_code, flags=re.IGNORECASE)
    if normalized_name.lower() in _MISSING_NAME_LABELS:
        return True
    if normalized_code and normalized_name == normalized_code:
        return True
    if code_digits and normalized_name == code_digits:
        return True
    return bool(_TICKER_NAME_RE.fullmatch(normalized_name))


def normalize_stock_code(code: str) -> str | None:
    """Return a 4-5 digit TSE code, or None if the value is not a ticker."""
    text = unicodedata.normalize("NFKC", (code or "").strip())
    match = _STOCK_CODE_RE.fullmatch(text)
    if match:
        return match.group(1)
    digits = re.sub(r"\D", "", text)
    if len(digits) in {4, 5}:
        return digits
    return None


def fold_company_name(name: str) -> str:
    text = unicodedata.normalize("NFKC", name or "")
    text = _COMPANY_SUFFIX_RE.sub("", text)
    return re.sub(r"[\s　・.．,，\-ー−_]", "", text).casefold()


def names_match(reported: str, official: str) -> bool:
    """True when the LLM name is the same listed company as the official name."""
    reported_fold = fold_company_name(reported)
    official_fold = fold_company_name(official)
    if not reported_fold or not official_fold:
        return False
    if reported_fold == official_fold:
        return True
    shorter, longer = (
        (reported_fold, official_fold)
        if len(reported_fold) <= len(official_fold)
        else (official_fold, reported_fold)
    )
    if shorter in _GENERIC_NAME_STEMS:
        return False
    return len(shorter) >= 3 and shorter in longer


class StockCandidate(BaseModel):
    rank: int
    name: str
    code: str
    business_description: str = ""
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
    run_date: str = Field(default_factory=now_run_date)
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


_CHANNEL_TOKEN_RE = re.compile(r"<\|?/?(?:channel|start|end)[^>]*>", re.IGNORECASE)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _strip_llm_noise(text: str) -> str:
    """Remove chat-template tokens such as ``<channel|>`` from model output."""
    return _CHANNEL_TOKEN_RE.sub("", text).strip()


def _iter_balanced_objects(text: str) -> list[str]:
    """Return balanced ``{...}`` slices, respecting JSON strings."""
    objects: list[str] = []
    i = 0
    length = len(text)
    while i < length:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        for j in range(i, length):
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    objects.append(text[i : j + 1])
                    i = j
                    break
        else:
            break
        i += 1
    return objects


def _json_object_score(payload: dict) -> int:
    score = 0
    if "candidates" in payload:
        score += 2
    if "evaluation" in payload:
        score += 3
    if "top3_comparison" in payload:
        score += 1
    return score


def extract_json_payload(text: str) -> dict:
    """Extract the most report-like JSON object from raw LLM output."""
    text = _strip_llm_noise(text)
    if not text:
        raise ValueError("Empty output")

    decoded: list[dict] = []
    for raw in _iter_balanced_objects(text):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            decoded.append(payload)

    if not decoded:
        raise ValueError("No JSON object found in output")

    best_index, best_payload = max(
        enumerate(decoded),
        key=lambda item: (_json_object_score(item[1]), item[0]),
    )
    del best_index
    return best_payload


def parse_first_number(text: str) -> float | None:
    """Parse the first numeric token from an LLM metric string."""
    cleaned = (text or "").replace(",", "").replace("，", "").replace("％", "%")
    match = _NUMBER_RE.search(cleaned)
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def parse_research_report(raw_output: str) -> ResearchReport:
    payload = extract_json_payload(raw_output)
    return ResearchReport.model_validate(payload)


def parse_ranker_output(raw_output: str) -> RankerOutput:
    payload = extract_json_payload(raw_output)
    return RankerOutput.model_validate(payload)


def stamp_run_date(report: ResearchReport, run_date: str | None = None) -> ResearchReport:
    return report.model_copy(update={"run_date": run_date or now_run_date()})


def format_run_context(research_prompt: str, run_date: str | None = None) -> str:
    """Append the actual run timestamp and identity rules to the research prompt."""
    stamped = run_date or now_run_date()
    return (
        f"{research_prompt}\n\n"
        f"【実行日時】 {stamped}\n"
        "run_date にはこの日時をそのまま使うこと。学習データの古い日付（例: 2024-05-23）を使わないこと。\n"
        "【銘柄の厳守】 実在する日本の上場企業のみを扱う。"
        "銘柄名と4桁コードは株探またはYahooファイナンスで確認した組み合わせだけを使う。"
        "存在しない社名の創作、コードの付け替え、同一コードの重複は禁止。"
    )
