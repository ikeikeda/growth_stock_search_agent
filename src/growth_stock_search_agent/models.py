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


RANKER_RECOVERY_NOTE = (
    "Evaluatorの最終出力から ResearchReport の JSON を取り出せなかったため、"
    "RankerのJSONを採用しました。"
    "ルーブリック評価は未実施で、合否は上場身元チェックのみです。"
    "評価スコアは 0 のまま、身元を確認できた銘柄だけを書き込みます。"
)
EVALUATOR_SKIPPED_ISSUE = "EvaluatorのJSONが無く、ルーブリック評価は未実施"


def extract_json_payload(text: str) -> dict:
    """Extract the first valid JSON object from raw LLM output.

    Nested objects inside ```json fences are decoded from the opening brace.
    A non-greedy fence match would stop at the first ``}`` and reject real reports.
    """
    text = text.strip()
    if not text:
        raise ValueError("Empty output")

    decoder = json.JSONDecoder()
    last_error: json.JSONDecodeError | None = None
    start = 0
    while True:
        index = text.find("{", start)
        if index == -1:
            break
        try:
            payload, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError as exc:
            last_error = exc
            start = index + 1
            continue
        if isinstance(payload, dict):
            return payload
        start = index + 1

    if last_error is not None:
        raise last_error
    raise ValueError("No JSON object found in output")
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


def research_report_from_ranker_json(raw_output: str) -> ResearchReport:
    """Build a report from Ranker JSON when Evaluator did not return one.

    Scores stay at 0 so downstream identity checks decide which names remain.
    """
    ranker = RankerOutput.model_validate(extract_json_payload(raw_output))
    evaluations = [
        StockEvaluation(
            code=candidate.code,
            passes_criteria=True,
            growth_score=0.0,
            valuation_score=0.0,
            unnoticed_score=0.0,
            exclusion_check_passed=True,
            data_freshness_ok=False,
            issues=[EVALUATOR_SKIPPED_ISSUE],
            overall_score=0.0,
        )
        for candidate in ranker.candidates
    ]
    return ResearchReport(
        run_date=ranker.run_date,
        candidates=list(ranker.candidates),
        top3_comparison=ranker.top3_comparison,
        evaluation=EvaluationReport(
            stock_evaluations=evaluations,
            report_quality_score=0.0,
            purpose_alignment_summary=RANKER_RECOVERY_NOTE,
            rejected_codes=[],
            recommendations=RANKER_RECOVERY_NOTE,
        ),
    )


def resolve_crew_report(
    final_raw: str,
    task_raws: list[str],
    *,
    ranker_index: int = 2,
) -> tuple[ResearchReport, str]:
    """Parse the final task, or the Ranker task when that JSON is missing.

    Returns the report and a recovery note. The note is empty when the final
    task already contained a valid ResearchReport.
    """
    try:
        return parse_research_report(final_raw), ""
    except (ValueError, json.JSONDecodeError) as original:
        ordered = list(task_raws)
        if 0 <= ranker_index < len(ordered):
            preferred = ordered.pop(ranker_index)
            ordered.insert(0, preferred)
        for raw in ordered:
            if not (raw or "").strip():
                continue
            try:
                return research_report_from_ranker_json(raw), RANKER_RECOVERY_NOTE
            except (ValueError, json.JSONDecodeError):
                continue
        raise original
def parse_ranker_output(raw_output: str) -> RankerOutput:
    payload = extract_json_payload(raw_output)
    return RankerOutput.model_validate(payload)


def stamp_run_date(report: ResearchReport, run_date: str | None = None) -> ResearchReport:
    return report.model_copy(update={"run_date": run_date or now_run_date()})


def format_run_context(
    research_prompt: str,
    run_date: str | None = None,
    lessons_text: str = "",
) -> str:
    """Append the actual run timestamp, identity rules, and optional lessons."""
    stamped = run_date or now_run_date()
    lessons_block = ""
    cleaned_lessons = (lessons_text or "").strip()
    if cleaned_lessons:
        lessons_block = f"\n{cleaned_lessons}\n"
    return (
        f"{research_prompt}\n\n"
        f"【実行日時】 {stamped}\n"
        "run_date にはこの日時をそのまま使うこと。学習データの古い日付（例: 2024-05-23）を使わないこと。\n"
        "【銘柄の厳守】 実在する日本の上場企業のみを扱う。"
        "銘柄名と4桁コードは株探またはYahooファイナンスで確認した組み合わせだけを使う。"
        "存在しない社名の創作、コードの付け替え、同一コードの重複は禁止。"
        f"{lessons_block}"
    )
