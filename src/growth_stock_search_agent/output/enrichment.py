from __future__ import annotations

import re

import httpx

from growth_stock_search_agent.models import (
    MISSING_NAME_PLACEHOLDER,
    EvaluationReport,
    ResearchReport,
    StockCandidate,
    StockEvaluation,
    is_missing_company_name,
    names_match,
    normalize_stock_code,
    now_run_date,
    stamp_run_date,
)

_USER_AGENT = (
    "Mozilla/5.0 (compatible; growth-stock-search-agent/0.1; +https://localhost)"
)
_BAD_NAME_FRAGMENTS = ("株の基本情報", "株探", "まとめ", "http", "https")


def lookup_listed_name(code: str, timeout: float = 10.0) -> str | None:
    """Return the official listed name for a TSE ticker, or None if unknown."""
    digits = normalize_stock_code(code)
    if not digits:
        return None
    return _lookup_kabutan(digits, timeout) or _lookup_yahoo(digits, timeout)


def _clean_company_name(raw: str, code: str) -> str | None:
    text = (raw or "").strip()
    if not text:
        return None
    bracket = re.search(rf"^(.+?)【{re.escape(code)}】", text)
    if bracket:
        text = bracket.group(1).strip()
    text = text.split("|", 1)[0].split("｜", 1)[0].strip()
    text = re.sub(rf"[\s　]*[（(]{re.escape(code)}[)）]$", "", text).strip()
    if any(fragment in text for fragment in _BAD_NAME_FRAGMENTS):
        return None
    if len(text) > 80 or is_missing_company_name(text, code):
        return None
    return text


def _lookup_yahoo(code: str, timeout: float) -> str | None:
    try:
        response = httpx.get(
            "https://query2.finance.yahoo.com/v1/finance/search",
            params={
                "q": f"{code}.T",
                "quotesCount": 5,
                "newsCount": 0,
                "listsCount": 0,
                "enableFuzzy": "false",
                "region": "JP",
                "lang": "ja-JP",
            },
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
        )
        response.raise_for_status()
        quotes = response.json().get("quotes") or []
    except Exception:
        return None

    for quote in quotes:
        symbol = str(quote.get("symbol") or "").upper()
        if symbol not in {f"{code}.T", code}:
            continue
        for key in ("shortname", "longname", "shortName", "longName"):
            cleaned = _clean_company_name(str(quote.get(key) or ""), code)
            if cleaned:
                return cleaned
    return None


def _lookup_kabutan(code: str, timeout: float) -> str | None:
    try:
        response = httpx.get(
            "https://kabutan.jp/stock/",
            params={"code": code},
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
    except Exception:
        return None

    title_match = re.search(r"<title>([^<]+)</title>", response.text, re.IGNORECASE)
    if not title_match:
        return None
    return _clean_company_name(title_match.group(1), code)


def _reject(
    evaluation: StockEvaluation | None,
    code: str,
    reason: str,
    rejected: list[str],
) -> StockEvaluation:
    rejected.append(code)
    if evaluation is None:
        return StockEvaluation(
            code=code,
            passes_criteria=False,
            growth_score=0.0,
            valuation_score=0.0,
            unnoticed_score=0.0,
            exclusion_check_passed=False,
            data_freshness_ok=False,
            issues=[reason],
            overall_score=0.0,
        )
    issues = list(evaluation.issues)
    if reason not in issues:
        issues.append(reason)
    return evaluation.model_copy(
        update={
            "code": code,
            "passes_criteria": False,
            "exclusion_check_passed": False,
            "issues": issues,
        }
    )


def _evaluation_for_code(
    evaluations: list[StockEvaluation],
    code: str,
    original_code: str,
) -> StockEvaluation | None:
    for item in evaluations:
        if item.code.strip() in {code, original_code}:
            return item
    return None


def verify_listed_identities(report: ResearchReport) -> ResearchReport:
    """Drop hallucinated tickers: invalid codes, name/code mismatch, duplicates."""
    kept_candidates: list[StockCandidate] = []
    updated_evaluations: list[StockEvaluation] = []
    rejected: list[str] = list(report.evaluation.rejected_codes)
    seen_codes: set[str] = set()
    name_cache: dict[str, str | None] = {}

    for candidate in report.candidates:
        original_code = candidate.code.strip()
        code = normalize_stock_code(original_code)
        evaluation = _evaluation_for_code(
            report.evaluation.stock_evaluations, code or "", original_code
        )

        if code is None:
            updated_evaluations.append(
                _reject(evaluation, original_code or "(空)", "銘柄コードが日本株の4〜5桁ではない", rejected)
            )
            continue

        if code in seen_codes:
            updated_evaluations.append(
                _reject(evaluation, code, "銘柄コードが重複している", rejected)
            )
            continue

        if code not in name_cache:
            name_cache[code] = lookup_listed_name(code)
        official_name = name_cache[code]

        if not official_name:
            updated_evaluations.append(
                _reject(evaluation, code, "上場銘柄として確認できないコード", rejected)
            )
            continue

        reported_name = candidate.name.strip()
        missing_name = is_missing_company_name(reported_name, code)
        if not missing_name and not names_match(reported_name, official_name):
            updated_evaluations.append(
                _reject(
                    evaluation,
                    code,
                    f"銘柄名とコードが不一致: 報告={reported_name} 公式={official_name}",
                    rejected,
                )
            )
            continue

        seen_codes.add(code)
        kept_candidates.append(
            candidate.model_copy(
                update={
                    "code": code,
                    "name": official_name,
                }
            )
        )
        if evaluation is None:
            updated_evaluations.append(
                StockEvaluation(
                    code=code,
                    passes_criteria=True,
                    growth_score=0.0,
                    valuation_score=0.0,
                    unnoticed_score=0.0,
                    exclusion_check_passed=True,
                    data_freshness_ok=True,
                    issues=[],
                    overall_score=0.0,
                )
            )
        else:
            updated_evaluations.append(evaluation.model_copy(update={"code": code}))

    kept_candidates = [
        candidate.model_copy(
            update={
                "rank": index,
                "is_top3": index <= 3,
            }
        )
        for index, candidate in enumerate(kept_candidates, start=1)
    ]

    unique_rejected = [
        code
        for code in dict.fromkeys(rejected)
        if code and code not in {candidate.code for candidate in kept_candidates}
    ]
    evaluation = report.evaluation
    if kept_candidates:
        passing_scores = [
            item.overall_score
            for item in updated_evaluations
            if item.passes_criteria
        ]
        quality = (
            sum(passing_scores) / len(passing_scores)
            if passing_scores
            else evaluation.report_quality_score
        )
    else:
        quality = 0.0

    return report.model_copy(
        update={
            "candidates": kept_candidates,
            "evaluation": EvaluationReport(
                stock_evaluations=updated_evaluations,
                report_quality_score=quality,
                purpose_alignment_summary=evaluation.purpose_alignment_summary,
                rejected_codes=unique_rejected,
                recommendations=evaluation.recommendations,
            ),
        }
    )


def prepare_report(
    report: ResearchReport,
    *,
    verify_identity: bool = True,
    run_date: str | None = None,
) -> ResearchReport:
    """Stamp the real run time, then optionally verify listed identities."""
    stamped = stamp_run_date(report, run_date or now_run_date())
    if not verify_identity:
        return stamped
    return verify_listed_identities(stamped)


def enrich_report(report: ResearchReport) -> ResearchReport:
    """Fill missing names for already-verified candidates."""
    updated = []
    for candidate in report.candidates:
        name = candidate.name
        if is_missing_company_name(name, candidate.code):
            looked_up = lookup_listed_name(candidate.code)
            name = looked_up or MISSING_NAME_PLACEHOLDER
        updated.append(candidate.model_copy(update={"name": name}))
    return report.model_copy(update={"candidates": updated})
