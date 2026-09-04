from __future__ import annotations

import re

import httpx

from growth_stock_search_agent.models import (
    MISSING_NAME_PLACEHOLDER,
    ResearchReport,
    is_missing_company_name,
)

_USER_AGENT = (
    "Mozilla/5.0 (compatible; growth-stock-search-agent/0.1; +https://localhost)"
)
_BAD_NAME_FRAGMENTS = ("株の基本情報", "株探", "まとめ", "http", "https")


def lookup_company_name(code: str, timeout: float = 10.0) -> str | None:
    """Best-effort Japanese company name lookup from ticker code."""
    digits = re.sub(r"\D", "", code or "")
    if len(digits) < 4:
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
        symbol = str(quote.get("symbol") or "")
        if not symbol.upper().startswith(code):
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


def enrich_report(report: ResearchReport) -> ResearchReport:
    """Fill missing company names. Leave business_description to the crew output."""
    updated = []
    for candidate in report.candidates:
        name = candidate.name
        if is_missing_company_name(name, candidate.code):
            looked_up = lookup_company_name(candidate.code)
            name = looked_up or MISSING_NAME_PLACEHOLDER
        updated.append(candidate.model_copy(update={"name": name}))
    return report.model_copy(update={"candidates": updated})
