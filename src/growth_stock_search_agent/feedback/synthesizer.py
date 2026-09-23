from __future__ import annotations

import logging
from collections import Counter

import httpx

from growth_stock_search_agent.config import Settings, get_settings
from growth_stock_search_agent.feedback.models import FailureEvent, FailureKind

logger = logging.getLogger(__name__)

_KIND_HINTS = {
    FailureKind.name_mismatch: "社名とコードは株探またはYahooファイナンスで同一企業と確認してから出す",
    FailureKind.unknown_code: "実在しないコードや創作社名を候補にしない。コード入力前に上場ページで社名を取る",
    FailureKind.invalid_code: "銘柄コードは日本株の4〜5桁のみを使う",
    FailureKind.duplicate_code: "同一コードを複数候補に使わない",
    FailureKind.empty_output: "検索クエリを変えて候補を増やす。業種やキーワードの偏りを避ける",
    FailureKind.criteria_fail: "見かけの低PERに偏らず、一次情報で売上・営業利益YoYと除外条件を確認する",
    FailureKind.lookup_outage: "一時的な照会失敗と創作コードを混同しない。再検索で公式社名が取れる銘柄は残す",
}


def template_search_hints(
    events: list[FailureEvent],
    recommendations: str = "",
) -> list[str]:
    """Deterministic next-run hints from failure kinds and evaluator text."""
    hints: list[str] = []
    seen: set[str] = set()
    kinds = {event.kind for event in events}
    for kind in (
        FailureKind.empty_output,
        FailureKind.unknown_code,
        FailureKind.name_mismatch,
        FailureKind.invalid_code,
        FailureKind.duplicate_code,
        FailureKind.criteria_fail,
        FailureKind.lookup_outage,
    ):
        if kind not in kinds:
            continue
        hint = _KIND_HINTS[kind]
        if hint not in seen:
            seen.add(hint)
            hints.append(hint)

    rec = recommendations.strip()
    if rec and rec not in seen:
        hints.append(rec[:240])
    return hints


def synthesize_search_hints(
    events: list[FailureEvent],
    recommendations: str = "",
    *,
    settings: Settings | None = None,
) -> list[str]:
    """Template hints plus an optional compact Ollama summary. Never raises."""
    hints = template_search_hints(events, recommendations)
    try:
        extra = _ollama_summary(events, recommendations, settings or get_settings())
        for line in extra:
            if line not in hints:
                hints.append(line)
    except Exception:
        logger.debug("Feedback hint synthesis fell back to templates", exc_info=True)
    return hints


def _ollama_summary(
    events: list[FailureEvent],
    recommendations: str,
    settings: Settings,
) -> list[str]:
    counts = Counter(event.kind.value for event in events)
    mismatch_lines = [
        f"{event.code}: 報告={event.reported_name} 公式={event.official_name}"
        for event in events
        if event.kind is FailureKind.name_mismatch
    ]
    unknown_codes = [
        event.code for event in events if event.kind is FailureKind.unknown_code and event.code
    ]
    prompt = (
        "以下は日本株の割安成長株リサーチが合格銘柄0件だった失敗です。"
        "次回のWeb検索と銘柄選定で守る短い指示を日本語で3項目以内、各1文で出力してください。"
        "番号・前置き・基準の変更は禁止。PER15倍未満・成長YoY10%以上・創作禁止は維持すること。\n\n"
        f"分類: {dict(counts)}\n"
        f"不一致: {mismatch_lines[:8]}\n"
        f"未確認コード: {unknown_codes[:8]}\n"
        f"Evaluator提案: {recommendations.strip()[:400]}\n"
    )
    url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
    response = httpx.post(
        url,
        json={
            "model": settings.ollama_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 400},
        },
        timeout=45.0,
    )
    response.raise_for_status()
    payload = response.json()
    message = payload.get("message") or {}
    text = (
        (message.get("content") or "")
        or (message.get("thinking") or "")
        or (payload.get("response") or "")
    )
    return _parse_hint_lines(text)


def _parse_hint_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in (text or "").splitlines():
        cleaned = raw.strip().lstrip("・*-").strip()
        cleaned = cleaned.lstrip("0123456789.）) ").strip()
        if 8 <= len(cleaned) <= 200:
            lines.append(cleaned)
        if len(lines) >= 3:
            break
    return lines
