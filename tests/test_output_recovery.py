from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from growth_stock_search_agent.crew.ollama_llm import (
    promote_reasoning_to_content,
    reasoning_is_usable_answer,
    strip_channel_thoughts,
    task_expects_json,
)
from growth_stock_search_agent.models import (
    RANKER_RECOVERY_NOTE,
    EvaluationReport,
    ResearchReport,
    StockCandidate,
    StockEvaluation,
    extract_json_payload,
    resolve_crew_report,
)
from growth_stock_search_agent.run_summary import (
    TaskSnapshot,
    write_task_output_log,
)


def _candidate_payload(code: str = "4384", name: str = "ラクスル") -> dict:
    return {
        "rank": 1,
        "name": name,
        "code": code,
        "business_description": "印刷",
        "current_price": "1000",
        "market_cap": "100億円",
        "forecast_per": "12",
        "revenue_growth": "20%",
        "operating_profit_growth": "25%",
        "undervalued_reason": "PERが低い",
        "unnoticed_reason": "出来高が小さい",
        "growth_drivers": "新規事業",
        "risks": "競争",
        "is_top3": True,
    }


def _ranker_json(code: str = "4384", name: str = "ラクスル") -> str:
    payload = {
        "run_date": "2024-05-23",
        "candidates": [_candidate_payload(code, name)],
        "top3_comparison": "比較",
    }
    return "<channel|>```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"


def _report_json() -> str:
    report = ResearchReport(
        run_date="2026-09-23T09:00:00+09:00",
        candidates=[StockCandidate.model_validate(_candidate_payload())],
        top3_comparison="比較",
        evaluation=EvaluationReport(
            stock_evaluations=[
                StockEvaluation(
                    code="4384",
                    passes_criteria=True,
                    growth_score=0.8,
                    valuation_score=0.8,
                    unnoticed_score=0.7,
                    exclusion_check_passed=True,
                    data_freshness_ok=True,
                    overall_score=0.8,
                )
            ],
            report_quality_score=0.8,
            purpose_alignment_summary="適合",
            rejected_codes=[],
            recommendations="継続",
        ),
    )
    return json.dumps(report.model_dump(), ensure_ascii=False)


def _message(content: str, reasoning: str, tool_calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    reasoning_content=reasoning,
                    tool_calls=tool_calls,
                    model_extra=None,
                )
            )
        ]
    )


def test_extract_json_reads_nested_object_after_channel_prefix() -> None:
    payload = extract_json_payload(_ranker_json())
    assert payload["candidates"][0]["code"] == "4384"
    assert payload["candidates"][0]["name"] == "ラクスル"


def test_resolve_uses_ranker_when_evaluator_has_no_json() -> None:
    audit = (
        "Audit the provided list of stocks based on specific criteria. "
        "Rank 1: 株式会社マレ (4891)"
    )
    report, note = resolve_crew_report(
        audit,
        ["候補メモ", "<|channel>thought " * 10, _ranker_json("7203", "トヨタ自動車"), audit],
    )
    assert note == RANKER_RECOVERY_NOTE
    assert report.candidates[0].code == "7203"
    assert report.evaluation.report_quality_score == 0.0
    assert report.evaluation.stock_evaluations[0].passes_criteria is True
    assert report.evaluation.stock_evaluations[0].overall_score == 0.0


def test_resolve_prefers_ranker_over_an_earlier_json_blob() -> None:
    report, note = resolve_crew_report(
        "最終出力にJSONなし",
        [
            _ranker_json("9999", "デコイ"),
            "分析メモ",
            _ranker_json("4384", "ラクスル"),
            "監査メモ",
        ],
    )
    assert note
    assert report.candidates[0].code == "4384"


def test_resolve_keeps_valid_final_report() -> None:
    report, note = resolve_crew_report(
        _report_json(),
        ["", "", _ranker_json("9999", "別会社"), _report_json()],
    )
    assert note == ""
    assert report.candidates[0].code == "4384"
    assert report.evaluation.report_quality_score == 0.8


def test_resolve_raises_when_no_task_has_json() -> None:
    with pytest.raises(ValueError, match="No JSON object found"):
        resolve_crew_report("監査メモだけ", ["メモ", "thought", "まだ文章"])


def test_thought_loop_is_not_promoted() -> None:
    reasoning = "<|channel>thought " * 12
    response = _message("", reasoning)
    assert promote_reasoning_to_content(response, require_json=False) is False
    assert response.choices[0].message.content == ""
    assert reasoning_is_usable_answer(reasoning, require_json=False) is False


def test_json_reasoning_is_promoted_for_json_tasks() -> None:
    reasoning = "<|channel>thought\n" + _ranker_json()
    response = _message("", reasoning)
    assert promote_reasoning_to_content(response, require_json=True) is True
    content = response.choices[0].message.content
    assert "<|channel>thought" not in content
    assert extract_json_payload(content)["candidates"][0]["code"] == "4384"


def test_prose_reasoning_is_not_an_answer_for_json_tasks() -> None:
    reasoning = "Audit the list and explain each criterion in English."
    response = _message("", reasoning)
    assert promote_reasoning_to_content(response, require_json=True) is False
    assert response.choices[0].message.content == ""


def test_prose_reasoning_is_promoted_for_research_tasks() -> None:
    reasoning = "日本株市場リサーチャーとして、候補を15銘柄集めました。売上成長は20%です。"
    response = _message("", reasoning)
    assert promote_reasoning_to_content(response, require_json=False) is True
    assert response.choices[0].message.content == reasoning


def test_tool_calls_are_left_untouched() -> None:
    response = _message("", "<|channel>thought", tool_calls=[{"name": "search"}])
    assert promote_reasoning_to_content(response, require_json=True) is False
    assert response.choices[0].message.content == ""


def test_strip_channel_thoughts_keeps_the_answer() -> None:
    cleaned = strip_channel_thoughts("<channel|>```json\n{\"a\": 1}\n```")
    assert cleaned.startswith("```json")
    assert "<channel" not in cleaned


def test_task_expects_json_reads_expected_output() -> None:
    assert task_expects_json(SimpleNamespace(expected_output="有効なJSON形式のResearchReport"))
    assert not task_expects_json(SimpleNamespace(expected_output="候補銘柄リスト"))
    assert not task_expects_json(None)


def test_write_task_output_log_keeps_full_text(tmp_path) -> None:
    raw = "監査メモ" * 500
    path = write_task_output_log(
        tasks=[
            TaskSnapshot(
                index=4,
                stage="Evaluator（最終レポートJSON）",
                agent="リサーチ品質監査者",
                chars=len(raw),
                has_json=False,
                preview=raw[:20],
            )
        ],
        raws=[raw],
        final_output=raw,
        error=ValueError("No JSON object found in output"),
        rejected_outputs=["<|channel>thought " * 5],
        logs_dir=tmp_path,
    )
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["tasks"][0]["raw"] == raw
    assert saved["final_output"] == raw
    assert saved["rejected_llm_outputs"]
    assert saved["error_message"] == "No JSON object found in output"
