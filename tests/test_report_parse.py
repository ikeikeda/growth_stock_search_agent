from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from growth_stock_search_agent.crew.evaluation import report_from_ranker
from growth_stock_search_agent.crew.report_parse import parse_crew_result
from growth_stock_search_agent.models import (
    RankerOutput,
    StockCandidate,
    extract_json_payload,
    parse_first_number,
    parse_ranker_output,
)


def _candidate(**overrides: object) -> StockCandidate:
    data = {
        "rank": 1,
        "name": "ラクスル",
        "code": "4384",
        "business_description": "印刷のマッチング",
        "current_price": "1,100円",
        "market_cap": "1,500億円",
        "forecast_per": "12.0",
        "revenue_growth": "25.0%",
        "operating_profit_growth": "30.0%",
        "undervalued_reason": "成長に対してPERが低い",
        "unnoticed_reason": "テーマ過熱なし",
        "growth_drivers": "新規事業",
        "risks": "競争激化",
        "is_top3": True,
    }
    data.update(overrides)
    return StockCandidate.model_validate(data)


class ExtractJsonTests(unittest.TestCase):
    def test_nested_object_with_channel_prefix(self) -> None:
        raw = (
            "<channel|>```json\n"
            '{"run_date":"2024-05-23","candidates":[{"rank":1,"name":"ラクスル",'
            '"code":"4384","business_description":"印刷","current_price":"1",'
            '"market_cap":"1","forecast_per":"12","revenue_growth":"20%",'
            '"operating_profit_growth":"25%","undervalued_reason":"低PER",'
            '"unnoticed_reason":"薄い","growth_drivers":"拡大","risks":"競争",'
            '"is_top3":true}],"top3_comparison":"比較"}\n```'
        )
        payload = extract_json_payload(raw)
        self.assertEqual(payload["candidates"][0]["code"], "4384")
        ranker = parse_ranker_output(raw)
        self.assertEqual(ranker.candidates[0].name, "ラクスル")

    def test_prefers_report_with_evaluation(self) -> None:
        raw = (
            '{"candidates":[]}'
            '{"run_date":"x","candidates":[],"top3_comparison":"",'
            '"evaluation":{"stock_evaluations":[],"report_quality_score":0,'
            '"purpose_alignment_summary":"none","rejected_codes":[],'
            '"recommendations":"none"}}'
        )
        payload = extract_json_payload(raw)
        self.assertIn("evaluation", payload)

    def test_no_json_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            extract_json_payload("Wait, I should check the PER again. Fail.")
        self.assertIn("No JSON object found", str(ctx.exception))


class ProgrammaticEvalTests(unittest.TestCase):
    def test_parse_first_number(self) -> None:
        self.assertEqual(parse_first_number("15.2倍"), 15.2)
        self.assertEqual(parse_first_number("12.5%"), 12.5)
        self.assertIsNone(parse_first_number("不明"))

    def test_high_per_is_rejected(self) -> None:
        ranker = RankerOutput(
            run_date="2026-09-15",
            candidates=[_candidate(forecast_per="32.4", code="4385")],
            top3_comparison="比較",
        )
        report = report_from_ranker(ranker)
        self.assertEqual(report.candidates, [])
        self.assertEqual(report.evaluation.rejected_codes, ["4385"])
        self.assertEqual(report.evaluation.report_quality_score, 0.0)

    def test_passing_candidate_kept(self) -> None:
        ranker = RankerOutput(
            run_date="2026-09-15",
            candidates=[_candidate()],
            top3_comparison="比較",
        )
        report = report_from_ranker(ranker)
        self.assertEqual(len(report.candidates), 1)
        self.assertTrue(report.evaluation.stock_evaluations[0].passes_criteria)


class CrewFallbackTests(unittest.TestCase):
    def test_falls_back_to_ranker_task_output(self) -> None:
        ranker_json = json.dumps(
            RankerOutput(
                run_date="2026-09-15",
                candidates=[_candidate(forecast_per="32.4", code="4385")],
                top3_comparison="比較",
            ).model_dump(),
            ensure_ascii=False,
        )
        result = SimpleNamespace(
            raw="The objective is to identify undervalued stocks. Wait, I should check.",
            pydantic=None,
            tasks_output=[
                SimpleNamespace(raw="research notes", pydantic=None, agent="researcher"),
                SimpleNamespace(raw=ranker_json, pydantic=None, agent="ranker"),
                SimpleNamespace(
                    raw="Wait, I should check the rejected_codes list.",
                    pydantic=None,
                    agent="evaluator",
                ),
            ],
        )
        report = parse_crew_result(result)
        self.assertEqual(report.evaluation.rejected_codes, ["4385"])
        self.assertIn("ranker_json_fallback", report.evaluation.purpose_alignment_summary)


if __name__ == "__main__":
    unittest.main()
