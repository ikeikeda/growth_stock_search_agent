from __future__ import annotations

from crewai import Task

from growth_stock_search_agent.crew.agents import (
    create_analyst_agent,
    create_evaluator_agent,
    create_ranker_agent,
    create_researcher_agent,
)
from growth_stock_search_agent.crew.evaluation_rubric import format_rubric_for_prompt


RANKER_JSON_SCHEMA = """
{
  "run_date": "実行時に上書きされる。プロンプトの【実行日時】を使う",
  "candidates": [
    {
      "rank": 1,
      "name": "正式社名（コード不可。株探/Yahooで確認した実在企業のみ。例: ラクスル）",
      "code": "4桁の実在コード（重複禁止）",
      "business_description": "事業内容（何の会社か1〜2文）",
      "current_price": "現在株価",
      "market_cap": "時価総額",
      "forecast_per": "予想PER",
      "revenue_growth": "売上高成長率",
      "operating_profit_growth": "営業利益成長率",
      "undervalued_reason": "割安理由",
      "unnoticed_reason": "未注目理由",
      "growth_drivers": "成長材料",
      "risks": "主なリスク",
      "is_top3": false
    }
  ],
  "top3_comparison": "特に有望な3銘柄の比較"
}
"""

EVALUATOR_JSON_SCHEMA = """
{
  "run_date": "実行時に上書きされる。プロンプトの【実行日時】を使う",
  "candidates": [ /* passes_criteria=true の銘柄のみ。Rankerと同スキーマ */ ],
  "top3_comparison": "Top3比較（合格銘柄ベース）",
  "evaluation": {
    "stock_evaluations": [
      {
        "code": "4桁コード",
        "passes_criteria": true,
        "growth_score": 0.0,
        "valuation_score": 0.0,
        "unnoticed_score": 0.0,
        "exclusion_check_passed": true,
        "data_freshness_ok": true,
        "issues": ["懸念点があれば記載"],
        "overall_score": 0.0
      }
    ],
    "report_quality_score": 0.0,
    "purpose_alignment_summary": "目的適合性の総評",
    "rejected_codes": ["不合格コード"],
    "recommendations": "次回リサーチへの改善提案"
  }
}
"""


def create_research_task(researcher, research_prompt: str) -> Task:
    return Task(
        description=(
            f"{research_prompt}\n\n"
            "【あなたの担当】 上記目的に沿い、Web検索で日本株の成長株候補を15〜20銘柄程度収集してください。"
            "各候補について正式社名（コードを社名代わりにしない）・4桁コード・事業内容・"
            "情報源URL・直近決算概要を整理してください。"
            "社名とコードは株探またはYahooファイナンスで同一企業と確認できた組だけを使う。"
            "創作社名・コードの付け替え・同一コードの重複は禁止。"
            "必ず最新情報を検索し、一次情報を優先してください。"
        ),
        expected_output=(
            "候補銘柄リスト（正式社名、コード、事業内容、情報源URL、"
            "直近決算の売上・営業利益成長率の概要、PER概算）"
        ),
        agent=researcher,
    )


def create_analysis_task(analyst, research_task: Task) -> Task:
    return Task(
        description=(
            "Researcherが収集した候補銘柄について、一次情報を抽出・検証してください。\n"
            "最初の応答では長文の最終分析を書かず、まず検索またはページ抽出ツールを使ってください。\n"
            "各候補の株探またはYahooファイナンス、可能ならIR・決算短信をツールで確認すること。\n"
            "・PER15倍未満（低PER業種は同業比較）\n"
            "・売上・営業利益YoY10%以上（20%以上優先）\n"
            "・赤字・希薄化・反動増は除外\n"
            "・直近決算と現在株価でPERを再計算\n"
            "条件を満たす銘柄を10〜15銘柄に絞り込み、各銘柄の詳細データを整理してください。\n"
            "社名とコードが一致しない候補、上場確認できないコードは除外してください。"
        ),
        expected_output=(
            "絞り込み後の候補銘柄リスト（正式社名、コード、事業内容、現在株価、時価総額、"
            "予想PER、売上高成長率、営業利益成長率、割安理由、未注目理由、"
            "成長材料、リスク、情報源）"
        ),
        agent=analyst,
        context=[research_task],
    )


def create_ranking_task(ranker, analysis_task: Task) -> Task:
    return Task(
        description=(
            "Analystの分析結果をもとに、候補を10銘柄程度に順位付けし、"
            "特に有望な3銘柄を選定してください。\n"
            "name は株探/Yahooで確認した正式社名のみ（4桁コードを銘柄名に入れない）。"
            "code は実在する4桁コードで、同一コードを複数行に使わない。"
            "business_description に事業内容を1〜2文で必ず入れること。\n"
            "前置き・説明・思考過程は書かない。出力はJSONオブジェクト1つのみ。\n"
            f"{RANKER_JSON_SCHEMA}"
        ),
        expected_output="有効なJSON形式のRankerOutput（candidates + top3_comparison）",
        agent=ranker,
        context=[analysis_task],
    )


def create_evaluation_task(evaluator, ranking_task: Task) -> Task:
    rubric = format_rubric_for_prompt()
    return Task(
        description=(
            f"{rubric}\n\n"
            "RankerのJSON出力を受け取り、各銘柄が本来の目的に合致しているか"
            "ルーブリックに沿って独立して評価してください。\n"
            "・passes_criteria=false の銘柄は candidates から除外\n"
            "・社名とコードが実在上場企業として一致しない場合は不合格\n"
            "・各銘柄は1回だけ判定し、同じ確認を繰り返さない\n"
            "・report_quality_score は合格銘柄の割合とスコア平均から算出\n"
            "・合格が0件でも candidates を空配列にしたJSONを必ず返す\n"
            "前置き・箇条書き・思考の再掲は禁止。最終出力はJSONオブジェクト1つのみ。\n"
            f"{EVALUATOR_JSON_SCHEMA}"
        ),
        expected_output="有効なJSON形式のResearchReport（evaluation付き、合格銘柄のみ）",
        agent=evaluator,
        context=[ranking_task],
    )


def build_tasks(research_prompt: str):
    from growth_stock_search_agent.crew.agents import (
        build_extractor_tool,
        build_llm,
        build_search_tool,
    )

    llm = build_llm()
    json_llm = build_llm(json_mode=True)
    search_tool = build_search_tool()
    extractor_tool = build_extractor_tool()

    researcher = create_researcher_agent(llm, search_tool)
    analyst = create_analyst_agent(llm, search_tool, extractor_tool)
    ranker = create_ranker_agent(json_llm)
    evaluator = create_evaluator_agent(json_llm)

    research_task = create_research_task(researcher, research_prompt)
    analysis_task = create_analysis_task(analyst, research_task)
    ranking_task = create_ranking_task(ranker, analysis_task)
    evaluation_task = create_evaluation_task(evaluator, ranking_task)

    return [research_task, analysis_task, ranking_task, evaluation_task]
