from __future__ import annotations

from crewai import Task

from growth_stock_search_agent.crew.agents import (
    create_analyst_agent,
    create_evaluator_agent,
    create_formatter_agent,
    create_ranker_agent,
    create_researcher_agent,
)
from growth_stock_search_agent.crew.evaluation_rubric import format_rubric_for_prompt


RANKER_JSON_SCHEMA = """
{
  "run_date": "ISO8601 datetime",
  "candidates": [
    {
      "rank": 1,
      "name": "正式社名（コード不可。例: ラクスル）",
      "code": "4桁コード",
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
  "run_date": "ISO8601 datetime",
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
            "各候補について正式社名（コードを社名代わりにしない）・コード・事業内容・"
            "情報源URL・直近決算概要を整理してください。"
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
            "・PER15倍未満（低PER業種は同業比較）\n"
            "・売上・営業利益YoY10%以上（20%以上優先）\n"
            "・赤字・希薄化・反動増は除外\n"
            "・直近決算と現在株価でPERを再計算\n"
            "条件を満たす銘柄を10〜15銘柄に絞り込み、各銘柄の詳細データを整理してください。\n"
            "社名がコードのみの場合は検索して正式社名と事業内容を補完してください。"
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
            "name は正式社名のみ（4桁コードを銘柄名に入れない）。"
            "business_description に事業内容を1〜2文で必ず入れること。\n"
            "出力は必ず以下のJSONスキーマに従った有効なJSONのみとしてください。\n"
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
            "【厳守】\n"
            "・Rankerの全候補を評価すること。最後に検索した1銘柄の分析で終えてはいけない\n"
            "・Web検索は数値の再確認にのみ使う。"
            "検索結果の解説や個別銘柄の投資レポートを最終回答にしてはいけない\n"
            "・Markdownの見出し・箇条書きの結論文は禁止。最終出力はJSONのみ\n"
            "・passes_criteria=false の銘柄は candidates から除外\n"
            "・疑義がある数値はWeb検索で再確認\n"
            "・report_quality_score は合格銘柄の割合とスコア平均から算出\n"
            "出力は必ず以下のJSONスキーマに従った有効なJSONのみとしてください。\n"
            f"{EVALUATOR_JSON_SCHEMA}"
        ),
        expected_output="有効なJSON形式のResearchReport（evaluation付き、合格銘柄のみ）",
        agent=evaluator,
        context=[ranking_task],
    )


def create_format_task(formatter, ranking_task: Task, evaluation_task: Task) -> Task:
    return Task(
        description=(
            "RankerのJSONとEvaluatorの評価内容を、以下のスキーマに従った"
            "有効なJSON（ResearchReport）へ変換してください。\n"
            "【厳守】\n"
            "・出力はJSONオブジェクトのみ。Markdown、見出し、解説、コードフェンス前後の文章は禁止\n"
            "・Rankerの全候補を反映すること。最後に言及された1銘柄だけで終わってはいけない\n"
            "・passes_criteria=false の銘柄は candidates から除外し、rejected_codes に入れる\n"
            "・name が空・コードと同じ・数字のみなら、文脈から正式社名を入れて補完する\n"
            "・business_description が空なら、文脈から事業内容を1〜2文で補完する\n"
            "・EvaluatorがMarkdownや単一銘柄メモしか出していない場合でも、"
            "Ranker JSONを正として全候補の evaluation を埋めること\n"
            f"{EVALUATOR_JSON_SCHEMA}"
        ),
        expected_output="有効なJSON形式のResearchReport（evaluation付き、合格銘柄のみ）",
        agent=formatter,
        context=[ranking_task, evaluation_task],
    )


def build_tasks(research_prompt: str):
    from growth_stock_search_agent.crew.agents import (
        build_extractor_tool,
        build_llm,
        build_search_tool,
    )

    llm = build_llm()
    search_tool = build_search_tool()
    extractor_tool = build_extractor_tool()

    researcher = create_researcher_agent(llm, search_tool)
    analyst = create_analyst_agent(llm, extractor_tool)
    ranker = create_ranker_agent(llm)
    evaluator = create_evaluator_agent(llm, search_tool)
    formatter = create_formatter_agent(llm)

    research_task = create_research_task(researcher, research_prompt)
    analysis_task = create_analysis_task(analyst, research_task)
    ranking_task = create_ranking_task(ranker, analysis_task)
    evaluation_task = create_evaluation_task(evaluator, ranking_task)
    format_task = create_format_task(formatter, ranking_task, evaluation_task)

    return [research_task, analysis_task, ranking_task, evaluation_task, format_task]
