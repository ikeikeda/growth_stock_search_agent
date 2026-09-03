from __future__ import annotations

from crewai import Agent, LLM
from crewai_tools import TavilyExtractorTool, TavilySearchTool

from growth_stock_search_agent.config import get_settings

JAPAN_FINANCE_DOMAINS = [
    "finance.yahoo.co.jp",
    "kabutan.jp",
    "irbank.net",
    "minkabu.jp",
    "stockweather.co.jp",
]


def build_llm() -> LLM:
    """Build an Ollama LLM via LiteLLM with settings tuned for gemma4 tool calling."""
    settings = get_settings()
    additional_params: dict[str, object] = {
        "num_ctx": settings.ollama_num_ctx,
    }
    if settings.ollama_disable_thinking:
        additional_params["think"] = False

    return LLM(
        model=f"ollama/{settings.ollama_model}",
        base_url=settings.ollama_base_url,
        provider="litellm",
        temperature=0.3,
        timeout=settings.ollama_timeout,
        max_tokens=4096,
        additional_params=additional_params,
    )


def build_search_tool() -> TavilySearchTool:
    return TavilySearchTool(
        search_depth="advanced",
        max_results=8,
        include_answer=True,
        include_domains=JAPAN_FINANCE_DOMAINS,
    )


def build_extractor_tool() -> TavilyExtractorTool:
    return TavilyExtractorTool(
        extract_depth="advanced",
        timeout=90,
    )


def create_researcher_agent(llm: LLM, search_tool: TavilySearchTool) -> Agent:
    return Agent(
        role="日本株市場リサーチャー",
        goal=(
            "Web検索で日本株の成長株候補を広く収集し、"
            "決算短信・IR・有価証券報告書など一次情報のURLを優先して整理する"
        ),
        backstory=(
            "日本株アナリストとして10年以上、未発掘の成長株を"
            "一次情報から調査してきた専門家。"
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        max_iter=10,
        respect_context_window=True,
    )


def create_analyst_agent(llm: LLM, extractor_tool: TavilyExtractorTool) -> Agent:
    return Agent(
        role="財務アナリスト",
        goal=(
            "候補銘柄のPER・成長率・除外条件を一次情報で検証し、"
            "割安成長株候補を絞り込む"
        ),
        backstory=(
            "決算分析とバリュエーションに精通したCFA。"
            "低PER業種の同業比較や反動増の見極めを得意とする。"
        ),
        tools=[extractor_tool],
        llm=llm,
        verbose=True,
        max_iter=10,
        respect_context_window=True,
    )


def create_ranker_agent(llm: LLM) -> Agent:
    return Agent(
        role="投資レポート作成者",
        goal=(
            "候補を10銘柄程度に順位付けし、"
            "特に有望な3銘柄を比較したJSONレポートを作成する"
        ),
        backstory=(
            "機関投資家向けレポートを執筆するストラテジスト。"
            "構造化されたJSON出力を正確に作成する。"
        ),
        llm=llm,
        verbose=True,
        max_iter=8,
        respect_context_window=True,
    )


def create_evaluator_agent(llm: LLM, search_tool: TavilySearchTool) -> Agent:
    return Agent(
        role="リサーチ品質監査者",
        goal=(
            "Rankerの選定結果が「市場でまだ注目されていない割安成長株」"
            "という本来の目的に合致しているか、ルーブリックに沿って銘柄ごとに判定する"
        ),
        backstory=(
            "独立した監査役として、リサーチ結果の品質と目的適合性を"
            "厳格かつ客観的に評価する専門家。Rankerの結論を鵜呑みにしない。"
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        max_iter=5,
        respect_context_window=True,
    )


def create_formatter_agent(llm: LLM) -> Agent:
    return Agent(
        role="JSON整形者",
        goal=(
            "RankerとEvaluatorの結果を指定スキーマの有効なJSON（ResearchReport）へ変換する。"
            "Markdownや解説文は一切出力しない。"
        ),
        backstory=(
            "構造化データ変換の専門家。"
            "与えられた評価結果を欠落なくJSONへ写し、説明文は付けない。"
        ),
        llm=llm,
        verbose=True,
        max_iter=3,
        respect_context_window=True,
    )
