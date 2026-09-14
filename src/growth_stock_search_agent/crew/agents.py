from __future__ import annotations

from crewai import Agent, LLM
from crewai_tools import TavilyExtractorTool, TavilySearchTool
from pydantic import BaseModel, Field

from growth_stock_search_agent.config import get_settings
from growth_stock_search_agent.crew.ollama_llm import OllamaCrewLLM, generation_token_budget

JAPAN_FINANCE_DOMAINS = [
    "finance.yahoo.co.jp",
    "kabutan.jp",
    "irbank.net",
    "minkabu.jp",
    "stockweather.co.jp",
]


def build_llm() -> LLM:
    """Build an Ollama chat LLM via LiteLLM with settings tuned for gemma4 tool calling.

    ``ollama_chat/`` uses ``/api/chat``. gemma4 often puts the reply in
    ``thinking`` and leaves ``message.content`` empty. ``OllamaCrewLLM``
    copies that thinking into content when there are no tool calls, which
    is what CrewAI requires. Keep thinking enabled for gemma4
    (``think=False`` discards thought tokens and often returns empty content).
    """
    settings = get_settings()
    max_tokens = generation_token_budget(
        settings.ollama_num_ctx, settings.ollama_max_tokens
    )
    additional_params: dict[str, object] = {
        "num_ctx": settings.ollama_num_ctx,
        # Ollama generation length; keep in sync with LiteLLM max_tokens.
        "num_predict": max_tokens,
    }
    if settings.ollama_disable_thinking:
        # Prefer leaving this unset/false for gemma4 — think=False can discard
        # thought tokens and still leave content empty.
        additional_params["think"] = False

    llm = OllamaCrewLLM(
        model=f"ollama_chat/{settings.ollama_model}",
        base_url=settings.ollama_base_url,
        provider="litellm",
        temperature=0.3,
        timeout=settings.ollama_timeout,
        max_tokens=max_tokens,
        additional_params=additional_params,
        stream=False,
    )
    context_size = int(settings.ollama_num_ctx * 0.85)
    llm.supports_function_calling = lambda: True  # type: ignore[method-assign]
    llm.get_context_window_size = lambda: context_size  # type: ignore[method-assign]
    return llm


def build_search_tool() -> TavilySearchTool:
    return TavilySearchTool(
        search_depth="advanced",
        max_results=8,
        include_answer=True,
        include_domains=JAPAN_FINANCE_DOMAINS,
    )


class TavilyExtractUrlsSchema(BaseModel):
    urls: str = Field(
        ...,
        description=(
            "Comma-separated http(s) URLs to extract "
            "(IR, earnings, kabutan, Yahoo Finance)."
        ),
    )


class SimpleTavilyExtractorTool(TavilyExtractorTool):
    """Extractor with a string-only schema. Union types confuse gemma4 tool calls."""

    name: str = "tavily_extract"
    description: str = (
        "Extract page text from IR / earnings / kabutan / Yahoo Finance URLs. "
        "Pass one or more comma-separated URLs."
    )
    args_schema: type[BaseModel] = TavilyExtractUrlsSchema

    def _run(self, urls: list[str] | str) -> str:
        if isinstance(urls, str):
            parsed = [
                part.strip()
                for part in urls.replace("\n", ",").split(",")
                if part.strip()
            ]
        else:
            parsed = list(urls)
        return super()._run(parsed)


def build_extractor_tool() -> TavilyExtractorTool:
    return SimpleTavilyExtractorTool(
        extract_depth="advanced",
        timeout=90,
    )


def create_researcher_agent(llm: LLM, search_tool: TavilySearchTool) -> Agent:
    return Agent(
        role="日本株市場リサーチャー",
        goal=(
            "Web検索で実在する日本の上場企業だけを収集し、"
            "正式社名と4桁コードを株探またはYahooファイナンスで一致確認したうえで、"
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


def create_analyst_agent(
    llm: LLM,
    search_tool: TavilySearchTool,
    extractor_tool: TavilyExtractorTool,
) -> Agent:
    return Agent(
        role="財務アナリスト",
        goal=(
            "候補銘柄のPER・成長率・除外条件を一次情報で検証し、"
            "割安成長株候補を絞り込む"
        ),
        backstory=(
            "決算分析とバリュエーションに精通したCFA。"
            "低PER業種の同業比較や反動増の見極めを得意とする。"
            "最終回答の前に必ず検索またはページ抽出ツールで一次情報を確認する。"
        ),
        tools=[search_tool, extractor_tool],
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
            "正式社名・実在コード・事業内容付きで特に有望な3銘柄を比較したJSONレポートを作成する"
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
            "創作銘柄や社名とコードの不一致を見逃さない。"
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        max_iter=10,
        respect_context_window=True,
    )
