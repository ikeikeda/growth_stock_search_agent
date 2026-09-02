from __future__ import annotations

from crewai import Crew, Process

from growth_stock_search_agent.config import get_settings
from growth_stock_search_agent.crew.tasks import build_tasks
from growth_stock_search_agent.models import ResearchReport, parse_research_report


def run_research_crew(research_prompt: str, retry_on_parse_error: bool = True) -> ResearchReport:
    tasks = build_tasks(research_prompt)
    crew = Crew(
        agents=[task.agent for task in tasks],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
        memory=False,
    )

    last_error: Exception | None = None
    attempts = 2 if retry_on_parse_error else 1

    for attempt in range(attempts):
        try:
            result = crew.kickoff()
            raw_output = str(result.raw if hasattr(result, "raw") else result)
            return parse_research_report(raw_output)
        except ValueError as exc:
            last_error = exc
            if "Invalid response from LLM call" in str(exc):
                raise ValueError(
                    "Ollama から空の応答が返されました。"
                    " Ollama が起動しているか、モデル名が正しいか確認してください。"
                    f" (model={get_settings().ollama_model}, attempt={attempt + 1}/{attempts})"
                ) from exc
        except Exception as exc:
            last_error = exc
            if "Invalid response from LLM call" in str(exc):
                raise ValueError(
                    "Ollama から空の応答が返されました。"
                    " gemma4 等の thinking モデルでは think=False が必要です。"
                    f" (attempt={attempt + 1}/{attempts})"
                ) from exc

    raise ValueError(f"Failed to parse crew output as ResearchReport: {last_error}")
