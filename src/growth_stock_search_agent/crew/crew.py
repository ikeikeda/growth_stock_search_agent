from __future__ import annotations

import time

from crewai import Crew, Process

from growth_stock_search_agent.config import get_settings
from growth_stock_search_agent.crew.tasks import build_tasks
from growth_stock_search_agent.models import ResearchReport, parse_research_report
from growth_stock_search_agent.output.enrichment import prepare_report


def _is_empty_llm_response(exc: BaseException) -> bool:
    return "Invalid response from LLM call" in str(exc)


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
            return prepare_report(parse_research_report(raw_output))
        except Exception as exc:
            last_error = exc
            if not _is_empty_llm_response(exc):
                # Parse / other ValueErrors: retry when attempts remain.
                if isinstance(exc, ValueError) and attempt + 1 < attempts:
                    continue
                if isinstance(exc, ValueError):
                    break
                raise

            settings = get_settings()
            if attempt + 1 < attempts:
                # Transient empty content (e.g. thinking budget edge cases).
                time.sleep(2 ** attempt)
                continue
            raise ValueError(
                "Ollama から空の応答が返されました。"
                " gemma4 等の thinking モデルでは生成予算（OLLAMA_MAX_TOKENS /"
                " num_predict）が thinking 中に尽きると content が空になります。"
                " OLLAMA_DISABLE_THINKING=false のまま OLLAMA_MAX_TOKENS を上げてください。"
                f" (model={settings.ollama_model},"
                f" max_tokens={settings.ollama_max_tokens},"
                f" attempt={attempt + 1}/{attempts})"
            ) from exc
        except ValueError as exc:
            last_error = exc
            if "Invalid response from LLM call" in str(exc):
                raise ValueError(
                    "Ollama から空の応答が返されました。"
                    " gemma4 では thinking を無効にすると本文が捨てられます。"
                    " OLLAMA_DISABLE_THINKING=false を確認してください。"
                    f" (model={get_settings().ollama_model}, attempt={attempt + 1}/{attempts})"
                ) from exc
        except Exception as exc:
            last_error = exc
            if "Invalid response from LLM call" in str(exc):
                raise ValueError(
                    "Ollama から空の応答が返されました。"
                    " gemma4 では thinking を無効にすると本文が捨てられます。"
                    f" (attempt={attempt + 1}/{attempts})"
                ) from exc

    raise ValueError(f"Failed to parse crew output as ResearchReport: {last_error}")
