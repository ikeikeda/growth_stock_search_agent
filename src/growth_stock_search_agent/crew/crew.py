from __future__ import annotations

from crewai import Crew, Process

from growth_stock_search_agent.crew.tasks import build_tasks
from growth_stock_search_agent.models import ResearchReport, parse_research_report


def run_research_crew(research_prompt: str, retry_on_parse_error: bool = True) -> ResearchReport:
    tasks = build_tasks(research_prompt)
    crew = Crew(
        agents=[task.agent for task in tasks],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )

    last_error: Exception | None = None
    attempts = 2 if retry_on_parse_error else 1

    for _ in range(attempts):
        result = crew.kickoff()
        raw_output = str(result.raw if hasattr(result, "raw") else result)
        try:
            return parse_research_report(raw_output)
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Failed to parse crew output as ResearchReport: {last_error}")
