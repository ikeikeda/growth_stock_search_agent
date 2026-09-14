from __future__ import annotations

import logging
import threading
from typing import Any

from crewai import LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.task import Task
from pydantic import BaseModel

logger = logging.getLogger(__name__)
_completion_lock = threading.Lock()


def generation_token_budget(num_ctx: int, max_tokens: int) -> int:
    """Cap generation so prompt and thinking still fit in ``num_ctx``.

    Ollama shares the context window between prompt and ``num_predict``.
    Setting both to the same value lets thinking fill the remaining window
    and leave ``message.content`` empty.
    """
    reserved_for_prompt = max(2048, num_ctx // 4)
    capped = max(512, num_ctx - reserved_for_prompt)
    return max(512, min(max_tokens, capped))


def promote_reasoning_to_content(response: Any) -> bool:
    """Copy ``reasoning_content`` into ``content`` when the visible answer is empty.

    gemma4 / Ollama often put the whole reply in ``thinking`` (mapped by
    LiteLLM to ``reasoning_content``) and leave ``content`` as ``\"\"``.
    CrewAI then treats the call as an empty LLM response. Tool calls are
    left unchanged so native function calling still works.
    """
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return False
        message = choices[0].message
        if getattr(message, "content", None):
            return False
        if getattr(message, "tool_calls", None):
            return False

        reasoning = getattr(message, "reasoning_content", None)
        if not reasoning and getattr(message, "model_extra", None):
            extra = message.model_extra or {}
            reasoning = extra.get("reasoning_content") or extra.get("thinking")
        if not reasoning:
            return False

        message.content = reasoning
        logger.warning(
            "Ollama returned empty content; using reasoning_content (%s chars)",
            len(reasoning),
        )
        return True
    except Exception:
        logger.debug("Could not promote reasoning_content", exc_info=True)
        return False


class OllamaCrewLLM(LLM):
    """CrewAI LLM that recovers gemma4 thinking-only replies as visible content."""

    def _handle_non_streaming_response(
        self,
        params: dict[str, Any],
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: BaseAgent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        from crewai import llm as crewai_llm

        client = crewai_llm.litellm
        if client is None:
            return super()._handle_non_streaming_response(
                params,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
                response_model=response_model,
            )

        original_completion = client.completion

        def wrapped_completion(*args: Any, **kwargs: Any) -> Any:
            response = original_completion(*args, **kwargs)
            promote_reasoning_to_content(response)
            return response

        with _completion_lock:
            client.completion = wrapped_completion
            try:
                return super()._handle_non_streaming_response(
                    params,
                    callbacks=callbacks,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )
            finally:
                client.completion = original_completion
