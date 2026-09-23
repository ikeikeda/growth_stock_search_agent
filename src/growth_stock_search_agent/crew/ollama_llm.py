from __future__ import annotations

import logging
import re
import threading
from typing import Any

from crewai import LLM
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.task import Task
from pydantic import BaseModel

logger = logging.getLogger(__name__)
_completion_lock = threading.Lock()
_REJECTED_OUTPUT_LIMIT = 6
_REJECTED_OUTPUT_CHARS = 300_000

# gemma4 sometimes leaks chat-template control tokens instead of an answer.
_CHANNEL_TOKEN_RE = re.compile(
    r"<\s*\|?\s*channel\s*\|?\s*>\s*thought\b|<\s*channel\s*\|>",
    re.IGNORECASE,
)


def generation_token_budget(num_ctx: int, max_tokens: int) -> int:
    """Cap generation so prompt and thinking still fit in ``num_ctx``.

    Ollama shares the context window between prompt and ``num_predict``.
    Setting both to the same value lets thinking fill the remaining window
    and leave ``message.content`` empty.
    """
    reserved_for_prompt = max(2048, num_ctx // 4)
    capped = max(512, num_ctx - reserved_for_prompt)
    return max(512, min(max_tokens, capped))


def strip_channel_thoughts(text: str) -> str:
    """Remove gemma chat-template thought markers from a visible answer."""
    return _CHANNEL_TOKEN_RE.sub("", text or "").strip()


def channel_thought_count(text: str) -> int:
    return len(_CHANNEL_TOKEN_RE.findall(text or ""))


def task_expects_json(from_task: Task | None) -> bool:
    if from_task is None:
        return False
    expected = str(getattr(from_task, "expected_output", "") or "")
    return "JSON" in expected or "json" in expected


def reasoning_is_usable_answer(text: str, *, require_json: bool) -> bool:
    """True when text is an answer rather than a thought-token loop.

    JSON tasks must contain an object. Repeated ``<|channel>thought`` leaks
    are ignored unless a JSON object remains after the markers are removed.
    """
    cleaned = strip_channel_thoughts(text)
    if not cleaned:
        return False
    has_json = "{" in cleaned and "}" in cleaned
    if channel_thought_count(text) >= 3 and not has_json:
        return False
    if require_json and not has_json:
        return False
    return True


def _message_reasoning(response: Any) -> str:
    try:
        choices = getattr(response, "choices", None)
        if not choices:
            return ""
        message = choices[0].message
        reasoning = getattr(message, "reasoning_content", None)
        if not reasoning and getattr(message, "model_extra", None):
            extra = message.model_extra or {}
            reasoning = extra.get("reasoning_content") or extra.get("thinking")
        return str(reasoning or "")
    except Exception:
        return ""


def promote_reasoning_to_content(response: Any, *, require_json: bool = False) -> bool:
    """Copy usable ``reasoning_content`` into ``content`` when content is empty.

    gemma4 / Ollama often put the whole reply in ``thinking`` (mapped by
    LiteLLM to ``reasoning_content``) and leave ``content`` as ``\"\"``.
    CrewAI then treats the call as an empty LLM response. Tool calls are
    left unchanged so native function calling still works.

    Thought-channel loops and, for JSON tasks, reasoning with no object are
    not copied. Promoting those makes CrewAI treat a monologue as the answer.
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
        if not reasoning_is_usable_answer(reasoning, require_json=require_json):
            logger.warning(
                "Ollama reasoning was not promoted (%s chars, json_required=%s)",
                len(reasoning),
                require_json,
            )
            return False

        visible = strip_channel_thoughts(reasoning)
        message.content = visible
        logger.warning(
            "Ollama returned empty content; using reasoning_content (%s chars)",
            len(visible),
        )
        from growth_stock_search_agent.models import extract_json_payload
        import json as json_lib

        visible = reasoning
        try:
            payload = extract_json_payload(reasoning)
            visible = json_lib.dumps(payload, ensure_ascii=False)
            logger.warning(
                "Ollama returned empty content; extracted JSON from reasoning_content (%s chars -> %s chars)",
                len(reasoning),
                len(visible),
            )
        except Exception:
            logger.warning(
                "Ollama returned empty content; using reasoning_content (%s chars)",
                len(reasoning),
            )

        message.content = visible
        return True
    except Exception:
        logger.debug("Could not promote reasoning_content", exc_info=True)
        return False


class OllamaCrewLLM(LLM):
    """CrewAI LLM that recovers gemma4 thinking-only replies as visible content."""

    def remember_rejected_output(self, text: str) -> None:
        """Keep discarded model text so a failed run can be inspected later."""
        remembered = getattr(self, "rejected_outputs", None)
        if not isinstance(remembered, list):
            remembered = []
            self.rejected_outputs = remembered
        remembered.append((text or "")[:_REJECTED_OUTPUT_CHARS])
        if len(remembered) > _REJECTED_OUTPUT_LIMIT:
            del remembered[:-_REJECTED_OUTPUT_LIMIT]

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
        require_json = task_expects_json(from_task)

        def wrapped_completion(*args: Any, **kwargs: Any) -> Any:
            response = original_completion(*args, **kwargs)
            promoted = promote_reasoning_to_content(
                response, require_json=require_json
            )
            if not promoted:
                reasoning = _message_reasoning(response)
                if reasoning and not reasoning_is_usable_answer(
                    reasoning, require_json=require_json
                ):
                    self.remember_rejected_output(reasoning)
            return response

        with _completion_lock:
            client.completion = wrapped_completion
            try:
                result = super()._handle_non_streaming_response(
                    params,
                    callbacks=callbacks,
                    available_functions=available_functions,
                    from_task=from_task,
                    from_agent=from_agent,
                    response_model=response_model,
                )
            finally:
                client.completion = original_completion

        if not isinstance(result, str):
            return result
        if not reasoning_is_usable_answer(result, require_json=require_json):
            if result.strip():
                self.remember_rejected_output(result)
                logger.warning(
                    "Discarded unusable Ollama output (%s chars, json_required=%s)",
                    len(result),
                    require_json,
                )
            return ""
        return strip_channel_thoughts(result)
