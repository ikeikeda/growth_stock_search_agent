from __future__ import annotations

import json
import sys

import dspy
from dotenv import load_dotenv

from growth_stock_search_agent.config import PROJECT_ROOT, get_settings
from growth_stock_search_agent.dspy_opt.metric import compute_metric, load_eval_examples
from growth_stock_search_agent.dspy_opt.module import ResearchPromptModule
from growth_stock_search_agent.prompts.loader import load_base_prompt, save_optimized_prompt


def _build_trainset(examples: list[dict]) -> list[dspy.Example]:
    trainset: list[dspy.Example] = []
    for item in examples:
        trainset.append(
            dspy.Example(
                instruction=item["instruction"],
                expected_fields=item.get("expected_fields", []),
                min_candidates=item.get("min_candidates", 1),
                min_alignment_score=item.get("min_alignment_score", 0.5),
            ).with_inputs("instruction")
        )
    return trainset


def _configure_dspy_lm() -> None:
    settings = get_settings()
    lm = dspy.LM(
        model=f"ollama_chat/{settings.ollama_model}",
        api_base=settings.ollama_base_url,
        api_key="",
        cache=False,
    )
    dspy.configure(lm=lm, track_usage=True)


def _extract_best_instruction(compiled: ResearchPromptModule, fallback: str) -> str:
    generate = compiled.generate
    if hasattr(generate, "signature") and hasattr(generate.signature, "instructions"):
        instructions = generate.signature.instructions
        if instructions and instructions.strip():
            return instructions.strip()

    if hasattr(generate, "extended_signature"):
        extended = generate.extended_signature
        if hasattr(extended, "instructions") and extended.instructions:
            return extended.instructions.strip()

    return fallback


def main(argv: list[str] | None = None) -> int:
    del argv
    load_dotenv(PROJECT_ROOT / ".env")
    _configure_dspy_lm()

    base_prompt = load_base_prompt()
    examples = load_eval_examples()
    trainset = _build_trainset(examples)

    module = ResearchPromptModule()

    try:
        from dspy.teleprompt import MIPROv2

        optimizer = MIPROv2(
            metric=compute_metric,
            num_candidates=5,
            init_temperature=0.3,
        )
        compiled = optimizer.compile(
            module,
            trainset=trainset,
            max_bootstrapped_demos=0,
            max_labeled_demos=0,
        )
    except Exception as exc:
        print(f"MIPROv2 unavailable ({exc}); using base prompt without optimization.")
        compiled = module

    best_instruction = _extract_best_instruction(compiled, base_prompt)

    scores: list[float] = []
    for example in trainset:
        pred = compiled(research_instruction=example.instruction)
        scores.append(compute_metric(example, pred))

    avg_score = sum(scores) / len(scores) if scores else 0.0
    output_path = save_optimized_prompt(
        prompt=best_instruction,
        score=avg_score,
        char_count=len(best_instruction),
    )

    print(json.dumps(
        {
            "optimized_prompt_path": str(output_path),
            "char_count": len(best_instruction),
            "avg_score": avg_score,
            "prompt_preview": best_instruction[:200] + "...",
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
