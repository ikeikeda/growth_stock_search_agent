from __future__ import annotations

import dspy


class ResearchSignature(dspy.Signature):
    """短い調査指示から成長株リサーチJSONを生成する。"""

    research_instruction: str = dspy.InputField(
        desc="割安成長株リサーチの指示文（短くても要件を満たすこと）"
    )
    stock_report_json: str = dspy.OutputField(
        desc="ResearchReport形式のJSON文字列（candidates, top3_comparison, evaluation含む）"
    )


class ResearchPromptModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.generate = dspy.ChainOfThought(ResearchSignature)

    def forward(self, research_instruction: str) -> dspy.Prediction:
        return self.generate(research_instruction=research_instruction)
