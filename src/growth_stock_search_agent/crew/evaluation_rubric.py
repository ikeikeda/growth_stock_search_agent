"""Evaluation rubric for the Quality Evaluator agent."""

RUBRIC_CATEGORIES = [
    {
        "category": "成長性",
        "item": "売上・営業利益成長率",
        "criteria": "直近実績または今期予想で原則YoY10%以上（20%以上を優先評価）",
    },
    {
        "category": "割安度",
        "item": "予想PER",
        "criteria": "原則15倍未満。卸売・建設・不動産・金融は同業比較で厳格判定",
    },
    {
        "category": "除外",
        "item": "赤字・希薄化・反動増",
        "criteria": "該当する場合は不合格",
    },
    {
        "category": "未注目度",
        "item": "株価上昇・テーマ過熱・PER放置",
        "criteria": "成長しているのに評価されていないことを確認",
    },
    {
        "category": "情報鮮度",
        "item": "決算・株価",
        "criteria": "直近決算と現在株価に基づくPER再計算がされているか",
    },
]

PURPOSE_STATEMENT = (
    "市場でまだ注目されていない割安成長株を発掘する。"
    "単なる低PER株ではなく、成長しているのに評価されていない銘柄を選定すること。"
)


def format_rubric_for_prompt() -> str:
    lines = [f"【目的】 {PURPOSE_STATEMENT}", "", "【評価ルーブリック】"]
    for entry in RUBRIC_CATEGORIES:
        lines.append(
            f"- {entry['category']} / {entry['item']}: {entry['criteria']}"
        )
    return "\n".join(lines)
