from __future__ import annotations

import os

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import json
import sys
from collections import Counter

from dotenv import load_dotenv

from growth_stock_search_agent.config import PROJECT_ROOT
from growth_stock_search_agent.feedback.store import (
    feedback_dir,
    load_failure_records,
    load_lessons,
)


def main(argv: list[str] | None = None) -> int:
    del argv
    load_dotenv(PROJECT_ROOT / ".env")
    directory = feedback_dir()
    records = load_failure_records(directory)
    lessons = load_lessons(directory)

    event_total = sum(len(record.events) for record in records)
    kinds: Counter[str] = Counter()
    for record in records:
        kinds.update(record.kind_counts)

    payload = {
        "feedback_dir": str(directory),
        "failure_records": len(records),
        "failure_events": event_total,
        "kind_counts": dict(kinds),
        "lessons": lessons.model_dump(),
    }
    if not records and not lessons.updated_at:
        payload["note"] = "蓄積された失敗はまだありません。"
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
