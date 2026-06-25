from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .paths import PAPER_IDS_PATH, PAPERS_PATH


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_paper_ids() -> list[str]:
    return load_json(PAPER_IDS_PATH)


def load_papers_by_id() -> dict[str, dict[str, Any]]:
    return {rec["paper_id"]: rec for rec in iter_jsonl(PAPERS_PATH)}


def load_papers_in_embedding_order() -> list[dict[str, Any]]:
    ids = load_paper_ids()
    by_id = load_papers_by_id()
    return [by_id[paper_id] for paper_id in ids]
