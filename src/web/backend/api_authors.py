from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from .state import query_variants, state

router = APIRouter()


@router.get("/api/authors/search")
def author_search(q: str, limit: int = 10) -> list[dict[str, Any]]:
    variants = query_variants(q)
    if not variants or max(len(v) for v in variants) < 2:
        return []
    hits = [row for row in state.authors if any(variant in row["query"] for variant in variants)]
    hits.sort(key=lambda row: (-row["n_papers"], row["display_name"]))
    return [
        {
            "author_id": row["author_id"],
            "display_name": row["display_name"],
            "n_papers": row["n_papers"],
            "openalex_url": row["openalex_url"],
            "last_papers": row["last_papers"][-5:],
        }
        for row in hits[:limit]
    ]
