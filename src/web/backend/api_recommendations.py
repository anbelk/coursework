from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from recommendation.pipeline import run_pipeline

from .state import state

router = APIRouter()


class RecommendationRequest(BaseModel):
    top_n: int = 10
    top_k: int = 3
    workers: int = 8


@router.post("/api/recommendations/{author_id}")
def recommendations(author_id: str, request: RecommendationRequest) -> dict[str, Any]:
    if author_id not in state.author_ids:
        raise HTTPException(status_code=404, detail="unknown or unqualified author")
    return run_pipeline(author_id, top_n=request.top_n, top_k=request.top_k, n_workers=request.workers)
