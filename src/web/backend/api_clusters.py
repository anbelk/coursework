from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from .state import openalex_url, state

router = APIRouter()


@router.get("/api/map/state")
def map_state() -> dict[str, Any]:
    return state.map_payload


@router.get("/api/papers/{paper_id}")
def paper_detail(paper_id: str) -> dict[str, Any]:
    paper = state.paper_by_id.get(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="unknown paper")
    fine_id, fine_label, meta_id, meta_label = state.paper_meta(paper_id)
    return {
        "paper_id": paper_id,
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract", ""),
        "year": paper.get("year"),
        "authors": paper.get("authors", []),
        "fine_id": fine_id,
        "fine_label": fine_label,
        "meta_id": meta_id,
        "meta_label": meta_label,
        "openalex_url": openalex_url(paper_id),
    }


@router.get("/api/clusters/info/{node_id}")
def cluster_info(node_id: str) -> dict[str, Any]:
    if node_id.startswith("fine_"):
        fine_id = int(node_id.split("_", 1)[1])
        return {
            "id": node_id,
            "type": "fine",
            "label": state.fine_label(fine_id),
            "paper_count": int(state.fine_sizes.get(str(fine_id), 0)),
            "top_terms": state.fine_terms.get(str(fine_id), [])[:3],
            "representative_papers": state.fine_reps.get(str(fine_id), [])[:3],
        }
    if node_id.startswith("meta_"):
        meta_id = int(node_id.split("_", 1)[1])
        return {
            "id": node_id,
            "type": "meta",
            "label": state.meta_label(meta_id),
            "paper_count": int(state.meta.get("meta_paper_count", {}).get(str(meta_id), 0)),
            "top_terms": state.meta_terms.get(str(meta_id), [])[:3],
            "representative_papers": state.meta_reps.get(str(meta_id), [])[:3],
        }
    raise HTTPException(status_code=404, detail="unknown cluster node")
