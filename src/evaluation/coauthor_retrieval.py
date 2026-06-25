from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.author_splits import HISTORY_CUTOFF, history_cutoff, split_author_ids, split_author_set
from common.compat import AUTHORS, RANDOM_STATE, l2_normalize, load_json, load_papers_in_embedding_order
from recommendation.model import AuthorTransformer
from recommendation.training_utils import (
    AuthorExampleDataset,
    collate_examples,
    mean_history_embeddings,
    predict_embeddings,
    retrieval_metrics_from_scores,
)


def cutoff_year_for_split(split: str) -> int:
    """Temporal cutoff is the same for all splits; split only defines the author pool."""
    _ = split
    try:
        return history_cutoff()
    except (FileNotFoundError, KeyError, OSError):
        return HISTORY_CUTOFF


@lru_cache(maxsize=1)
def _coauthor_index() -> dict[str, Any]:
    papers = load_papers_in_embedding_order()
    author_index = load_json(AUTHORS / "author_index.json")
    author_by_id = {a["author_id"]: a for a in author_index}
    qualified = set(load_json(AUTHORS / "qualified_authors.json"))
    qualified_sorted = sorted(qualified)
    return {
        "papers": papers,
        "author_by_id": author_by_id,
        "qualified": qualified,
        "qualified_sorted": qualified_sorted,
    }


def past_coauthors(author_id: str, cutoff: int) -> set[str]:
    idx = _coauthor_index()
    papers = idx["papers"]
    entry = idx["author_by_id"][author_id]
    past: set[str] = set()
    for paper_idx, year in zip(entry["paper_idxs"], entry["years"], strict=False):
        if int(year) > cutoff:
            continue
        for author in papers[int(paper_idx)].get("authors", []):
            aid = author.get("author_id")
            if aid and aid != author_id:
                past.add(aid)
    return past


def relevant_future_coauthors(
    ex: dict[str, Any],
    author_pool: set[str] | None = None,
) -> set[str]:
    idx = _coauthor_index()
    papers = idx["papers"]
    pool = author_pool if author_pool is not None else idx["qualified"]
    author_id = ex["author_id"]
    cutoff = int(ex["cutoff_year"])
    past = past_coauthors(author_id, cutoff)
    relevant: set[str] = set()
    for paper_idx in ex["future_paper_idxs"]:
        for author in papers[int(paper_idx)].get("authors", []):
            aid = author.get("author_id")
            if aid and aid != author_id and aid not in past and aid in pool:
                relevant.add(aid)
    return relevant


def cutoff_examples(
    cutoff: int,
    max_history: int,
    split: str | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    idx = _coauthor_index()
    author_by_id = idx["author_by_id"]
    if split is not None:
        author_ids = split_author_ids(split)
    else:
        author_ids = idx["qualified_sorted"]
    candidate_ids: list[str] = []
    examples: list[dict[str, Any]] = []
    for aid in author_ids:
        entry = author_by_id.get(aid)
        if entry is None:
            continue
        hist = [
            int(paper_idx)
            for paper_idx, year in zip(entry["paper_idxs"], entry["years"], strict=False)
            if int(year) <= cutoff
        ]
        if not hist:
            continue
        candidate_ids.append(aid)
        examples.append(
            {
                "author_id": aid,
                "cutoff_year": cutoff,
                "history_paper_idxs": hist[-max_history:],
                "future_paper_idxs": [],
            }
        )
    return candidate_ids, examples


_mean_candidate_cache: dict[tuple[str, int, int], np.ndarray] = {}


def mean_candidate_embeddings(
    cutoff: int,
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    max_history: int = 20,
    split: str | None = None,
) -> tuple[list[str], np.ndarray]:
    cache_key = (split or "all", cutoff, max_history)
    if cache_key in _mean_candidate_cache:
        candidate_ids, _ = cutoff_examples(cutoff, max_history, split=split)
        return candidate_ids, _mean_candidate_cache[cache_key]

    candidate_ids, cand_examples = cutoff_examples(cutoff, max_history, split=split)
    ds = AuthorExampleDataset(cand_examples, embeddings, q, years, max_history=max_history)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_examples, num_workers=0)
    cand_emb = mean_history_embeddings(loader)
    _mean_candidate_cache[cache_key] = cand_emb
    return candidate_ids, cand_emb


@torch.inference_mode()
def model_candidate_embeddings(
    model: AuthorTransformer,
    cutoff: int,
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    device: torch.device,
    max_history: int = 20,
    split: str | None = None,
) -> tuple[list[str], np.ndarray]:
    candidate_ids, cand_examples = cutoff_examples(cutoff, max_history, split=split)
    ds = AuthorExampleDataset(cand_examples, embeddings, q, years, max_history=max_history)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_examples, num_workers=0)
    cand_emb, _ = predict_embeddings(model, loader, device)
    return candidate_ids, cand_emb.astype(np.float32, copy=False)


def evaluate_coauthor_from_scores(
    scores: np.ndarray,
    examples: list[dict[str, Any]],
    candidate_ids: list[str],
    split: str,
) -> dict[str, float]:
    scores = scores.astype(np.float32, copy=True)
    cid_to_pos = {cid: i for i, cid in enumerate(candidate_ids)}
    author_pool = split_author_set(split)
    relevant_positions: list[set[int]] = []
    n_with_relevant = 0
    for i, ex in enumerate(examples):
        author_id = ex["author_id"]
        cutoff = int(ex["cutoff_year"])
        excluded = {author_id} | past_coauthors(author_id, cutoff)
        for cid in excluded:
            pos = cid_to_pos.get(cid)
            if pos is not None:
                scores[i, pos] = -np.inf

        rel = relevant_future_coauthors(ex, author_pool=author_pool)
        rel_pos = {cid_to_pos[aid] for aid in rel if aid in cid_to_pos}
        if rel_pos:
            n_with_relevant += 1
        relevant_positions.append(rel_pos)

    metrics = retrieval_metrics_from_scores(scores, relevant_positions)
    metrics["n_with_relevant"] = float(n_with_relevant)
    return metrics


def evaluate_coauthor_from_embeddings(
    focal_emb: np.ndarray,
    examples: list[dict[str, Any]],
    candidate_ids: list[str],
    candidate_emb: np.ndarray,
    split: str,
) -> dict[str, float]:
    focal_emb = l2_normalize(focal_emb.astype(np.float32, copy=False))
    candidate_emb = l2_normalize(candidate_emb.astype(np.float32, copy=False))
    scores = focal_emb @ candidate_emb.T
    return evaluate_coauthor_from_scores(scores, examples, candidate_ids, split)


@torch.inference_mode()
def evaluate_coauthor_retrieval(
    model: AuthorTransformer,
    loader,
    examples: list[dict[str, Any]],
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    cutoff_year: int,
    device: torch.device,
    split: str,
    max_history: int = 20,
) -> dict[str, float]:
    candidate_ids, candidate_emb = model_candidate_embeddings(
        model, cutoff_year, embeddings, q, years, device, max_history, split=split
    )
    focal_emb, _ = predict_embeddings(model, loader, device)
    return evaluate_coauthor_from_embeddings(focal_emb, examples, candidate_ids, candidate_emb, split)


def evaluate_mean_coauthor_retrieval(
    loader,
    examples: list[dict[str, Any]],
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    cutoff_year: int,
    split: str,
    max_history: int = 20,
) -> dict[str, float]:
    candidate_ids, candidate_emb = mean_candidate_embeddings(
        cutoff_year, embeddings, q, years, max_history, split=split
    )
    focal_emb = mean_history_embeddings(loader)
    return evaluate_coauthor_from_embeddings(focal_emb, examples, candidate_ids, candidate_emb, split)


def evaluate_random_coauthor_retrieval(
    n_examples: int,
    examples: list[dict[str, Any]],
    cutoff_year: int,
    split: str,
    max_history: int = 20,
    seed: int = RANDOM_STATE,
) -> dict[str, float]:
    candidate_ids, _ = cutoff_examples(cutoff_year, max_history, split=split)
    rng = np.random.default_rng(seed)
    scores = rng.random((n_examples, len(candidate_ids)), dtype=np.float32)
    return evaluate_coauthor_from_scores(scores, examples, candidate_ids, split)
