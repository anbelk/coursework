from __future__ import annotations

from typing import Any

import numpy as np
import torch

from recommendation.model import AuthorTransformer
from recommendation.training_utils import AuthorExampleDataset, collate_examples, move_batch


def encode_author_list(
    model: AuthorTransformer,
    author_ids: list[str],
    cutoff: int,
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    max_history: int,
    device: torch.device,
    detach: bool = True,
    chunk: int = 512,
) -> tuple[list[str], torch.Tensor]:
    """Encode authors with the Transformer at a single cutoff.

    Returns the ids that actually had history (<= cutoff) and the [M, d] tensor of
    their embeddings, in the same order. Authors without pre-cutoff history are dropped.

    Candidates are encoded in chunks (bounded memory) and, by default, under no_grad:
    with the residual-from-mean architecture the focal embedding is anchored to the mean
    paper embedding, so detached candidate targets do not cause representation collapse,
    while keeping memory/compute independent of how many distinct coauthors a batch hits.
    """
    from evaluation.coauthor_retrieval import _coauthor_index

    author_by_id = _coauthor_index()["author_by_id"]
    examples: list[dict[str, Any]] = []
    present: list[str] = []
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
        present.append(aid)
        examples.append(
            {
                "author_id": aid,
                "cutoff_year": int(cutoff),
                "history_paper_idxs": hist[-max_history:],
                "future_paper_idxs": [],
            }
        )
    if not examples:
        return [], torch.empty(0, model.config.emb_dim, device=device)

    ds = AuthorExampleDataset(examples, embeddings, q, years, max_history=max_history)
    grad_ctx = torch.no_grad() if detach else torch.enable_grad()
    outs: list[torch.Tensor] = []
    with grad_ctx:
        for start in range(0, len(ds), chunk):
            mini = move_batch(
                collate_examples([ds[i] for i in range(start, min(start + chunk, len(ds)))]),
                device,
            )
            pred = model(mini["history_emb"], mini["history_q"], mini["years"], mini["mask"])["pred_emb"]
            outs.append(pred)
    return present, torch.cat(outs, dim=0)


def coauthor_infonce_loss(
    model: AuthorTransformer,
    focal_emb: torch.Tensor,
    batch: dict[str, Any],
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    max_history: int,
    device: torch.device,
    tau: float,
    n_negatives: int = 256,
    author_pool: set[str] | None = None,
    max_positives: int = 8,
) -> torch.Tensor:
    """Symmetric multi-positive InfoNCE on Transformer author embeddings.

    Focal author (history -> Transformer) vs future coauthors (positives) and a shared
    pool of random authors (negatives). Candidates are encoded by the same Transformer
    (detached, batched) and shared across the batch, so other rows' positives act as
    in-batch negatives. The residual-from-mean architecture prevents collapse.

    Positives per row are capped (max_positives) and negatives sampled (n_negatives) so
    the number of distinct candidates per batch is bounded, keeping memory stable on MPS.
    """
    from evaluation.coauthor_retrieval import _coauthor_index, past_coauthors

    pool = list(author_pool) if author_pool is not None else _coauthor_index()["qualified_sorted"]
    bsz = focal_emb.shape[0]
    cutoff = int(batch["cutoff_year"][0]) if bsz else 0

    positives_per_row: list[set[str]] = []
    all_positives: set[str] = set()
    for i in range(bsz):
        pos = batch["future_coauthor_ids"][i]
        if len(pos) > max_positives:
            pos = np.random.choice(np.asarray(pos, dtype=object), size=max_positives, replace=False).tolist()
        pos = set(pos)
        positives_per_row.append(pos)
        all_positives.update(pos)

    if not all_positives:
        return focal_emb.sum() * 0.0

    n_sample = min(n_negatives, len(pool))
    neg_sample = set(np.random.choice(np.asarray(pool, dtype=object), size=n_sample, replace=False).tolist())

    candidate_ids = sorted(all_positives | neg_sample)
    present_ids, cand_emb = encode_author_list(
        model, candidate_ids, cutoff, embeddings, q, years, max_history, device
    )
    if not present_ids:
        return focal_emb.sum() * 0.0
    col_of = {aid: j for j, aid in enumerate(present_ids)}

    sims = focal_emb @ cand_emb.t() / tau  # [B, M]
    neg_inf = torch.finfo(sims.dtype).min

    pos_mask = torch.zeros_like(sims, dtype=torch.bool)
    valid_mask = torch.zeros_like(sims, dtype=torch.bool)
    rows_with_pos: list[int] = []
    for i in range(bsz):
        pos_cols = [col_of[a] for a in positives_per_row[i] if a in col_of]
        if not pos_cols:
            continue
        for col in pos_cols:
            pos_mask[i, col] = True
        # Denominator = negatives only. Positives contribute via the numerator; they must
        # not compete as negatives (including other true coauthors of the same anchor).
        valid_mask[i, :] = True
        exclude = (
            {batch["author_id"][i]}
            | past_coauthors(batch["author_id"][i], cutoff)
            | positives_per_row[i]
        )
        for aid in exclude:
            col = col_of.get(aid)
            if col is not None:
                valid_mask[i, col] = False
        rows_with_pos.append(i)

    if not rows_with_pos:
        return focal_emb.sum() * 0.0

    idx = torch.tensor(rows_with_pos, device=sims.device, dtype=torch.long)
    sims = sims.index_select(0, idx)
    pos_mask = pos_mask.index_select(0, idx)
    valid_mask = valid_mask.index_select(0, idx)

    log_pos = torch.where(pos_mask, sims, torch.full_like(sims, neg_inf)).logsumexp(dim=1)
    log_den = torch.where(valid_mask, sims, torch.full_like(sims, neg_inf)).logsumexp(dim=1)
    return (-log_pos + log_den).mean()
