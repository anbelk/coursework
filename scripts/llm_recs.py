from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline_common import (
    LLMCache,
    PREDICTIONS,
    RESULTS,
    compact_abstract,
    load_json,
    load_papers_in_embedding_order,
    save_json,
)


SYSTEM_PROMPT = """You evaluate potential scientific collaboration relevance.
Return only valid JSON:
{
  "ratings": [
    {"candidate_id": "author id exactly as provided", "score": 0, "reason": "brief reason"}
  ]
}
Score each candidate independently using this scale:
0 = irrelevant
1 = broad area is similar, but the connection is weak
2 = relevant: close topic/methods, potentially good coauthor
3 = very relevant: strong match or useful complementarity
Do not assume the candidates are ranked. Do not use any hidden score."""


def stable_shuffle(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def build_author_maps() -> tuple[dict[str, set[str]], dict[str, set[int]]]:
    papers = load_papers_in_embedding_order()
    coauthors: dict[str, set[str]] = defaultdict(set)
    papers_by_author: dict[str, set[int]] = defaultdict(set)
    for idx, paper in enumerate(papers):
        aids = [a.get("author_id") for a in paper.get("authors", []) if a.get("author_id")]
        for aid in aids:
            papers_by_author[aid].add(idx)
            coauthors[aid].update(x for x in aids if x != aid)
    return coauthors, papers_by_author


def paper_context(paper_idxs: list[int], papers: list[dict], max_chars: int = 600) -> str:
    chunks = []
    for i, idx in enumerate(paper_idxs[-5:], start=1):
        p = papers[int(idx)]
        chunks.append(
            f"{i}. Title: {p.get('title', '')}\n"
            f"Abstract: {compact_abstract(p.get('abstract', ''), max_chars)}"
        )
    return "\n".join(chunks)


def build_prompt(user_id: str, user_ctx: str, candidates: list[dict[str, Any]]) -> str:
    cand_blocks = []
    for cand in candidates:
        cand_blocks.append(
            f"Candidate id: {cand['author_id']}\n"
            f"Candidate recent papers:\n{cand['context']}"
        )
    return f"""User author id: {user_id}

User recent papers:
{user_ctx}

Potential collaborators to evaluate (unordered):
{chr(10).join(cand_blocks)}

Evaluate the potential relevance of collaboration between the user and each candidate."""


def graded_ndcg(scores: list[int]) -> float:
    if not scores:
        return 0.0
    gains = np.array([2**s - 1 for s in scores], dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, len(scores) + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(gains)[::-1]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg else 0.0


def normalize_score(value: Any) -> int:
    try:
        score = int(value)
    except Exception:
        score = 0
    return max(0, min(3, score))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = RESULTS / "llm_recs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print("[skip] LLM recommendation eval exists; use --force")
        return 0

    papers = load_papers_in_embedding_order()
    author_ids = load_json(PREDICTIONS / "author_ids.json")
    history_index = load_json(PREDICTIONS / "author_history_index.json")
    pred = np.load(PREDICTIONS / "author_pred_emb.npy").astype(np.float32, copy=False)
    pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    author_pos = {aid: i for i, aid in enumerate(author_ids)}
    coauthors, papers_by_author = build_author_maps()
    active_2026 = [
        aid
        for aid in author_ids
        if aid in history_index
        and any(int(papers[idx]["year"]) == 2026 for idx in papers_by_author.get(aid, set()))
    ]
    rng = random.Random(args.seed)
    user_pool = rng.sample(active_2026, min(args.n_users, len(active_2026)))
    save_json(out_dir / "user_pool.json", {"seed": args.seed, "user_ids": user_pool})

    recommendations = []
    all_author_set = set(author_ids)
    for uid in user_pool:
        u_pos = author_pos[uid]
        excluded = {uid} | (coauthors.get(uid, set()) & all_author_set)
        candidate_positions = [i for i, aid in enumerate(author_ids) if aid not in excluded]
        scores = pred[candidate_positions] @ pred[u_pos]
        top_local = np.argsort(-scores)[:10]
        top = []
        for local in top_local:
            pos = candidate_positions[int(local)]
            aid = author_ids[pos]
            top.append({"author_id": aid, "cosine": float(scores[int(local)])})
        recommendations.append({"user_id": uid, "candidates": top})
    save_json(out_dir / "recommendations.json", recommendations)

    cache = LLMCache()
    llm_rows = []
    per_user = []
    for rec in tqdm(recommendations, desc="llm recs"):
        uid = rec["user_id"]
        user_ctx = paper_context(history_index[uid]["last5_paper_idxs"], papers)
        candidates = []
        for cand in rec["candidates"]:
            aid = cand["author_id"]
            candidates.append(
                {
                    "author_id": aid,
                    "context": paper_context(history_index[aid]["last5_paper_idxs"], papers),
                }
            )
        shuffled = stable_shuffle(candidates, uid)
        response = cache.complete_json(SYSTEM_PROMPT, build_prompt(uid, user_ctx, shuffled))
        rating_by_id = {
            str(r.get("candidate_id")): {
                "score": normalize_score(r.get("score")),
                "reason": str(r.get("reason", "")).strip(),
            }
            for r in response.get("ratings", [])
        }
        ordered_scores = []
        for rank, cand in enumerate(rec["candidates"], start=1):
            rating = rating_by_id.get(cand["author_id"], {"score": 0, "reason": "missing rating"})
            score = int(rating["score"])
            ordered_scores.append(score)
            llm_rows.append(
                {
                    "user_id": uid,
                    "rank": rank,
                    "candidate_id": cand["author_id"],
                    "model_cosine": cand["cosine"],
                    "llm_score": score,
                    "reason": rating["reason"],
                }
            )
        per_user.append(
            {
                "user_id": uid,
                "llm_ndcg@10": graded_ndcg(ordered_scores),
                "llm_precision@10_ge2": float(np.mean([s >= 2 for s in ordered_scores])),
                "mean_llm_score@10": float(np.mean(ordered_scores)),
            }
        )

    save_json(out_dir / "llm_eval.json", {"ratings": llm_rows, "per_user": per_user})
    df_rows = pd.DataFrame(llm_rows)
    df_users = pd.DataFrame(per_user)
    df_rows.to_csv(out_dir / "ratings.csv", index=False)
    df_users.to_csv(out_dir / "summary.csv", index=False)
    summary = {
        "n_users": int(len(per_user)),
        "n_rated_pairs": int(len(llm_rows)),
        "llm_ndcg@10": float(df_users["llm_ndcg@10"].mean()),
        "llm_precision@10_ge2": float(df_users["llm_precision@10_ge2"].mean()),
        "mean_llm_score@10": float(df_users["mean_llm_score@10"].mean()),
    }
    save_json(summary_path, summary)

    plt.figure(figsize=(6, 4))
    df_rows["llm_score"].value_counts().sort_index().reindex([0, 1, 2, 3], fill_value=0).plot(kind="bar")
    plt.xlabel("LLM relevance score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "score_histogram.png", dpi=160)
    plt.close()
    print(f"[done] {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
