from __future__ import annotations

import argparse
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from common.compat import RESULTS, save_json
from recommendation.pipeline import (
    graded_ndcg,
    resources,
    stage1_dense_retrieval,
    stage2_relevance_scoring,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = RESULTS / "llm_recs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print("[skip] LLM recommendation eval exists; use --force")
        return 0

    res = resources()
    rng = random.Random(args.seed)
    user_pool = rng.sample(res["active_2026"], min(args.n_users, len(res["active_2026"])))
    save_json(out_dir / "user_pool.json", {"seed": args.seed, "user_ids": user_pool})

    recommendations = []
    llm_rows = []
    per_user = []
    for uid in tqdm(user_pool, desc="llm recs"):
        candidates = stage1_dense_retrieval(uid, top_n=10)
        recommendations.append(
            {
                "user_id": uid,
                "candidates": [
                    {"author_id": cand["author_id"], "cosine": cand["model_cosine"]}
                    for cand in candidates
                ],
            }
        )
        scored = stage2_relevance_scoring(uid, candidates, n_workers=args.workers)
        ordered_scores = []
        for rank, cand in enumerate(scored, start=1):
            score = int(cand["llm_score"])
            ordered_scores.append(score)
            llm_rows.append(
                {
                    "user_id": uid,
                    "rank": rank,
                    "candidate_id": cand["author_id"],
                    "candidate_name": cand["name"],
                    "model_cosine": cand["model_cosine"],
                    "llm_score": score,
                    "reason": cand.get("llm_reason", ""),
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

    save_json(out_dir / "recommendations.json", recommendations)
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
