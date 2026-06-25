from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

from common.compat import RESULTS, save_json
from recommendation.pipeline import resources, run_pipeline


def evaluate_user(user_id: str, top_n: int, top_k: int, llm_workers: int) -> dict:
    try:
        result = run_pipeline(user_id, top_n=top_n, top_k=top_k, n_workers=llm_workers)
        selected = result.get("selected", [])
        scores = [int(c.get("llm_score", 0)) for c in result.get("candidates", [])]
        return {
            "user_id": user_id,
            "success": bool(selected),
            "n_selected": len(selected),
            "has_enough": bool(result.get("has_enough", False)),
            "top_score": max(scores) if scores else 0,
            "error": "",
        }
    except Exception as exc:
        return {
            "user_id": user_id,
            "success": False,
            "n_selected": 0,
            "has_enough": False,
            "top_score": 0,
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--pipeline-workers", type=int, default=4)
    parser.add_argument("--llm-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = RESULTS / "recs_coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print("[skip] recommendation coverage exists; use --force")
        return 0

    active = resources()["active_2026"]
    rng = random.Random(args.seed)
    user_pool = rng.sample(active, min(args.n_users, len(active)))

    rows = []
    with ThreadPoolExecutor(max_workers=args.pipeline_workers) as executor:
        futures = [
            executor.submit(evaluate_user, uid, args.top_n, args.top_k, args.llm_workers)
            for uid in user_pool
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="coverage"):
            rows.append(future.result())

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "coverage_per_user.csv", index=False)
    summary = {
        "n_users": int(len(df)),
        "top_n": args.top_n,
        "top_k": args.top_k,
        "coverage": float(df["success"].mean()) if len(df) else 0.0,
        "enough_recommendations_rate": float(df["has_enough"].mean()) if len(df) else 0.0,
        "mean_top_score": float(df["top_score"].mean()) if len(df) else 0.0,
        "median_top_score": float(np.median(df["top_score"])) if len(df) else 0.0,
        "n_errors": int((df["error"] != "").sum()) if len(df) else 0,
    }
    save_json(summary_path, summary)
    print(f"[done] {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
