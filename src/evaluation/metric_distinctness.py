from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from common.compat import ALL_VARIANTS, LLMCache, load_json, save_json, topic_dir, variant_dir


SYSTEM_PROMPT = """You decide whether two scientific-paper clusters are duplicate topics.
Return only valid JSON:
{
  "is_duplicate": false,
  "reason": "brief explanation"
}
Mark duplicate only if the two clusters describe essentially the same research topic, not merely related fields."""


def build_prompt(label_a: dict, reps_a: list[dict], label_b: dict, reps_b: list[dict], sim: float) -> str:
    titles_a = "\n".join(f"- {r['title']}" for r in reps_a[:5])
    titles_b = "\n".join(f"- {r['title']}" for r in reps_b[:5])
    return f"""Centroid cosine similarity: {sim:.4f}

Cluster A label:
Name: {label_a.get('name', '')}
Description: {label_a.get('description', '')}
Representative titles:
{titles_a}

Cluster B label:
Name: {label_b.get('name', '')}
Description: {label_b.get('description', '')}
Representative titles:
{titles_b}

Are A and B duplicate topics?"""


def candidate_pairs(centroids: np.ndarray, tau: float, max_pairs: int) -> list[tuple[int, int, float]]:
    sim = centroids @ centroids.T
    iu = np.triu_indices(sim.shape[0], k=1)
    pairs = [
        (int(i), int(j), float(s))
        for i, j, s in zip(iu[0], iu[1], sim[iu], strict=False)
        if float(s) >= tau
    ]
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:max_pairs]


def _check_pair(
    i: int,
    j: int,
    sim: float,
    labels_llm: dict,
    reps: dict,
    cache: LLMCache,
) -> dict:
    response = cache.complete_json(
        SYSTEM_PROMPT,
        build_prompt(labels_llm[str(i)], reps[str(i)], labels_llm[str(j)], reps[str(j)], sim),
    )
    is_dup = bool(response.get("is_duplicate", False))
    return {
        "cluster_a": i,
        "cluster_b": j,
        "cosine_similarity": sim,
        "is_duplicate": is_dup,
        "reason": str(response.get("reason", "")).strip(),
    }


def run_variant(
    name: str,
    cache: LLMCache,
    force: bool,
    tau: float,
    max_pairs: int,
    workers: int = 1,
) -> None:
    out_path = topic_dir(name) / "distinctness.json"
    if out_path.exists() and not force:
        print(f"[skip] {name} distinctness exists; use --force")
        return
    centroids = np.load(variant_dir(name) / "centroids_qwen.npy").astype(np.float32, copy=False)
    labels_llm = load_json(topic_dir(name) / "llm_label.json")
    reps = load_json(topic_dir(name) / "representative_papers.json")
    pairs = candidate_pairs(centroids, tau, max_pairs)
    rows: list[dict] = []
    n_dup = 0
    if workers <= 1:
        for i, j, sim in tqdm(pairs, desc=f"distinct {name}", leave=False):
            row = _check_pair(i, j, sim, labels_llm, reps, cache)
            n_dup += int(row["is_duplicate"])
            rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_check_pair, i, j, sim, labels_llm, reps, cache) for i, j, sim in pairs
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"distinct {name}", leave=False):
                row = future.result()
                n_dup += int(row["is_duplicate"])
                rows.append(row)
    result = {
        "variant": name,
        "tau": tau,
        "max_pairs": max_pairs,
        "n_pairs": len(rows),
        "n_duplicates": n_dup,
        "duplicate_pair_rate": float(n_dup / len(rows)) if rows else None,
        "pairs": rows,
    }
    save_json(out_path, result)
    print(f"[done] {name}: pairs={len(rows)}, duplicate_rate={result['duplicate_pair_rate']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=[v.name for v in ALL_VARIANTS])
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Parallel LLM calls per variant")
    args = parser.parse_args()

    cache = LLMCache(model=args.model) if args.model else LLMCache()
    for name in args.variants:
        run_variant(name, cache, args.force, args.tau, args.max_pairs, workers=max(1, args.workers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
