from __future__ import annotations

import argparse
import sys

import numpy as np

from common.compat import META_ALL_VARIANTS, load_json, save_json, topic_dir, variant_dir


def coherence_proxy(name: str, force: bool) -> None:
    out_path = topic_dir(name) / "coherence.json"
    if out_path.exists() and not force:
        return
    labels_llm = load_json(topic_dir(name) / "llm_label.json")
    reps = load_json(topic_dir(name) / "representative_papers.json")
    hard_labels = np.load(variant_dir(name) / "labels.npy")
    rows = {}
    weighted_sum = 0.0
    weight_total = 0
    scores = []
    for cluster_id in sorted(labels_llm, key=lambda x: int(x)):
        size = int((hard_labels == int(cluster_id)).sum())
        n_reps = len(reps.get(cluster_id, []))
        score = 3 if n_reps >= 8 and size >= 3 else 2 if n_reps >= 4 else 1
        rows[cluster_id] = {
            "score": score,
            "reason": "Proxy coherence: based on representative coverage; replace with LLM score when API is available.",
            "size": size,
        }
        weighted_sum += score * size
        weight_total += size
        scores.append(score)
    result = {
        "variant": name,
        "n_clusters": len(labels_llm),
        "weighted_mean": float(weighted_sum / weight_total) if weight_total else None,
        "unweighted_mean": float(np.mean(scores)) if scores else None,
        "clusters": rows,
        "proxy": True,
    }
    save_json(out_path, result)


def distinctness_proxy(name: str, force: bool, tau: float, duplicate_tau: float, max_pairs: int) -> None:
    out_path = topic_dir(name) / "distinctness.json"
    if out_path.exists() and not force:
        return
    centroids = np.load(variant_dir(name) / "centroids_qwen.npy").astype(np.float32, copy=False)
    sim = centroids @ centroids.T
    iu = np.triu_indices(sim.shape[0], k=1)
    pairs = [
        (int(i), int(j), float(s))
        for i, j, s in zip(iu[0], iu[1], sim[iu], strict=False)
        if float(s) >= tau
    ]
    pairs.sort(key=lambda x: x[2], reverse=True)
    rows = []
    n_dup = 0
    for i, j, score in pairs[:max_pairs]:
        is_dup = score >= duplicate_tau
        n_dup += int(is_dup)
        rows.append(
            {
                "cluster_a": i,
                "cluster_b": j,
                "cosine_similarity": score,
                "is_duplicate": is_dup,
                "reason": "Proxy distinctness: duplicate if centroid cosine is extremely high; replace with LLM judgement when API is available.",
            }
        )
    result = {
        "variant": name,
        "tau": tau,
        "max_pairs": max_pairs,
        "n_pairs": len(rows),
        "n_duplicates": n_dup,
        "duplicate_pair_rate": float(n_dup / len(rows)) if rows else 0.0,
        "pairs": rows,
        "proxy": True,
    }
    save_json(out_path, result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=[v.name for v in META_ALL_VARIANTS])
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--duplicate-tau", type=float, default=0.97)
    parser.add_argument("--max-pairs", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for name in args.variants:
        coherence_proxy(name, args.force)
        distinctness_proxy(name, args.force, args.tau, args.duplicate_tau, args.max_pairs)
        print(f"[done] proxy metrics {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
