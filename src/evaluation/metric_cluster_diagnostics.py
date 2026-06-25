from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
from tqdm import tqdm

from common.compat import EMBEDDINGS_PATH, load_json, save_json, topic_dir, variant_dir


DEFAULT_VARIANTS = [
    "hdbscan_fine",
    "hdbscan_medium",
    "kmeans_umap10_k979",
    "kmeans_umap10_k242",
]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def mean_pairwise_cosine(x: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    x = l2_normalize(x)
    sim = x @ x.T
    n = len(x)
    return float((sim.sum() - n) / (n * (n - 1)))


def assignment_confidence(labels: np.ndarray, proba: np.ndarray, chunk_size: int = 65_536) -> np.ndarray:
    confidence = np.zeros((len(labels),), dtype=np.float32)
    for start in range(0, len(labels), chunk_size):
        end = min(start + chunk_size, len(labels))
        lab = labels[start:end]
        good = (lab >= 0) & (lab < proba.shape[1])
        if not np.any(good):
            continue
        rows = np.arange(start, end, dtype=np.int64)[good]
        cols = lab[good].astype(np.int64, copy=False)
        confidence[start:end][good] = np.asarray(proba[rows, cols], dtype=np.float32)
    return confidence


def size_balance_entropy(labels: np.ndarray, k: int) -> tuple[float, np.ndarray]:
    non_noise = labels[labels >= 0]
    sizes = np.bincount(non_noise, minlength=k).astype(np.float64)
    total = float(sizes.sum())
    if total <= 0 or k <= 1:
        return 0.0, sizes
    p = sizes[sizes > 0] / total
    entropy = float(-(p * np.log(p)).sum() / np.log(k))
    return entropy, sizes


def embedding_coherence_lift(
    labels: np.ndarray,
    embeddings: np.ndarray,
    sizes: np.ndarray,
    rng: np.random.Generator,
    docs_per_cluster: int,
    max_clusters: int | None,
) -> dict[str, Any]:
    cluster_ids = np.flatnonzero(sizes >= 2)
    if max_clusters is not None and len(cluster_ids) > max_clusters:
        cluster_ids = np.sort(rng.choice(cluster_ids, size=max_clusters, replace=False))

    non_noise_idx = np.flatnonzero(labels >= 0)
    rows = []
    for cluster_id in tqdm(cluster_ids.tolist(), desc="embedding lift", leave=False):
        idxs = np.flatnonzero(labels == int(cluster_id))
        n = min(int(docs_per_cluster), len(idxs))
        if n < 2:
            continue
        chosen = rng.choice(idxs, size=n, replace=False)
        random_chosen = rng.choice(non_noise_idx, size=n, replace=False)
        cluster_score = mean_pairwise_cosine(np.asarray(embeddings[chosen], dtype=np.float32))
        random_score = mean_pairwise_cosine(np.asarray(embeddings[random_chosen], dtype=np.float32))
        if cluster_score is None or random_score is None:
            continue
        rows.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(sizes[int(cluster_id)]),
                "n_sampled_docs": int(n),
                "cluster_coherence": cluster_score,
                "random_same_size_coherence": random_score,
                "lift": cluster_score - random_score,
            }
        )

    if not rows:
        return {
            "embedding_coherence": None,
            "random_same_size_coherence": None,
            "embedding_coherence_lift": None,
            "embedding_coherence_lift_weighted": None,
            "n_clusters_evaluated": 0,
            "clusters": [],
        }

    weights = np.array([row["size"] for row in rows], dtype=np.float64)
    lifts = np.array([row["lift"] for row in rows], dtype=np.float64)
    return {
        "embedding_coherence": float(np.mean([row["cluster_coherence"] for row in rows])),
        "random_same_size_coherence": float(np.mean([row["random_same_size_coherence"] for row in rows])),
        "embedding_coherence_lift": float(np.mean(lifts)),
        "embedding_coherence_lift_weighted": float(np.average(lifts, weights=weights)),
        "n_clusters_evaluated": len(rows),
        "clusters": rows,
    }


def run_variant(name: str, args: argparse.Namespace) -> None:
    out_path = topic_dir(name) / "intrinsic_diagnostics.json"
    if out_path.exists() and not args.force:
        print(f"[skip] {name} diagnostics exist; use --force")
        return

    labels = np.load(variant_dir(name) / "labels.npy", mmap_mode="r")
    proba = np.load(variant_dir(name) / "proba.npy", mmap_mode="r")
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    k = int(proba.shape[1])
    balance, sizes = size_balance_entropy(np.asarray(labels), k)
    confidence = assignment_confidence(np.asarray(labels), proba)
    assigned_conf = confidence[np.asarray(labels) >= 0]
    rng = np.random.default_rng(args.seed)
    lift = embedding_coherence_lift(
        np.asarray(labels),
        embeddings,
        sizes,
        rng,
        args.docs_per_cluster,
        args.max_clusters,
    )

    quant_path = topic_dir(name) / "metrics_quant.json"
    quant = load_json(quant_path) if quant_path.exists() else {}
    result = {
        "variant": name,
        "n_clusters": k,
        "size_balance_entropy": balance,
        "assignment_confidence_mean": float(assigned_conf.mean()) if len(assigned_conf) else None,
        "assignment_confidence_p25": float(np.percentile(assigned_conf, 25)) if len(assigned_conf) else None,
        "assignment_confidence_p50": float(np.percentile(assigned_conf, 50)) if len(assigned_conf) else None,
        "assignment_confidence_p75": float(np.percentile(assigned_conf, 75)) if len(assigned_conf) else None,
        "mean_assignment_entropy_norm": quant.get("mean_assignment_entropy_norm"),
        "noise_ratio": float((np.asarray(labels) == -1).mean()),
        "docs_per_cluster": args.docs_per_cluster,
        "seed": args.seed,
        **lift,
    }
    save_json(out_path, result)
    print(
        f"[done] {name}: lift={result['embedding_coherence_lift']} "
        f"balance={balance:.4f} confidence={result['assignment_confidence_mean']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute non-LLM intrinsic clustering diagnostics")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--docs-per-cluster", type=int, default=50)
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for name in args.variants:
        run_variant(name, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
