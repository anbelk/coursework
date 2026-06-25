from __future__ import annotations

import argparse
import sys

import numpy as np
from tqdm import tqdm

from common.compat import ALL_VARIANTS, save_json, topic_dir, variant_dir


def assignment_entropy(proba: np.ndarray, labels: np.ndarray, method: str) -> tuple[float, float]:
    p = np.clip(proba.astype(np.float64, copy=False), 0.0, 1.0)
    if method == "hdbscan":
        noise = np.clip(1.0 - p.sum(axis=1, keepdims=True), 0.0, 1.0)
        p = np.concatenate([p, noise], axis=1)
    h = -(p * np.log(np.maximum(p, 1e-12))).sum(axis=1)
    denom = np.log(p.shape[1]) if p.shape[1] > 1 else 1.0
    return float(h.mean()), float((h / denom).mean())


def run_variant(name: str, force: bool) -> None:
    out_path = topic_dir(name) / "metrics_quant.json"
    if out_path.exists() and not force:
        print(f"[skip] {name} quant metrics exist; use --force")
        return
    labels = np.load(variant_dir(name) / "labels.npy")
    proba = np.load(variant_dir(name) / "proba.npy")
    if "hdbscan" in name:
        method = "hdbscan"
    elif "fcm" in name:
        method = "fcm"
    elif "gmm" in name:
        method = "gmm"
    elif name.startswith("random_"):
        method = "random"
    elif name.startswith("kmeans_"):
        method = "kmeans"
    else:
        method = name.split("_", 1)[0]
    k = int(proba.shape[1])
    non_noise_labels = labels[labels >= 0]
    sizes = np.array([(non_noise_labels == i).sum() for i in range(k)], dtype=np.int64)
    nonzero_sizes = sizes[sizes > 0]
    n_non_noise = int(non_noise_labels.size)
    top5 = np.sort(sizes)[-5:].sum() if len(sizes) else 0
    entropy, entropy_norm = assignment_entropy(proba, labels, method)
    result = {
        "variant": name,
        "method": method,
        "n_clusters": k,
        "mean_assignment_entropy": entropy,
        "mean_assignment_entropy_norm": entropy_norm,
        "size_p25": float(np.percentile(nonzero_sizes, 25)) if len(nonzero_sizes) else 0.0,
        "size_p50": float(np.percentile(nonzero_sizes, 50)) if len(nonzero_sizes) else 0.0,
        "size_p75": float(np.percentile(nonzero_sizes, 75)) if len(nonzero_sizes) else 0.0,
        "top5_concentration": float(top5 / n_non_noise) if n_non_noise else 0.0,
        "noise_ratio": float((labels == -1).mean()) if method == "hdbscan" else 0.0,
        "n_non_noise": n_non_noise,
        "cluster_sizes": {str(i): int(sizes[i]) for i in range(k)},
    }
    save_json(out_path, result)
    print(f"[done] {name}: entropy_norm={entropy_norm:.3f}, top5={result['top5_concentration']:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=[v.name for v in ALL_VARIANTS])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for name in tqdm(args.variants, desc="quant metrics"):
        run_variant(name, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
