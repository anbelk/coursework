from __future__ import annotations

import argparse
import sys

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from tqdm import tqdm

from common.compat import (
    BASELINE_VARIANTS,
    RANDOM_STATE,
    REDUCED,
    load_embeddings,
    save_variant_artifacts,
    variant_dir,
    weighted_centroids,
)


def run_random(features: np.ndarray, embeddings: np.ndarray, variant, force: bool) -> None:
    out_dir = variant_dir(variant.name)
    if (out_dir / "labels.npy").exists() and (out_dir / "proba.npy").exists() and not force:
        print(f"[skip] {variant.name} exists; use --force to recompute")
        return

    k = int(variant.k)
    rng = np.random.default_rng(RANDOM_STATE)
    labels = rng.integers(0, k, size=len(features), dtype=np.int32)
    proba = np.zeros((len(labels), k), dtype=np.float32)
    proba[np.arange(len(labels)), labels] = 1.0
    centroids = weighted_centroids(embeddings, proba, labels)
    save_variant_artifacts(
        variant.name,
        {
            "method": "random",
            "feature_space": "umap_n10",
            "k": k,
            "random_state": RANDOM_STATE,
        },
        labels,
        proba,
        centroids,
    )
    print(f"[done] {variant.name}: K={k}, random baseline")


def run_kmeans(features: np.ndarray, embeddings: np.ndarray, variant, force: bool) -> None:
    out_dir = variant_dir(variant.name)
    if (out_dir / "labels.npy").exists() and (out_dir / "proba.npy").exists() and not force:
        print(f"[skip] {variant.name} exists; use --force to recompute")
        return

    k = int(variant.k)
    # MiniBatchKMeans scales to ~500k points and a K-grid up to 1000; exact KMeans
    # with n_init=10 would be prohibitively slow at large K on this corpus.
    n = len(features)
    if n > 100_000:
        model = MiniBatchKMeans(
            n_clusters=k, random_state=RANDOM_STATE, n_init=3,
            batch_size=4096, max_iter=200,
        )
    else:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(features).astype(np.int32)
    proba = np.zeros((len(labels), k), dtype=np.float32)
    proba[np.arange(len(labels)), labels] = 1.0
    centroids = weighted_centroids(embeddings, proba, labels)
    save_variant_artifacts(
        variant.name,
        {
            "method": "kmeans",
            "feature_space": "umap_n10",
            "k": k,
            "random_state": RANDOM_STATE,
            "algo": type(model).__name__,
            "inertia": float(model.inertia_),
        },
        labels,
        proba,
        centroids,
    )
    print(f"[done] {variant.name}: K={k}, inertia={model.inertia_:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    features = np.load(REDUCED / "umap_n10.npy").astype(np.float32, copy=False)
    embeddings = load_embeddings()
    runners = {
        "random": run_random,
        "kmeans": run_kmeans,
    }
    for variant in tqdm(BASELINE_VARIANTS, desc="cluster baselines"):
        runners[variant.method](features, embeddings, variant, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
