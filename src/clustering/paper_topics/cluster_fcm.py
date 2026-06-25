from __future__ import annotations

import argparse
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

from common.compat import (
    FCM_VARIANTS,
    RANDOM_STATE,
    hard_labels_from_proba,
    l2_normalize,
    load_embeddings,
    save_variant_artifacts,
    variant_dir,
)


def initialize_centroids(x: np.ndarray, k: int) -> np.ndarray:
    km = MiniBatchKMeans(
        n_clusters=k,
        init="k-means++",
        n_init=3,
        batch_size=4096,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    km.fit(x)
    return l2_normalize(km.cluster_centers_.astype(np.float32))


def update_membership(x: np.ndarray, centroids: np.ndarray, m: float) -> np.ndarray:
    sim = x @ centroids.T
    dist = np.maximum(1.0 - sim, 1e-8).astype(np.float32)
    power = 1.0 / (m - 1.0)
    inv = dist ** (-power)
    membership = inv / np.maximum(inv.sum(axis=1, keepdims=True), 1e-12)
    return membership.astype(np.float32)


def fit_spherical_fcm(
    x: np.ndarray,
    k: int,
    m: float,
    max_iter: int,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    centroids = initialize_centroids(x, k)
    last_shift = np.inf
    membership = update_membership(x, centroids, m)
    for it in range(1, max_iter + 1):
        weights = membership ** m
        new_centroids = weights.T @ x
        new_centroids /= np.maximum(weights.sum(axis=0)[:, None], 1e-12)
        new_centroids = l2_normalize(new_centroids.astype(np.float32))
        last_shift = float(np.linalg.norm(new_centroids - centroids, axis=1).max())
        centroids = new_centroids
        membership = update_membership(x, centroids, m)
        if last_shift < tol:
            return membership, centroids, it, last_shift
    return membership, centroids, max_iter, last_shift


def run_variant(x: np.ndarray, variant, force: bool, m: float, max_iter: int, tol: float) -> None:
    out_dir = variant_dir(variant.name)
    if (out_dir / "labels.npy").exists() and (out_dir / "proba.npy").exists() and not force:
        print(f"[skip] {variant.name} exists; use --force to recompute")
        return

    proba, centroids, n_iter, final_shift = fit_spherical_fcm(x, int(variant.k), m, max_iter, tol)
    labels = hard_labels_from_proba(proba)
    save_variant_artifacts(
        variant.name,
        {
            "method": "spherical_fcm",
            "feature_space": "qwen_l2",
            "k": int(variant.k),
            "m": m,
            "max_iter": max_iter,
            "tol": tol,
            "n_iter": n_iter,
            "final_centroid_shift": final_shift,
            "random_state": RANDOM_STATE,
        },
        labels,
        proba,
        centroids,
    )
    print(f"[done] {variant.name}: K={variant.k}, iter={n_iter}, shift={final_shift:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    x = load_embeddings()
    for variant in tqdm(FCM_VARIANTS, desc="fcm variants"):
        run_variant(x, variant, args.force, args.m, args.max_iter, args.tol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
