from __future__ import annotations

import argparse
import sys

import hdbscan
import numpy as np
from tqdm import tqdm

from common.compat import (
    HDBSCAN_VARIANTS,
    REDUCED,
    hdbscan_labels_from_proba,
    load_embeddings,
    save_variant_artifacts,
    variant_dir,
    weighted_centroids,
)


def run_variant(features: np.ndarray, embeddings: np.ndarray, variant, force: bool) -> None:
    out_dir = variant_dir(variant.name)
    if (out_dir / "labels.npy").exists() and (out_dir / "proba.npy").exists() and not force:
        print(f"[skip] {variant.name} exists; use --force to recompute")
        return

    params = dict(variant.params or {})
    model = hdbscan.HDBSCAN(**params, core_dist_n_jobs=-1)
    labels = model.fit_predict(features).astype(np.int64)
    # Soft membership via HDBSCAN's per-point confidence `probabilities_` (computed
    # during fit, free). all_points_membership_vectors / membership_vector are O(N*K)
    # and take ~5.5h/variant at 509k points, so we use the confidence-weighted hard
    # assignment: proba[i, label_i] = probabilities_[i]; noise points (label -1) get an
    # all-zero row -> noise weight 1 in q_with_noise. Entropy reflects assignment
    # confidence.
    probs = model.probabilities_.astype(np.float32)
    k = int(labels.max()) + 1 if labels.max() >= 0 else 1
    membership = np.zeros((len(labels), k), dtype=np.float32)
    in_cluster = labels >= 0
    membership[np.where(in_cluster)[0], labels[in_cluster]] = probs[in_cluster]
    centroids = weighted_centroids(embeddings, membership, labels)

    save_variant_artifacts(
        variant.name,
        {
            "method": "hdbscan",
            "feature_space": "umap_n10",
            **params,
            "n_clusters": int(membership.shape[1]),
            "noise_ratio": float((labels == -1).mean()),
        },
        labels,
        membership,
        centroids,
    )
    print(
        f"[done] {variant.name}: K={membership.shape[1]}, "
        f"noise={(labels == -1).mean():.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    features = np.load(REDUCED / "umap_n10.npy").astype(np.float32, copy=False)
    embeddings = load_embeddings()
    for variant in tqdm(HDBSCAN_VARIANTS, desc="hdbscan variants"):
        run_variant(features, embeddings, variant, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
