from __future__ import annotations

import argparse
import sys

import hdbscan
import numpy as np
from tqdm import tqdm

from pipeline_common import (
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
    model = hdbscan.HDBSCAN(**params, prediction_data=True)
    labels_raw = model.fit_predict(features)
    membership = hdbscan.all_points_membership_vectors(model).astype(np.float32)
    if membership.ndim == 1:
        membership = membership[:, None]
    row_sum = membership.sum(axis=1, keepdims=True)
    membership = np.where(row_sum > 1.0, membership / np.maximum(row_sum, 1e-12), membership)
    labels = hdbscan_labels_from_proba(membership, labels_raw == -1)
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
