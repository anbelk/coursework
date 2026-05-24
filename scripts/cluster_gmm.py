from __future__ import annotations

import argparse
import sys

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from pipeline_common import (
    GMM_VARIANTS,
    RANDOM_STATE,
    REDUCED,
    hard_labels_from_proba,
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

    k = int(variant.k)
    model = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        n_init=3,
        reg_covar=1e-6,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    labels = model.fit_predict(features).astype(np.int32)
    proba = model.predict_proba(features).astype(np.float32)
    labels = hard_labels_from_proba(proba)
    centroids = weighted_centroids(embeddings, proba, labels)
    save_variant_artifacts(
        variant.name,
        {
            "method": "gmm",
            "feature_space": "pca",
            "k": k,
            "covariance_type": "diag",
            "n_init": 3,
            "reg_covar": 1e-6,
            "random_state": RANDOM_STATE,
            "lower_bound": float(model.lower_bound_),
            "n_iter": int(model.n_iter_),
        },
        labels,
        proba,
        centroids,
    )
    print(f"[done] {variant.name}: K={k}, iter={model.n_iter_}, lower_bound={model.lower_bound_:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    features = np.load(REDUCED / "pca_features.npy").astype(np.float32, copy=False)
    embeddings = load_embeddings()
    for variant in tqdm(GMM_VARIANTS, desc="gmm variants"):
        run_variant(features, embeddings, variant, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
