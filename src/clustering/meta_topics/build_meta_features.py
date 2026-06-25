from __future__ import annotations

import argparse
import sys

import numpy as np
import umap
from sklearn.decomposition import PCA

from common.compat import DATA, RANDOM_STATE, l2_normalize, save_json


META_DIR = DATA / "meta"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--umap-components", type=int, default=5)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--pca-var", type=float, default=0.85)
    parser.add_argument("--pca-max-components", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    embeddings = l2_normalize(np.load(META_DIR / "cluster_embeddings.npy").astype(np.float32, copy=False))
    umap_path = META_DIR / "umap5.npy"
    pca_path = META_DIR / "pca.npy"
    if umap_path.exists() and pca_path.exists() and not args.force:
        print("[skip] meta features exist; use --force")
        return 0

    reducer = umap.UMAP(
        n_components=args.umap_components,
        n_neighbors=args.umap_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
        verbose=True,
    )
    umap_features = reducer.fit_transform(embeddings).astype(np.float32)
    np.save(umap_path, umap_features)
    save_json(
        META_DIR / "umap_meta.json",
        {
            "shape": list(umap_features.shape),
            "n_components": args.umap_components,
            "n_neighbors": args.umap_neighbors,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": RANDOM_STATE,
        },
    )

    max_components = min(args.pca_max_components, embeddings.shape[0], embeddings.shape[1])
    pca_full = PCA(n_components=max_components, random_state=RANDOM_STATE)
    full = pca_full.fit_transform(embeddings)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    keep = int(np.searchsorted(cumsum, args.pca_var) + 1)
    keep = min(max(1, keep), max_components)
    pca_features = full[:, :keep].astype(np.float32)
    np.save(pca_path, pca_features)
    save_json(
        META_DIR / "pca_meta.json",
        {
            "shape": list(pca_features.shape),
            "target_variance": args.pca_var,
            "max_components": args.pca_max_components,
            "selected_components": keep,
            "explained_variance": float(cumsum[keep - 1]),
            "random_state": RANDOM_STATE,
        },
    )
    print(f"[done] meta features: umap={umap_features.shape}, pca={pca_features.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
