from __future__ import annotations

import argparse
import sys

import numpy as np
from sklearn.decomposition import PCA

from common.compat import RANDOM_STATE, REDUCED, ensure_dirs, load_embeddings, save_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variance", type=float, default=0.85)
    parser.add_argument("--max-components", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    out_path = REDUCED / "pca_features.npy"
    meta_path = REDUCED / "pca_meta.json"
    if out_path.exists() and meta_path.exists() and not args.force:
        print(f"[skip] {out_path} exists; use --force to recompute")
        return 0

    embeddings = load_embeddings()
    pca_full = PCA(n_components=args.max_components, random_state=RANDOM_STATE, svd_solver="randomized")
    features_full = pca_full.fit_transform(embeddings).astype(np.float32)
    cumsum = np.cumsum(pca_full.explained_variance_ratio_)
    n_for_variance = int(np.searchsorted(cumsum, args.variance) + 1)
    n_components = min(args.max_components, n_for_variance)
    features = features_full[:, :n_components].astype(np.float32, copy=False)
    np.save(out_path, features)
    save_json(
        meta_path,
        {
            "target_variance": args.variance,
            "max_components": args.max_components,
            "n_components": n_components,
            "explained_variance_sum": float(cumsum[n_components - 1]),
            "explained_variance_ratio": pca_full.explained_variance_ratio_[:n_components].tolist(),
            "shape": list(features.shape),
            "random_state": RANDOM_STATE,
        },
    )
    print(
        f"[done] wrote {out_path} shape={features.shape}; "
        f"variance={cumsum[n_components - 1]:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
