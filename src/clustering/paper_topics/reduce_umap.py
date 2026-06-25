from __future__ import annotations

import argparse
import shutil
import sys

import numpy as np
import umap

from common.compat import RANDOM_STATE, REDUCED, ensure_dirs, load_embeddings, load_json, save_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--min-dist", type=float, default=0.0)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--n-neighbors", type=int, default=None)
    parser.add_argument(
        "--fast", action="store_true",
        help="Allow UMAP parallelism (random_state=None): much faster on large N, "
             "at the cost of exact reproducibility.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    out_path = REDUCED / "umap_n10.npy"
    meta_path = REDUCED / "umap_meta.json"
    if out_path.exists() and meta_path.exists() and not args.force:
        print(f"[skip] {out_path} exists; use --force to recompute")
        return 0

    if args.n_neighbors is None:
        sweep = load_json(REDUCED / "umap_sweep.json")
        n_neighbors = int(sweep["selected_n_neighbors"])
    else:
        n_neighbors = args.n_neighbors

    cached = REDUCED / f"umap_neighbors_{n_neighbors}_n{args.n_components}.npy"
    if cached.exists() and not args.force:
        shutil.copyfile(cached, out_path)
        features = np.load(out_path)
        source = str(cached)
    else:
        embeddings = load_embeddings()
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=args.n_components,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=None if args.fast else RANDOM_STATE,
            low_memory=True,
        )
        features = reducer.fit_transform(embeddings).astype(np.float32)
        np.save(out_path, features)
        source = "computed"

    save_json(
        meta_path,
        {
            "n_neighbors": n_neighbors,
            "n_components": args.n_components,
            "min_dist": args.min_dist,
            "metric": args.metric,
            "random_state": None if args.fast else RANDOM_STATE,
            "shape": list(features.shape),
            "source": source,
        },
    )
    print(f"[done] wrote {out_path} shape={features.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
