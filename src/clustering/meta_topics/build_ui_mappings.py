from __future__ import annotations

import argparse
import sys

import numpy as np
import umap

from common.compat import DATA, RANDOM_STATE, load_embeddings, load_json, save_json, variant_dir


LAYOUT_DIR = DATA / "layout"
HIERARCHY_DIR = DATA / "hierarchy"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="hdbscan_fine")
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--min-dist", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    xy_path = LAYOUT_DIR / "papers_xy.npy"
    centers_path = LAYOUT_DIR / "cluster_centers.json"
    if xy_path.exists() and centers_path.exists() and not args.force:
        print("[skip] layout exists; use --force")
        return 0

    LAYOUT_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = load_embeddings()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric="cosine",
        random_state=RANDOM_STATE,
        verbose=True,
    )
    xy = reducer.fit_transform(embeddings).astype(np.float32)
    np.save(xy_path, xy)

    labels = np.load(variant_dir(args.variant) / "labels.npy")
    tree = load_json(HIERARCHY_DIR / "tree.json")
    fine_to_papers = {
        cid: np.flatnonzero(labels == cid).astype(np.int64)
        for cid in range(int(labels.max()) + 1)
    }
    centers = {}
    for node_id, node in tree["nodes"].items():
        paper_idxs = np.concatenate(
            [fine_to_papers.get(int(cid), np.array([], dtype=np.int64)) for cid in node["fine_cluster_ids"]]
        )
        if len(paper_idxs):
            center = np.median(xy[paper_idxs], axis=0)
        else:
            center = np.array([0.0, 0.0], dtype=np.float32)
        centers[node_id] = {
            "x": float(center[0]),
            "y": float(center[1]),
            "paper_count": int(node["paper_count"]),
        }

    save_json(centers_path, centers)
    save_json(
        LAYOUT_DIR / "params.json",
        {
            "variant": args.variant,
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "metric": "cosine",
            "random_state": RANDOM_STATE,
            "shape": list(xy.shape),
        },
    )
    print(f"[done] layout: papers={xy.shape[0]} centers={len(centers)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
