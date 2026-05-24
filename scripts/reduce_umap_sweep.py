from __future__ import annotations

import argparse
import sys
import time

import hdbscan
import numpy as np
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import umap

from pipeline_common import RANDOM_STATE, REDUCED, ensure_dirs, load_embeddings, save_json


def relative_range(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean())
    if mean <= 1e-12:
        return 0.0
    return float((arr.max() - arr.min()) / mean)


def cluster_metrics(features: np.ndarray, sample_size: int) -> dict:
    model = hdbscan.HDBSCAN(
        min_cluster_size=30,
        min_samples=5,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = model.fit_predict(features)
    non_noise = labels >= 0
    clusters = sorted(int(x) for x in np.unique(labels[non_noise]))
    sizes = np.array([(labels == c).sum() for c in clusters], dtype=np.int64)

    sil = None
    if len(clusters) > 1 and int(non_noise.sum()) > 100:
        idx = np.flatnonzero(non_noise)
        if len(idx) > sample_size:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(idx, size=sample_size, replace=False)
        try:
            sil = float(silhouette_score(features[idx], labels[idx]))
        except ValueError:
            sil = None

    return {
        "n_clusters": int(len(clusters)),
        "noise_ratio": float(1.0 - non_noise.mean()),
        "size_p50": float(np.percentile(sizes, 50)) if len(sizes) else 0.0,
        "size_p95": float(np.percentile(sizes, 95)) if len(sizes) else 0.0,
        "silhouette_sample": sil,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--neighbors", nargs="+", type=int, default=[15, 30, 50])
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--min-dist", type=float, default=0.0)
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    out_path = REDUCED / "umap_sweep.json"
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path} exists; use --force to recompute")
        return 0

    embeddings = load_embeddings()
    results = []
    for n_neighbors in tqdm(args.neighbors, desc="umap sweep"):
        t0 = time.time()
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=args.n_components,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=RANDOM_STATE,
            low_memory=True,
        )
        features = reducer.fit_transform(embeddings).astype(np.float32)
        np.save(REDUCED / f"umap_neighbors_{n_neighbors}_n{args.n_components}.npy", features)
        metrics = cluster_metrics(features, args.sample_size)
        metrics.update(
            {
                "n_neighbors": n_neighbors,
                "n_components": args.n_components,
                "min_dist": args.min_dist,
                "metric": args.metric,
                "elapsed_sec": round(time.time() - t0, 3),
            }
        )
        print(f"[info] n_neighbors={n_neighbors}: {metrics}")
        results.append(metrics)

    k_spread = relative_range([r["n_clusters"] for r in results])
    noise_spread = relative_range([r["noise_ratio"] for r in results])
    stable = k_spread < 0.15 and noise_spread < 0.15
    if stable:
        selected = 30 if 30 in args.neighbors else args.neighbors[len(args.neighbors) // 2]
        reason = "relative spread in K and noise_ratio < 15%; fixed default n_neighbors=30"
    else:
        eligible = [r for r in results if r["n_clusters"] > 0]
        selected_row = min(
            eligible,
            key=lambda r: (r["noise_ratio"], abs(r["n_clusters"] - np.median([x["n_clusters"] for x in eligible]))),
        )
        selected = int(selected_row["n_neighbors"])
        reason = "spread was not small; selected by low noise and moderate cluster count"

    save_json(
        out_path,
        {
            "results": results,
            "k_relative_spread": k_spread,
            "noise_relative_spread": noise_spread,
            "stable": stable,
            "selected_n_neighbors": selected,
            "selection_reason": reason,
        },
    )
    print(f"[done] selected n_neighbors={selected}; wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
