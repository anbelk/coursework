from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from common.compat import DATA, l2_normalize, load_json, save_json, topic_dir, variant_dir


HIERARCHY_DIR = DATA / "hierarchy"


def fit_agglomerative(x: np.ndarray, n_clusters: int) -> np.ndarray:
    try:
        model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=n_clusters, affinity="cosine", linkage="average")
    return model.fit_predict(x).astype(np.int32)


def weighted_centroid(centroids: np.ndarray, cluster_ids: list[int], sizes: np.ndarray) -> np.ndarray:
    weights = sizes[cluster_ids].astype(np.float32)
    if float(weights.sum()) <= 0:
        weights = np.ones_like(weights)
    vec = np.average(centroids[cluster_ids], axis=0, weights=weights)
    return l2_normalize(vec[None, :])[0].astype(np.float32)


def aggregate_terms(fine_cluster_ids: list[int], fine_terms: dict[str, list[dict[str, Any]]], sizes: np.ndarray, top_n: int) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    for cid in fine_cluster_ids:
        weight = float(max(1, int(sizes[cid])))
        for item in fine_terms.get(str(cid), []):
            term = str(item.get("term", "")).strip()
            if not term:
                continue
            scores[term] += float(item.get("score", 0.0)) * weight
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"term": term, "score": float(score)} for term, score in ranked]


def label_for_fine(cluster_id: int, fine_labels: dict[str, Any], fine_terms: dict[str, list[dict[str, Any]]]) -> str:
    label = fine_labels.get(str(cluster_id), {})
    if isinstance(label, dict) and label.get("name"):
        return str(label["name"])
    terms = fine_terms.get(str(cluster_id), [])
    return str(terms[0]["term"]) if terms else f"Fine topic {cluster_id}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="hdbscan_fine")
    parser.add_argument("--level0", type=int, default=25)
    parser.add_argument("--level1", type=int, default=150)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    cdir = variant_dir(args.variant)
    tdir = topic_dir(args.variant)
    centroids = np.load(cdir / "centroids_qwen.npy").astype(np.float32, copy=False)
    centroids = l2_normalize(centroids)
    sizes_json = load_json(cdir / "sizes.json")
    sizes = np.array([int(sizes_json.get(str(i), 0)) for i in range(centroids.shape[0])], dtype=np.int64)
    fine_terms = load_json(tdir / "top_terms.json")
    fine_labels = load_json(tdir / "llm_label.json")

    if args.level1 >= centroids.shape[0]:
        raise ValueError("--level1 must be smaller than the number of fine clusters")
    if args.level0 >= args.level1:
        raise ValueError("--level0 must be smaller than --level1")

    raw_l1 = fit_agglomerative(centroids, args.level1)
    l1_raw_values = sorted(int(x) for x in np.unique(raw_l1))
    l1_raw_to_id = {raw: f"L1_{pos:03d}" for pos, raw in enumerate(l1_raw_values)}
    l1_groups = {
        l1_raw_to_id[raw]: [int(i) for i in np.flatnonzero(raw_l1 == raw)]
        for raw in l1_raw_values
    }
    l1_ids = list(l1_groups)
    l1_centroids = np.vstack([weighted_centroid(centroids, l1_groups[node_id], sizes) for node_id in l1_ids])

    raw_l0 = fit_agglomerative(l1_centroids, args.level0)
    l0_raw_values = sorted(int(x) for x in np.unique(raw_l0))
    l0_raw_to_id = {raw: f"L0_{pos:02d}" for pos, raw in enumerate(l0_raw_values)}
    l0_groups = {
        l0_raw_to_id[raw]: [l1_ids[int(i)] for i in np.flatnonzero(raw_l0 == raw)]
        for raw in l0_raw_values
    }

    nodes: dict[str, dict[str, Any]] = {}
    node_terms: dict[str, list[dict[str, Any]]] = {}
    levels = {"0": [], "1": [], "2": []}

    for l0_id, child_l1_ids in l0_groups.items():
        fine_ids = [cid for l1_id in child_l1_ids for cid in l1_groups[l1_id]]
        terms = aggregate_terms(fine_ids, fine_terms, sizes, args.top_n)
        nodes[l0_id] = {
            "id": l0_id,
            "level": 0,
            "parent_id": None,
            "children": child_l1_ids,
            "fine_cluster_ids": fine_ids,
            "paper_count": int(sizes[fine_ids].sum()),
            "label": terms[0]["term"] if terms else l0_id,
        }
        node_terms[l0_id] = terms
        levels["0"].append(l0_id)

    l1_parent: dict[str, str] = {}
    for l0_id, child_l1_ids in l0_groups.items():
        for l1_id in child_l1_ids:
            l1_parent[l1_id] = l0_id

    for l1_id in l1_ids:
        fine_ids = l1_groups[l1_id]
        terms = aggregate_terms(fine_ids, fine_terms, sizes, args.top_n)
        child_l2 = [f"L2_{cid:03d}" for cid in fine_ids]
        nodes[l1_id] = {
            "id": l1_id,
            "level": 1,
            "parent_id": l1_parent[l1_id],
            "children": child_l2,
            "fine_cluster_ids": fine_ids,
            "paper_count": int(sizes[fine_ids].sum()),
            "label": terms[0]["term"] if terms else l1_id,
        }
        node_terms[l1_id] = terms
        levels["1"].append(l1_id)

    for cid in range(centroids.shape[0]):
        node_id = f"L2_{cid:03d}"
        parent = l1_raw_to_id[int(raw_l1[cid])]
        terms = fine_terms.get(str(cid), [])[: args.top_n]
        nodes[node_id] = {
            "id": node_id,
            "level": 2,
            "parent_id": parent,
            "children": [],
            "fine_cluster_ids": [cid],
            "cluster_id": cid,
            "paper_count": int(sizes[cid]),
            "label": label_for_fine(cid, fine_labels, fine_terms),
        }
        node_terms[node_id] = terms
        levels["2"].append(node_id)

    HIERARCHY_DIR.mkdir(parents=True, exist_ok=True)
    save_json(HIERARCHY_DIR / "params.json", {
        "variant": args.variant,
        "level0": args.level0,
        "level1": args.level1,
        "level2": int(centroids.shape[0]),
        "algorithm": "AgglomerativeClustering",
        "metric": "cosine",
        "linkage": "average",
    })
    save_json(HIERARCHY_DIR / "tree.json", {"levels": levels, "nodes": nodes})
    save_json(HIERARCHY_DIR / "node_top_terms.json", node_terms)
    print(f"[done] hierarchy: L0={len(levels['0'])} L1={len(levels['1'])} L2={len(levels['2'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
