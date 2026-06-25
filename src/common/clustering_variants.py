from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RANDOM_STATE = 42


@dataclass(frozen=True)
class Variant:
    name: str
    method: str
    k: int | None = None
    params: dict[str, Any] | None = None


# Params scaled for the ~509k AI corpus so cluster counts stay moderate (K in the
# hundreds-to-~1000 range): keeps the dense membership matrix (N x K) and the
# transformer q-features feasible. (On the old 43k NLP corpus fine used mcs=10.)
HDBSCAN_VARIANTS = [
    Variant("hdbscan_fine", "hdbscan", params={"min_cluster_size": 50, "min_samples": 10, "cluster_selection_method": "leaf"}),
    Variant("hdbscan_medium", "hdbscan", params={"min_cluster_size": 200, "min_samples": 25, "cluster_selection_method": "eom"}),
    Variant("hdbscan_coarse", "hdbscan", params={"min_cluster_size": 500, "min_samples": 25, "cluster_selection_method": "eom"}),
]
FCM_VARIANTS = [Variant(f"fcm_{k}", "fcm", k=k) for k in (50, 100, 200)]
GMM_VARIANTS = [Variant(f"gmm_{k}", "gmm", k=k) for k in (50, 100, 200)]
ALL_VARIANTS = HDBSCAN_VARIANTS + FCM_VARIANTS + GMM_VARIANTS

# Baselines are swept over a grid of cluster counts (up to 1000) on UMAP-10, so the
# clustering quality metric is reported as a curve over K for random and k-means,
# rather than at a single arbitrary K.
BASELINE_K_GRID = [50, 100, 200, 400, 600, 800, 1000]
BASELINE_K = 801  # legacy single-K reference (kept for backward compatibility)
BASELINE_VARIANTS = [
    Variant(f"random_umap10_k{k}", "random", k=k) for k in BASELINE_K_GRID
] + [
    Variant(f"kmeans_umap10_k{k}", "kmeans", k=k) for k in BASELINE_K_GRID
]
ALL_CLUSTERING_EVAL_VARIANTS = ALL_VARIANTS + BASELINE_VARIANTS
BASELINE_VARIANT_NAMES = {v.name for v in BASELINE_VARIANTS}

META_HDBSCAN_VARIANTS = [
    Variant("meta_hdbscan_fine", "hdbscan", params={"min_cluster_size": 3, "min_samples": 2, "cluster_selection_method": "leaf"}),
    Variant("meta_hdbscan_medium", "hdbscan", params={"min_cluster_size": 5, "min_samples": 2, "cluster_selection_method": "eom"}),
    Variant("meta_hdbscan_coarse", "hdbscan", params={"min_cluster_size": 10, "min_samples": 3, "cluster_selection_method": "eom"}),
]
META_FCM_VARIANTS = [Variant(f"meta_fcm_{k}", "fcm", k=k) for k in (10, 20, 40)]
META_GMM_VARIANTS = [Variant(f"meta_gmm_{k}", "gmm", k=k) for k in (10, 20, 40)]
META_ALL_VARIANTS = META_HDBSCAN_VARIANTS + META_FCM_VARIANTS + META_GMM_VARIANTS
META_BASELINE_VARIANT_NAMES = {name for name in ()}  # filled dynamically; see is_meta_baseline()


def is_meta_baseline(name: str) -> bool:
    return name.startswith("meta_kmeans_umap10_k")
