from __future__ import annotations

import argparse
import sys

import hdbscan
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from clustering.paper_topics.cluster_fcm import fit_spherical_fcm
from common.compat import (
    DATA,
    META_ALL_VARIANTS,
    META_FCM_VARIANTS,
    META_GMM_VARIANTS,
    META_HDBSCAN_VARIANTS,
    RANDOM_STATE,
    hard_labels_from_proba,
    hdbscan_labels_from_proba,
    l2_normalize,
    save_variant_artifacts,
    variant_dir,
    weighted_centroids,
)


META_DIR = DATA / "meta"


def run_hdbscan(features: np.ndarray, embeddings: np.ndarray, variant, force: bool) -> None:
    out_dir = variant_dir(variant.name)
    if (out_dir / "labels.npy").exists() and (out_dir / "proba.npy").exists() and not force:
        print(f"[skip] {variant.name} exists; use --force")
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
            "feature_space": "meta_umap5",
            **params,
            "n_clusters": int(membership.shape[1]),
            "noise_ratio": float((labels == -1).mean()),
        },
        labels,
        membership,
        centroids,
    )
    print(f"[done] {variant.name}: K={membership.shape[1]}, noise={(labels == -1).mean():.3f}")


def run_fcm(embeddings: np.ndarray, variant, force: bool, m: float, max_iter: int, tol: float) -> None:
    out_dir = variant_dir(variant.name)
    if (out_dir / "labels.npy").exists() and (out_dir / "proba.npy").exists() and not force:
        print(f"[skip] {variant.name} exists; use --force")
        return
    proba, centroids, n_iter, final_shift = fit_spherical_fcm(embeddings, int(variant.k), m, max_iter, tol)
    labels = hard_labels_from_proba(proba)
    save_variant_artifacts(
        variant.name,
        {
            "method": "spherical_fcm",
            "feature_space": "meta_qwen_l2",
            "k": int(variant.k),
            "m": m,
            "max_iter": max_iter,
            "tol": tol,
            "n_iter": n_iter,
            "final_centroid_shift": final_shift,
            "random_state": RANDOM_STATE,
        },
        labels,
        proba,
        centroids,
    )
    print(f"[done] {variant.name}: K={variant.k}, iter={n_iter}, shift={final_shift:.6f}")


def run_gmm(features: np.ndarray, embeddings: np.ndarray, variant, force: bool) -> None:
    out_dir = variant_dir(variant.name)
    if (out_dir / "labels.npy").exists() and (out_dir / "proba.npy").exists() and not force:
        print(f"[skip] {variant.name} exists; use --force")
        return
    k = int(variant.k)
    model = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        n_init=3,
        reg_covar=1e-6,
        random_state=RANDOM_STATE,
        verbose=0,
    )
    labels = model.fit_predict(features).astype(np.int32)
    proba = model.predict_proba(features).astype(np.float32)
    labels = hard_labels_from_proba(proba)
    centroids = weighted_centroids(embeddings, proba, labels)
    save_variant_artifacts(
        variant.name,
        {
            "method": "gmm",
            "feature_space": "meta_pca",
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
    print(f"[done] {variant.name}: K={k}, iter={model.n_iter_}")


def run_kmeans(features: np.ndarray, embeddings: np.ndarray, variant, force: bool) -> None:
    out_dir = variant_dir(variant.name)
    if (out_dir / "labels.npy").exists() and (out_dir / "proba.npy").exists() and not force:
        print(f"[skip] {variant.name} exists; use --force")
        return
    k = int(variant.k)
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(features).astype(np.int32)
    proba = np.zeros((len(labels), k), dtype=np.float32)
    proba[np.arange(len(labels)), labels] = 1.0
    centroids = weighted_centroids(embeddings, proba, labels)
    save_variant_artifacts(
        variant.name,
        {
            "method": "kmeans",
            "feature_space": "meta_umap5",
            "k": k,
            "random_state": RANDOM_STATE,
            "inertia": float(model.inertia_),
        },
        labels,
        proba,
        centroids,
    )
    print(f"[done] {variant.name}: K={k}, inertia={model.inertia_:.3f}")


def selected_variants(names: list[str]) -> list:
    by_name = {variant.name: variant for variant in META_ALL_VARIANTS}
    if names == ["all"]:
        return META_ALL_VARIANTS
    out = []
    for name in names:
        if name in by_name:
            out.append(by_name[name])
        elif name.startswith("meta_kmeans_umap10_k"):
            k = int(name.removeprefix("meta_kmeans_umap10_k"))
            from common.clustering_variants import Variant

            out.append(Variant(name, "kmeans", k=k))
        else:
            raise KeyError(f"unknown meta variant: {name}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=["all"])
    parser.add_argument("--m", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    embeddings = l2_normalize(np.load(META_DIR / "cluster_embeddings.npy").astype(np.float32, copy=False))
    umap_features = np.load(META_DIR / "umap5.npy").astype(np.float32, copy=False)
    pca_features = np.load(META_DIR / "pca.npy").astype(np.float32, copy=False)
    for variant in tqdm(selected_variants(args.variants), desc="meta clustering"):
        if variant in META_HDBSCAN_VARIANTS:
            run_hdbscan(umap_features, embeddings, variant, args.force)
        elif variant.method == "kmeans":
            run_kmeans(umap_features, embeddings, variant, args.force)
        elif variant in META_FCM_VARIANTS:
            run_fcm(embeddings, variant, args.force, args.m, args.max_iter, args.tol)
        elif variant in META_GMM_VARIANTS:
            run_gmm(pca_features, embeddings, variant, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
