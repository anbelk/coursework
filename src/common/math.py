from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def load_embeddings(path) -> np.ndarray:
    arr = np.load(path).astype(np.float32, copy=False)
    return l2_normalize(arr)


def weighted_centroids(embeddings: np.ndarray, proba: np.ndarray, hard_labels: np.ndarray | None = None) -> np.ndarray:
    weights = proba.astype(np.float32, copy=False)
    denom = weights.sum(axis=0)[:, None]
    centroids = weights.T @ embeddings
    centroids = centroids / np.maximum(denom, 1e-12)
    if hard_labels is not None:
        for k in range(weights.shape[1]):
            if denom[k, 0] <= 1e-8:
                members = embeddings[hard_labels == k]
                if len(members):
                    centroids[k] = members.mean(axis=0)
    return l2_normalize(centroids.astype(np.float32))


def hard_labels_from_proba(proba: np.ndarray) -> np.ndarray:
    return np.argmax(proba, axis=1).astype(np.int32)


def hdbscan_labels_from_proba(proba: np.ndarray, noise: np.ndarray) -> np.ndarray:
    labels = np.argmax(proba, axis=1).astype(np.int32)
    labels[noise] = -1
    return labels


def cluster_sizes(labels: np.ndarray, k: int) -> np.ndarray:
    return np.array([(labels == i).sum() for i in range(k)], dtype=np.int64)


def representative_indices(proba: np.ndarray, labels: np.ndarray, k: int, top_n: int = 10) -> dict[str, list[int]]:
    reps: dict[str, list[int]] = {}
    for cluster_id in range(k):
        scores = proba[:, cluster_id].copy()
        if labels is not None:
            scores[labels == -1] = -1.0
        order = np.argsort(-scores)
        good = [int(i) for i in order[:top_n] if scores[i] > 0]
        reps[str(cluster_id)] = good
    return reps


def minmax_score(values: list[float], higher_better: bool = True) -> list[float]:
    arr = np.array(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return [0.0 for _ in values]
    fill = np.nanmedian(arr[finite])
    arr = np.where(finite, arr, fill)
    if not higher_better:
        arr = -arr
    lo = arr.min()
    hi = arr.max()
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return ((arr - lo) / (hi - lo)).tolist()


def minmax_score_with_bounds(values: list[float], lo: float, hi: float) -> list[float]:
    arr = np.array(values, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return [0.0 for _ in values]
    fill = np.nanmedian(arr[finite])
    arr = np.where(finite, arr, fill)
    if abs(hi - lo) < 1e-12:
        return [0.5 for _ in values]
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).tolist()
