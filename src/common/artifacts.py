from __future__ import annotations

import numpy as np

from .io import save_json
from .math import l2_normalize
from .paths import AUTHORS, CLUSTERING, DATA, MODELS, PREDICTIONS, REDUCED, RESULTS, TOPICS, variant_dir


def ensure_dirs() -> None:
    for path in (DATA, RESULTS, REDUCED, CLUSTERING, TOPICS, AUTHORS, PREDICTIONS, MODELS):
        path.mkdir(parents=True, exist_ok=True)


def save_variant_artifacts(name: str, params: dict, labels: np.ndarray, proba: np.ndarray, centroids_qwen: np.ndarray) -> None:
    out_dir = variant_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = labels.astype(np.int32, copy=False)
    proba = proba.astype(np.float32, copy=False)
    centroids_qwen = l2_normalize(centroids_qwen.astype(np.float32, copy=False))
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "proba.npy", proba)
    np.save(out_dir / "centroids_qwen.npy", centroids_qwen)
    non_noise = labels[labels >= 0]
    sizes = {str(i): int((non_noise == i).sum()) for i in range(int(proba.shape[1]))}
    save_json(out_dir / "params.json", params)
    save_json(out_dir / "sizes.json", sizes)
