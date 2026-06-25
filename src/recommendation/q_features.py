from __future__ import annotations

import numpy as np

from common.compat import AUTHORS, DATA, load_json


VALID_Q_MODES = {"none", "fine", "metacluster", "fine_metacluster"}


def _meta_matrix(n_fine: int, n_meta: int) -> np.ndarray:
    assignments = load_json(DATA / "meta" / "meta_assignments.json")
    fine_to_meta = assignments["fine_to_meta"]
    mat = np.zeros((n_fine, n_meta), dtype=np.float32)
    for fine_id_raw, meta_id_raw in fine_to_meta.items():
        fine_id = int(fine_id_raw)
        meta_id = int(meta_id_raw)
        if 0 <= fine_id < n_fine and 0 <= meta_id < n_meta:
            mat[fine_id, meta_id] = 1.0
    return mat


def ensure_metacluster_q(force: bool = False, chunk_size: int = 8192) -> np.ndarray:
    out_path = AUTHORS / "q_metacluster_with_noise.npy"
    if out_path.exists() and not force:
        return np.load(out_path, mmap_mode="r")

    fine = np.load(AUTHORS / "q_with_noise.npy", mmap_mode="r")
    n_docs = fine.shape[0]
    n_fine = fine.shape[1] - 1
    n_meta = 76
    mat = _meta_matrix(n_fine, n_meta)
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n_docs, n_meta + 1))
    for start in range(0, n_docs, chunk_size):
        end = min(start + chunk_size, n_docs)
        fine_chunk = np.asarray(fine[start:end, :n_fine], dtype=np.float32)
        out[start:end, :n_meta] = fine_chunk @ mat
        out[start:end, n_meta] = np.asarray(fine[start:end, n_fine], dtype=np.float32)
    out.flush()
    return np.load(out_path, mmap_mode="r")


def ensure_fine_metacluster_q(force: bool = False, chunk_size: int = 8192) -> np.ndarray:
    out_path = AUTHORS / "q_fine_metacluster_with_noise.npy"
    if out_path.exists() and not force:
        return np.load(out_path, mmap_mode="r")

    fine = np.load(AUTHORS / "q_with_noise.npy", mmap_mode="r")
    meta = ensure_metacluster_q(force=force, chunk_size=chunk_size)
    n_docs = fine.shape[0]
    out_dim = fine.shape[1] + meta.shape[1]
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(n_docs, out_dim))
    for start in range(0, n_docs, chunk_size):
        end = min(start + chunk_size, n_docs)
        out[start:end, : fine.shape[1]] = fine[start:end]
        out[start:end, fine.shape[1] :] = meta[start:end]
    out.flush()
    return np.load(out_path, mmap_mode="r")


def load_q_features(q_mode: str, force: bool = False) -> np.ndarray:
    if q_mode not in VALID_Q_MODES:
        raise ValueError(f"unknown q_mode={q_mode!r}; expected one of {sorted(VALID_Q_MODES)}")
    if q_mode == "none":
        n_docs = np.load(AUTHORS / "q_with_noise.npy", mmap_mode="r").shape[0]
        return np.zeros((n_docs, 0), dtype=np.float32)
    if q_mode == "fine":
        return np.load(AUTHORS / "q_with_noise.npy", mmap_mode="r")
    if q_mode == "metacluster":
        return ensure_metacluster_q(force=force)
    return ensure_fine_metacluster_q(force=force)
