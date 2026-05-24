from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
REDUCED = DATA / "reduced"
CLUSTERING = DATA / "clustering"
TOPICS = DATA / "topics"
AUTHORS = DATA / "authors"
PREDICTIONS = DATA / "predictions"
MODELS = ROOT / "models"

EMBEDDINGS_PATH = DATA / "embeddings.npy"
PAPER_IDS_PATH = DATA / "paper_ids.json"
PAPERS_PATH = DATA / "openalex_clean.jsonl"
LLM_CACHE_PATH = DATA / "llm_cache.sqlite"

RANDOM_STATE = 42
LLM_MODEL = "gpt-4o-mini"
LLM_VERSION = "topic_eval_v1"


@dataclass(frozen=True)
class Variant:
    name: str
    method: str
    k: int | None = None
    params: dict[str, Any] | None = None


HDBSCAN_VARIANTS = [
    Variant(
        "hdbscan_fine",
        "hdbscan",
        params={
            "min_cluster_size": 10,
            "min_samples": 5,
            "cluster_selection_method": "leaf",
        },
    ),
    Variant(
        "hdbscan_medium",
        "hdbscan",
        params={
            "min_cluster_size": 30,
            "min_samples": 5,
            "cluster_selection_method": "eom",
        },
    ),
    Variant(
        "hdbscan_coarse",
        "hdbscan",
        params={
            "min_cluster_size": 100,
            "min_samples": 5,
            "cluster_selection_method": "eom",
        },
    ),
]

FCM_VARIANTS = [Variant(f"fcm_{k}", "fcm", k=k) for k in (50, 100, 200)]
GMM_VARIANTS = [Variant(f"gmm_{k}", "gmm", k=k) for k in (50, 100, 200)]
ALL_VARIANTS = HDBSCAN_VARIANTS + FCM_VARIANTS + GMM_VARIANTS


def ensure_dirs() -> None:
    for path in (DATA, RESULTS, REDUCED, CLUSTERING, TOPICS, AUTHORS, PREDICTIONS, MODELS):
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_embeddings() -> np.ndarray:
    arr = np.load(EMBEDDINGS_PATH).astype(np.float32, copy=False)
    return l2_normalize(arr)


def load_paper_ids() -> list[str]:
    return load_json(PAPER_IDS_PATH)


def load_papers_by_id() -> dict[str, dict[str, Any]]:
    return {rec["paper_id"]: rec for rec in iter_jsonl(PAPERS_PATH)}


def load_papers_in_embedding_order() -> list[dict[str, Any]]:
    ids = load_paper_ids()
    by_id = load_papers_by_id()
    return [by_id[paper_id] for paper_id in ids]


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


def variant_dir(name: str) -> Path:
    return CLUSTERING / name


def topic_dir(name: str) -> Path:
    return TOPICS / name


def save_variant_artifacts(
    name: str,
    params: dict[str, Any],
    labels: np.ndarray,
    proba: np.ndarray,
    centroids_qwen: np.ndarray,
) -> None:
    out_dir = variant_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = labels.astype(np.int32, copy=False)
    proba = proba.astype(np.float32, copy=False)
    centroids_qwen = l2_normalize(centroids_qwen.astype(np.float32, copy=False))
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "proba.npy", proba)
    np.save(out_dir / "centroids_qwen.npy", centroids_qwen)
    non_noise = labels[labels >= 0]
    sizes = {
        str(i): int((non_noise == i).sum())
        for i in range(int(proba.shape[1]))
    }
    save_json(out_dir / "params.json", params)
    save_json(out_dir / "sizes.json", sizes)


def weighted_centroids(
    embeddings: np.ndarray,
    proba: np.ndarray,
    hard_labels: np.ndarray | None = None,
) -> np.ndarray:
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


def representative_indices(
    proba: np.ndarray,
    labels: np.ndarray,
    k: int,
    top_n: int = 10,
) -> dict[str, list[int]]:
    reps: dict[str, list[int]] = {}
    for cluster_id in range(k):
        scores = proba[:, cluster_id].copy()
        if labels is not None:
            scores[labels == -1] = -1.0
        order = np.argsort(-scores)
        good = [int(i) for i in order[:top_n] if scores[i] > 0]
        reps[str(cluster_id)] = good
    return reps


def compact_abstract(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


class LLMCache:
    def __init__(self, path: Path = LLM_CACHE_PATH, model: str = LLM_MODEL) -> None:
        load_dotenv(ROOT / ".env")
        self.path = path
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def key(self, system_prompt: str, user_prompt: str) -> str:
        payload = f"{LLM_VERSION}\n{self.model}\n{system_prompt}\n{user_prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 6,
    ) -> dict[str, Any]:
        key = self.key(system_prompt, user_prompt)
        row = self.conn.execute(
            "SELECT response FROM cache WHERE key = ?",
            (key,),
        ).fetchone()
        if row:
            return json.loads(row[0])

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                raw = response.choices[0].message.content or "{}"
                obj = json.loads(raw)
                self.conn.execute(
                    "INSERT INTO cache(key, model, prompt, response, created_at) VALUES (?, ?, ?, ?, ?)",
                    (key, self.model, user_prompt, json.dumps(obj, ensure_ascii=False), time.strftime("%Y-%m-%dT%H:%M:%S")),
                )
                self.conn.commit()
                return obj
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("unreachable")


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
