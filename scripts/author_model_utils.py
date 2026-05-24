from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset

from model import AuthorTransformer, AuthorTransformerConfig
from pipeline_common import (
    AUTHORS,
    EMBEDDINGS_PATH,
    MODELS,
    RANDOM_STATE,
    l2_normalize,
    load_json,
    load_papers_in_embedding_order,
)


CONFIGS = {
    "embedding_only": {"lambda_cluster": 0.0, "lambda_emb": 1.0},
    "hybrid_02_08": {"lambda_cluster": 0.2, "lambda_emb": 0.8},
    "hybrid_05_05": {"lambda_cluster": 0.5, "lambda_emb": 0.5},
    "hybrid_08_02": {"lambda_cluster": 0.8, "lambda_emb": 0.2},
}

KS = (10, 50, 100)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_examples(split: str) -> list[dict[str, Any]]:
    return read_jsonl(AUTHORS / f"{split}_examples.jsonl")


def load_author_arrays() -> tuple[np.ndarray, np.ndarray]:
    emb = np.load(EMBEDDINGS_PATH).astype(np.float32, copy=False)
    emb = l2_normalize(emb)
    q = np.load(AUTHORS / "q_with_noise.npy").astype(np.float32, copy=False)
    return emb, q


def load_paper_years() -> np.ndarray:
    papers = load_papers_in_embedding_order()
    return np.array([int(p["year"]) for p in papers], dtype=np.int16)


class AuthorExampleDataset(Dataset):
    def __init__(
        self,
        examples: list[dict[str, Any]],
        embeddings: np.ndarray,
        q: np.ndarray,
        years: np.ndarray,
        max_history: int = 20,
    ) -> None:
        self.examples = examples
        self.embeddings = embeddings
        self.q = q
        self.years = years
        self.max_history = max_history

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        hist = list(ex["history_paper_idxs"])[-self.max_history :]
        future = list(ex["future_paper_idxs"])
        pad = self.max_history - len(hist)
        hist_padded = [-1] * pad + hist
        mask = [False] * pad + [True] * len(hist)
        emb = np.zeros((self.max_history, self.embeddings.shape[1]), dtype=np.float32)
        q = np.zeros((self.max_history, self.q.shape[1]), dtype=np.float32)
        yrs = np.zeros((self.max_history,), dtype=np.int64)
        real = np.array(hist, dtype=np.int64)
        if len(real):
            emb[pad:] = self.embeddings[real]
            q[pad:] = self.q[real]
            yrs[pad:] = self.years[real]
        future_arr = np.array(future, dtype=np.int64)
        target_cluster = self.q[future_arr].mean(axis=0).astype(np.float32)
        return {
            "history_emb": emb,
            "history_q": q,
            "years": yrs,
            "mask": np.array(mask, dtype=bool),
            "future_idxs": future_arr,
            "target_cluster": target_cluster,
            "author_id": ex["author_id"],
            "cutoff_year": int(ex["cutoff_year"]),
            "history_paper_idxs": np.array(hist_padded, dtype=np.int64),
        }


def collate_examples(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_future = max(len(x["future_idxs"]) for x in batch)
    future = np.full((len(batch), max_future), -1, dtype=np.int64)
    future_mask = np.zeros((len(batch), max_future), dtype=bool)
    for i, item in enumerate(batch):
        n = len(item["future_idxs"])
        future[i, :n] = item["future_idxs"]
        future_mask[i, :n] = True
    out = {
        "history_emb": torch.tensor(np.stack([x["history_emb"] for x in batch]), dtype=torch.float32),
        "history_q": torch.tensor(np.stack([x["history_q"] for x in batch]), dtype=torch.float32),
        "years": torch.tensor(np.stack([x["years"] for x in batch]), dtype=torch.long),
        "mask": torch.tensor(np.stack([x["mask"] for x in batch]), dtype=torch.bool),
        "future_idxs": torch.tensor(future, dtype=torch.long),
        "future_mask": torch.tensor(future_mask, dtype=torch.bool),
        "target_cluster": torch.tensor(np.stack([x["target_cluster"] for x in batch]), dtype=torch.float32),
        "author_id": [x["author_id"] for x in batch],
        "cutoff_year": [x["cutoff_year"] for x in batch],
    }
    return out


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def embedding_loss(pred_emb: torch.Tensor, future_idxs: torch.Tensor, future_mask: torch.Tensor, embeddings: torch.Tensor, tau: float) -> torch.Tensor:
    safe_idxs = future_idxs.clamp_min(0)
    future_emb = embeddings[safe_idxs]
    sims = (future_emb * pred_emb[:, None, :]).sum(dim=-1) / tau
    sims = sims.masked_fill(~future_mask, -1e9)
    return (-tau * torch.logsumexp(sims, dim=1)).mean()


def cluster_loss(pred_cluster: torch.Tensor, target_cluster: torch.Tensor) -> torch.Tensor:
    return -(target_cluster * torch.log(pred_cluster.clamp_min(1e-12))).sum(dim=1).mean()


def total_losses(
    model: AuthorTransformer,
    batch: dict[str, Any],
    embeddings_t: torch.Tensor,
    lambda_cluster: float,
    lambda_emb: float,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out = model(batch["history_emb"], batch["history_q"], batch["years"], batch["mask"])
    lc = cluster_loss(out["pred_cluster"], batch["target_cluster"])
    le = embedding_loss(out["pred_emb"], batch["future_idxs"], batch["future_mask"], embeddings_t, tau)
    loss = lambda_cluster * lc + lambda_emb * le
    return loss, lc, le, out["pred_emb"]


def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def retrieval_metrics_from_scores(scores: np.ndarray, relevant_positions: list[set[int]], ks: tuple[int, ...] = KS) -> dict[str, float]:
    metrics: dict[str, float] = {"n_examples": float(len(relevant_positions))}
    order = np.argsort(-scores, axis=1)
    for k in ks:
        hits = []
        mrrs = []
        ndcgs = []
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        for row, rel in zip(order[:, :k], relevant_positions, strict=False):
            rel_flags = np.array([idx in rel for idx in row], dtype=np.float32)
            hits.append(float(rel_flags.any()))
            if rel_flags.any():
                first = int(np.flatnonzero(rel_flags)[0]) + 1
                mrrs.append(1.0 / first)
            else:
                mrrs.append(0.0)
            dcg = float((rel_flags * discounts).sum())
            ideal_n = min(len(rel), k)
            idcg = float(discounts[:ideal_n].sum()) if ideal_n else 0.0
            ndcgs.append(dcg / idcg if idcg else 0.0)
        metrics[f"hit@{k}"] = float(np.mean(hits)) if hits else 0.0
        metrics[f"mrr@{k}"] = float(np.mean(mrrs)) if mrrs else 0.0
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    return metrics


@torch.inference_mode()
def predict_embeddings(
    model: AuthorTransformer,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    model.eval()
    preds = []
    meta = []
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["history_emb"], batch["history_q"], batch["years"], batch["mask"])
        preds.append(out["pred_emb"].detach().cpu().numpy())
        meta.extend(
            {"author_id": aid, "cutoff_year": cutoff}
            for aid, cutoff in zip(batch["author_id"], batch["cutoff_year"], strict=False)
        )
    return np.vstack(preds).astype(np.float32), meta


def evaluate_retrieval(
    model: AuthorTransformer,
    loader,
    examples: list[dict[str, Any]],
    embeddings: np.ndarray,
    candidate_idxs: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    pred, _ = predict_embeddings(model, loader, device)
    candidate_emb = embeddings[candidate_idxs]
    scores = pred @ candidate_emb.T
    pos = {int(idx): i for i, idx in enumerate(candidate_idxs)}
    relevant_positions = []
    for ex in examples:
        rel = {pos[int(idx)] for idx in ex["future_paper_idxs"] if int(idx) in pos}
        relevant_positions.append(rel)
    return retrieval_metrics_from_scores(scores, relevant_positions)


def save_training_plots(out_dir: Path, log: list[dict[str, Any]]) -> None:
    if not log:
        return
    epochs = [row["epoch"] for row in log]
    plots = [
        ("train_loss.png", "train_loss", "Train loss"),
        ("val_loss.png", "val_loss", "Validation loss"),
        ("val_ndcg10.png", "val_ndcg@10", "Validation nDCG@10"),
    ]
    for filename, key, title in plots:
        vals = [row.get(key) for row in log]
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, vals, marker="o")
        plt.xlabel("epoch")
        plt.ylabel(key)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=160)
        plt.close()


def model_dir(name: str) -> Path:
    return MODELS / name


def load_model_checkpoint(path: Path, device: torch.device) -> tuple[AuthorTransformer, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    cfg = AuthorTransformerConfig(**payload["model_config"])
    model = AuthorTransformer(cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def save_checkpoint(path: Path, model: AuthorTransformer, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_config": model.config.to_dict(),
        **extra,
    }
    torch.save(payload, path)


def seed_everything(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def warmup_cosine_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return max(1e-8, (step + 1) / max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
