from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterator

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset

from recommendation.model import AuthorTransformer, AuthorTransformerConfig
from recommendation.q_features import load_q_features
from common.author_splits import split_author_set
from common.compat import (
    AUTHORS,
    EMBEDDINGS_PATH,
    MODELS,
    RANDOM_STATE,
    l2_normalize,
    load_json,
    load_papers_in_embedding_order,
)


MODEL_NAME = "author_retriever"
LEGACY_PAPER_MODEL_NAME = "embedding_only"
INFONCE_MODEL_NAME = "embedding_infonce"
COAUTHOR_INFONCE_MODEL_NAME = "author_retriever"
MEAN_BASELINE_NAME = "mean_author_embedding"

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


def load_author_arrays(q_mode: str = "fine") -> tuple[np.ndarray, np.ndarray]:
    # Normalize in place to avoid a transient duplicate of the (large) embedding matrix.
    emb = np.load(EMBEDDINGS_PATH).astype(np.float32, copy=False)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    np.maximum(norm, 1e-12, out=norm)
    emb /= norm
    q = load_q_features(q_mode)
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
        author_pool: set[str] | None = None,
    ) -> None:
        self.examples = examples
        self.embeddings = embeddings
        self.q = q
        self.years = years
        self.max_history = max_history
        self.author_pool = author_pool
        self.future_coauthor_ids = self._resolve_future_coauthors(examples)

    def _resolve_future_coauthors(self, examples: list[dict[str, Any]]) -> list[list[str]]:
        from evaluation.coauthor_retrieval import relevant_future_coauthors

        return [
            sorted(relevant_future_coauthors(ex, author_pool=self.author_pool))
            for ex in examples
        ]

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
        if len(future_arr):
            target_cluster = self.q[future_arr].mean(axis=0).astype(np.float32)
        else:
            target_cluster = np.zeros((self.q.shape[1],), dtype=np.float32)
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
            "future_coauthor_ids": self.future_coauthor_ids[idx],
        }


def examples_with_positives(examples: list[dict[str, Any]], author_pool: set[str] | None = None) -> list[dict[str, Any]]:
    """Keep only examples that have at least one future coauthor in the eval pool."""
    from evaluation.coauthor_retrieval import relevant_future_coauthors

    kept: list[dict[str, Any]] = []
    for ex in examples:
        if relevant_future_coauthors(ex, author_pool=author_pool):
            kept.append(ex)
    return kept


class MultiPositiveBatchSampler(BatchSampler):
    """Batch sampler for multi-positive contrastive training.

    - Drops anchors without future coauthors (they contribute no gradient).
    - Each batch is built around anchors with positives; optionally adds focal rows
      for positive coauthors when they exist in the training set, so in-batch
      negatives include authors that are positives for other anchors.
    """

    def __init__(
        self,
        dataset: AuthorExampleDataset,
        batch_size: int,
        extra_positive_rows: int = 16,
        seed: int = RANDOM_STATE,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.extra_positive_rows = extra_positive_rows
        self.seed = seed
        self.epoch = 0
        self.pos_indices = [
            i for i in range(len(dataset)) if len(dataset.future_coauthor_ids[i]) > 0
        ]
        self.author_to_idx = {dataset.examples[i]["author_id"]: i for i in range(len(dataset))}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        indices = list(self.pos_indices)
        rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            core = indices[start : start + self.batch_size]
            if len(core) < max(4, self.batch_size // 4):
                break
            core_set = set(core)
            extra: list[int] = []
            for idx in core:
                for aid in self.dataset.future_coauthor_ids[idx]:
                    j = self.author_to_idx.get(aid)
                    if j is not None and j not in core_set and j not in extra:
                        extra.append(j)
                        if len(extra) >= self.extra_positive_rows:
                            break
                if len(extra) >= self.extra_positive_rows:
                    break
            yield core + extra

    def __len__(self) -> int:
        return max(1, len(self.pos_indices) // self.batch_size)


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
        "future_coauthor_ids": [x["future_coauthor_ids"] for x in batch],
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


def infonce_embedding_loss(
    pred_emb: torch.Tensor,
    future_idxs: torch.Tensor,
    future_mask: torch.Tensor,
    embeddings: torch.Tensor,
    tau: float,
    n_negatives: int = 256,
) -> torch.Tensor:
    """Multi-positive InfoNCE: future papers as positives, random corpus papers as negatives."""
    safe_idxs = future_idxs.clamp_min(0)
    future_emb = embeddings[safe_idxs]
    pos_sims = (future_emb * pred_emb[:, None, :]).sum(dim=-1) / tau
    pos_sims = pos_sims.masked_fill(~future_mask, -1e9)
    log_pos = torch.logsumexp(pos_sims, dim=1)

    n_papers = embeddings.shape[0]
    neg_idxs = torch.randint(0, n_papers, (pred_emb.shape[0], n_negatives), device=pred_emb.device)
    neg_emb = embeddings[neg_idxs]
    neg_sims = (neg_emb * pred_emb[:, None, :]).sum(dim=-1) / tau
    log_neg = torch.logsumexp(neg_sims, dim=1)

    return (-log_pos + torch.logaddexp(log_pos, log_neg)).mean()


def compute_embedding_loss(
    loss_type: str,
    model: AuthorTransformer,
    focal_emb: torch.Tensor,
    batch: dict[str, Any],
    embeddings_np: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    max_history: int,
    device: torch.device,
    embeddings_t: torch.Tensor,
    tau: float,
    n_negatives: int = 256,
    author_pool: set[str] | None = None,
) -> torch.Tensor:
    if loss_type in ("coauthor_infonce", "infonce"):
        from recommendation.coauthor_loss import coauthor_infonce_loss

        return coauthor_infonce_loss(
            model,
            focal_emb,
            batch,
            embeddings_np,
            q,
            years,
            max_history,
            device,
            tau,
            n_negatives,
            author_pool=author_pool,
        )
    # legacy paper-level losses (deprecated)
    if loss_type == "paper_infonce":
        return infonce_embedding_loss(
            focal_emb, batch["future_idxs"], batch["future_mask"], embeddings_t, tau, n_negatives
        )
    return embedding_loss(focal_emb, batch["future_idxs"], batch["future_mask"], embeddings_t, tau)


def total_losses(
    model: AuthorTransformer,
    batch: dict[str, Any],
    embeddings_t: torch.Tensor,
    embeddings_np: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    max_history: int,
    device: torch.device,
    tau: float,
    loss_type: str = "coauthor_infonce",
    n_negatives: int = 256,
    author_pool: set[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = model(batch["history_emb"], batch["history_q"], batch["years"], batch["mask"])
    le = compute_embedding_loss(
        loss_type,
        model,
        out["pred_emb"],
        batch,
        embeddings_np,
        q,
        years,
        max_history,
        device,
        embeddings_t,
        tau,
        n_negatives,
        author_pool=author_pool,
    )
    return le, out["pred_emb"]


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
        recalls = []
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        for row, rel in zip(order[:, :k], relevant_positions, strict=False):
            rel_flags = np.array([idx in rel for idx in row], dtype=np.float32)
            hits.append(float(rel_flags.any()))
            recalls.append(float(rel_flags.sum() / len(rel)) if rel else 0.0)
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
        metrics[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
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


@torch.inference_mode()
def evaluate_retrieval(
    model: AuthorTransformer,
    loader,
    examples: list[dict[str, Any]],
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    cutoff_year: int,
    device: torch.device,
    split: str,
    max_history: int = 20,
) -> dict[str, float]:
    from evaluation.coauthor_retrieval import evaluate_coauthor_retrieval

    return evaluate_coauthor_retrieval(
        model, loader, examples, embeddings, q, years, cutoff_year, device, split, max_history
    )


def mean_history_embeddings(loader) -> np.ndarray:
    preds = []
    for batch in loader:
        hist = batch["history_emb"].numpy()
        mask = batch["mask"].numpy()
        out = np.zeros((hist.shape[0], hist.shape[2]), dtype=np.float32)
        for i in range(hist.shape[0]):
            valid = hist[i, mask[i]]
            if len(valid):
                out[i] = valid.mean(axis=0)
        preds.append(l2_normalize(out.astype(np.float32)))
    return np.vstack(preds).astype(np.float32)


def evaluate_retrieval_from_predictions(
    pred: np.ndarray,
    examples: list[dict[str, Any]],
    embeddings: np.ndarray,
    candidate_idxs: np.ndarray,
) -> dict[str, float]:
    pred = l2_normalize(pred.astype(np.float32, copy=False))
    candidate_emb = embeddings[candidate_idxs]
    scores = pred @ candidate_emb.T
    pos = {int(idx): i for i, idx in enumerate(candidate_idxs)}
    relevant_positions = []
    for ex in examples:
        rel = {pos[int(idx)] for idx in ex["future_paper_idxs"] if int(idx) in pos}
        relevant_positions.append(rel)
    return retrieval_metrics_from_scores(scores, relevant_positions)


def evaluate_mean_history_retrieval(
    loader,
    examples: list[dict[str, Any]],
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    cutoff_year: int,
    split: str,
    max_history: int = 20,
) -> dict[str, float]:
    from evaluation.coauthor_retrieval import evaluate_mean_coauthor_retrieval

    return evaluate_mean_coauthor_retrieval(
        loader, examples, embeddings, q, years, cutoff_year, split, max_history
    )


def evaluate_random_retrieval(
    n_examples: int,
    examples: list[dict[str, Any]],
    cutoff_year: int,
    split: str,
    max_history: int = 20,
    seed: int = RANDOM_STATE,
) -> dict[str, float]:
    from evaluation.coauthor_retrieval import evaluate_random_coauthor_retrieval

    return evaluate_random_coauthor_retrieval(
        n_examples, examples, cutoff_year, split, max_history, seed
    )


@torch.inference_mode()
def evaluate_model(
    model_name: str,
    split: str,
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, float]:
    examples = split_examples(split)
    author_pool = split_author_set(split)
    ds = AuthorExampleDataset(examples, embeddings, q, years, author_pool=author_pool)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_examples, num_workers=0)
    from evaluation.coauthor_retrieval import cutoff_year_for_split

    cutoff_year = cutoff_year_for_split(split)
    model, _ = load_model_checkpoint(model_dir(model_name) / "best.pt", device)
    return evaluate_retrieval(
        model, loader, examples, embeddings, q, years, cutoff_year, device, split
    )


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
    model.load_state_dict(payload["state_dict"], strict=False)
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
