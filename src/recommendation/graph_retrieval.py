from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.author_splits import history_cutoff, split_author_set
from common.compat import (
    AUTHORS,
    CLUSTERING,
    DATA,
    EMBEDDINGS_PATH,
    MODELS,
    RANDOM_STATE,
    load_json,
    save_json,
)
from evaluation.coauthor_retrieval import (
    cutoff_examples,
    cutoff_year_for_split,
    evaluate_coauthor_from_embeddings,
    past_coauthors,
    relevant_future_coauthors,
)
from recommendation.training_utils import (
    COAUTHOR_INFONCE_MODEL_NAME,
    split_examples,
    save_training_plots,
    seed_everything,
)


GRAPH_MODEL_NAMES = {
    "author": "graphsage_author",
    "author_metacluster": "graphsage_author_metacluster",
}


@dataclass
class GraphRetrievalConfig:
    variant: str = "author"
    emb_dim: int = 768
    hidden_dim: int = 256
    n_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 80
    patience: int = 10
    tau: float = 0.05
    n_negatives: int = 512
    max_positives: int = 8
    loss_batch_size: int = 128
    res_scale_init: float = 0.1
    feature_source: str = "mean"
    feature_model: str = COAUTHOR_INFONCE_MODEL_NAME
    max_history: int = 20
    meta_min_weight: float = 0.02
    seed: int = RANDOM_STATE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphData:
    author_ids: list[str]
    n_authors: int
    n_meta: int
    x: torch.Tensor
    relation_adj: dict[str, torch.Tensor]
    train_anchor_idx: list[int]
    train_positive_idx: list[list[int]]
    train_negative_pool_idx: np.ndarray
    train_excluded_idx: list[set[int]]
    meta: dict[str, Any]


class RelGraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, relation_names: list[str]) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.rel_linears = nn.ModuleDict(
            {name: nn.Linear(in_dim, out_dim, bias=False) for name in relation_names}
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, relation_adj: dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.self_linear(x)
        for name, adj in relation_adj.items():
            out = out + self.rel_linears[name](torch.sparse.mm(adj, x))
        return self.norm(out)


class RelGraphSAGE(nn.Module):
    def __init__(
        self,
        config: GraphRetrievalConfig,
        relation_names: list[str],
    ) -> None:
        super().__init__()
        dims = [config.emb_dim] + [config.hidden_dim] * config.n_layers
        self.layers = nn.ModuleList(
            RelGraphSAGELayer(dims[i], dims[i + 1], relation_names)
            for i in range(config.n_layers)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.out_proj = nn.Linear(config.hidden_dim, config.emb_dim)
        self.res_scale = nn.Parameter(torch.tensor([config.res_scale_init], dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        relation_adj: dict[str, torch.Tensor],
        n_authors: int,
    ) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h, relation_adj)
            h = F.gelu(h)
            h = self.dropout(h)
        author_delta = self.out_proj(h[:n_authors])
        return F.normalize(x[:n_authors] + self.res_scale * author_delta, p=2, dim=-1)


def l2_normalize_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    np.maximum(norm, 1e-12, out=norm)
    return x / norm


def author_model_dir(name: str) -> Path:
    return MODELS / name


def load_author_rows() -> list[dict[str, Any]]:
    rows = load_json(AUTHORS / "author_index.json")
    rows.sort(key=lambda x: x["author_id"])
    return rows


def history_paper_idxs(row: dict[str, Any], cutoff: int, max_history: int) -> list[int]:
    hist = [
        int(paper_idx)
        for paper_idx, year in zip(row["paper_idxs"], row["years"], strict=False)
        if int(year) <= cutoff
    ]
    return hist[-max_history:]


def build_author_features(
    author_rows: list[dict[str, Any]],
    cutoff: int,
    max_history: int,
    feature_source: str,
    feature_model: str,
) -> np.ndarray:
    cache_dir = DATA / "graphs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"author_mean_features_cutoff{cutoff}_max{max_history}.npy"
    if cache_path.exists():
        mean_features = np.load(cache_path).astype(np.float32, copy=False)
    else:
        embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
        mean_features = np.zeros((len(author_rows), int(embeddings.shape[1])), dtype=np.float32)
        for i, row in enumerate(author_rows):
            idxs = history_paper_idxs(row, cutoff, max_history)
            if not idxs:
                continue
            paper_emb = np.asarray(embeddings[idxs], dtype=np.float32)
            paper_emb = l2_normalize_np(paper_emb)
            mean_features[i] = paper_emb.mean(axis=0)
        mean_features = l2_normalize_np(mean_features)
        np.save(cache_path, mean_features)

    if feature_source == "mean":
        return mean_features
    if feature_source != "transformer":
        raise ValueError(f"unknown feature_source: {feature_source}")

    model_path = MODELS / feature_model
    emb_path = model_path / "author_embeddings.npy"
    ids_path = model_path / "author_ids.json"
    if not emb_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            f"missing transformer author features: {emb_path}; run scripts/run_hybrid_retrieval.py first"
        )
    transformer_ids = load_json(ids_path)
    transformer_emb = l2_normalize_np(np.load(emb_path).astype(np.float32, copy=False))
    pos = {aid: i for i, aid in enumerate(transformer_ids)}
    out = mean_features.copy()
    n_loaded = 0
    for i, row in enumerate(author_rows):
        j = pos.get(row["author_id"])
        if j is not None:
            out[i] = transformer_emb[j]
            n_loaded += 1
    if n_loaded == 0:
        raise RuntimeError(f"no transformer features aligned for {feature_model}")
    return l2_normalize_np(out)


def build_meta_features(n_meta: int, emb_dim: int) -> np.ndarray:
    assignments = load_json(DATA / "meta" / "meta_assignments.json")
    fine_to_meta = {int(k): int(v) for k, v in assignments["fine_to_meta"].items()}
    fine_centroids = np.load(CLUSTERING / "hdbscan_fine" / "centroids_qwen.npy", mmap_mode="r")
    fine_sizes = load_json(CLUSTERING / "hdbscan_fine" / "sizes.json")
    out = np.zeros((n_meta, emb_dim), dtype=np.float32)
    weights = np.zeros((n_meta,), dtype=np.float32)
    for fine_id, meta_id in fine_to_meta.items():
        if meta_id < 0 or meta_id >= n_meta:
            continue
        w = float(fine_sizes.get(str(fine_id), 1))
        out[meta_id] += np.asarray(fine_centroids[fine_id], dtype=np.float32) * w
        weights[meta_id] += w
    nonzero = weights > 0
    out[nonzero] /= weights[nonzero, None]
    return l2_normalize_np(out)


def row_normalized_sparse(
    rows: list[int],
    cols: list[int],
    weights: list[float],
    shape: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    if not rows:
        idx = torch.empty((2, 0), dtype=torch.long, device=device)
        val = torch.empty((0,), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(idx, val, shape, device=device).coalesce()
    row_np = np.asarray(rows, dtype=np.int64)
    col_np = np.asarray(cols, dtype=np.int64)
    weight_np = np.asarray(weights, dtype=np.float32)
    denom = np.bincount(row_np, weights=weight_np, minlength=shape[0]).astype(np.float32)
    weight_np = weight_np / np.maximum(denom[row_np], 1e-12)
    idx = torch.tensor(np.vstack([row_np, col_np]), dtype=torch.long, device=device)
    val = torch.tensor(weight_np, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, shape, device=device).coalesce()


def build_author_author_relation(
    author_rows: list[dict[str, Any]],
    author_to_idx: dict[str, int],
    cutoff: int,
    n_total: int,
    device: torch.device,
) -> torch.Tensor:
    paper_to_authors: dict[int, list[int]] = {}
    for row in author_rows:
        author_idx = author_to_idx[row["author_id"]]
        for paper_idx, year in zip(row["paper_idxs"], row["years"], strict=False):
            if int(year) > cutoff:
                continue
            paper_to_authors.setdefault(int(paper_idx), []).append(author_idx)

    edge_counts: dict[tuple[int, int], int] = {}
    for authors in paper_to_authors.values():
        uniq = sorted(set(authors))
        for i, src in enumerate(uniq):
            for dst in uniq[i + 1 :]:
                edge_counts[(src, dst)] = edge_counts.get((src, dst), 0) + 1
                edge_counts[(dst, src)] = edge_counts.get((dst, src), 0) + 1

    rows = [src for src, _ in edge_counts]
    cols = [dst for _, dst in edge_counts]
    # Binary relation: repeated old collaborations should not dominate recommendation
    # of new coauthors.
    weights = [1.0] * len(rows)
    return row_normalized_sparse(rows, cols, weights, (n_total, n_total), device)


def build_author_meta_relations(
    author_rows: list[dict[str, Any]],
    author_to_idx: dict[str, int],
    cutoff: int,
    max_history: int,
    n_authors: int,
    n_meta: int,
    min_weight: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    assignments = load_json(DATA / "meta" / "meta_assignments.json")
    fine_to_meta = np.full((len(assignments["fine_to_meta"]),), -1, dtype=np.int64)
    for k, v in assignments["fine_to_meta"].items():
        fine_to_meta[int(k)] = int(v)
    proba = np.load(CLUSTERING / "hdbscan_fine" / "proba.npy", mmap_mode="r")

    am_rows: list[int] = []
    am_cols: list[int] = []
    am_weights: list[float] = []
    ma_rows: list[int] = []
    ma_cols: list[int] = []
    ma_weights: list[float] = []

    n_authors_with_meta = 0
    n_author_meta_edges = 0
    for row in author_rows:
        author_idx = author_to_idx[row["author_id"]]
        idxs = history_paper_idxs(row, cutoff, max_history)
        if not idxs:
            continue
        fine_mass = np.asarray(proba[idxs], dtype=np.float32).sum(axis=0)
        meta_mass = np.bincount(
            fine_to_meta[fine_to_meta >= 0],
            weights=fine_mass[fine_to_meta >= 0],
            minlength=n_meta,
        ).astype(np.float32)
        total = float(meta_mass.sum())
        if total <= 1e-12:
            continue
        meta_mass /= total
        keep = np.flatnonzero(meta_mass >= min_weight)
        if len(keep) == 0:
            keep = np.array([int(meta_mass.argmax())], dtype=np.int64)
        n_authors_with_meta += 1
        for meta_id in keep.tolist():
            w = float(meta_mass[meta_id])
            meta_node = n_authors + int(meta_id)
            am_rows.append(author_idx)
            am_cols.append(meta_node)
            am_weights.append(w)
            ma_rows.append(meta_node)
            ma_cols.append(author_idx)
            ma_weights.append(w)
            n_author_meta_edges += 1

    shape = (n_authors + n_meta, n_authors + n_meta)
    meta = {
        "n_authors_with_meta_edges": n_authors_with_meta,
        "n_author_meta_edges": n_author_meta_edges,
        "meta_min_weight": min_weight,
    }
    return (
        row_normalized_sparse(am_rows, am_cols, am_weights, shape, device),
        row_normalized_sparse(ma_rows, ma_cols, ma_weights, shape, device),
        meta,
    )


def build_training_pairs(
    author_to_idx: dict[str, int],
    split: str,
) -> tuple[list[int], list[list[int]], np.ndarray, list[set[int]]]:
    examples = split_examples(split)
    author_pool = split_author_set(split)
    pool_idx = np.array(
        [author_to_idx[aid] for aid in sorted(author_pool) if aid in author_to_idx],
        dtype=np.int64,
    )
    anchors: list[int] = []
    positives: list[list[int]] = []
    excluded: list[set[int]] = []
    for ex in examples:
        aid = ex["author_id"]
        if aid not in author_to_idx:
            continue
        pos_ids = relevant_future_coauthors(ex, author_pool=author_pool)
        pos_idx = sorted(author_to_idx[x] for x in pos_ids if x in author_to_idx)
        if not pos_idx:
            continue
        anchor_idx = author_to_idx[aid]
        past_idx = {
            author_to_idx[x]
            for x in past_coauthors(aid, int(ex["cutoff_year"]))
            if x in author_to_idx
        }
        anchors.append(anchor_idx)
        positives.append(pos_idx)
        excluded.append({anchor_idx, *past_idx, *pos_idx})
    return anchors, positives, pool_idx, excluded


def build_graph_data(config: GraphRetrievalConfig, device: torch.device) -> GraphData:
    cutoff = history_cutoff()
    author_rows = load_author_rows()
    author_ids = [row["author_id"] for row in author_rows]
    author_to_idx = {aid: i for i, aid in enumerate(author_ids)}
    n_authors = len(author_ids)
    n_meta = 76 if config.variant == "author_metacluster" else 0
    n_total = n_authors + n_meta

    author_x = build_author_features(
        author_rows,
        cutoff,
        config.max_history,
        config.feature_source,
        config.feature_model,
    )
    if n_meta:
        meta_x = build_meta_features(n_meta, author_x.shape[1])
        x_np = np.vstack([author_x, meta_x]).astype(np.float32, copy=False)
    else:
        x_np = author_x.astype(np.float32, copy=False)

    relation_adj = {
        "author_author": build_author_author_relation(
            author_rows,
            author_to_idx,
            cutoff,
            n_total,
            device,
        )
    }
    graph_meta: dict[str, Any] = {
        "cutoff": cutoff,
        "n_authors": n_authors,
        "n_meta": n_meta,
        "n_total_nodes": n_total,
        "n_author_author_edges_directed": int(relation_adj["author_author"]._nnz()),
        "feature_source": config.feature_source,
        "feature_model": config.feature_model,
    }
    if n_meta:
        am_adj, ma_adj, am_meta = build_author_meta_relations(
            author_rows,
            author_to_idx,
            cutoff,
            config.max_history,
            n_authors,
            n_meta,
            config.meta_min_weight,
            device,
        )
        relation_adj["author_meta"] = am_adj
        relation_adj["meta_author"] = ma_adj
        graph_meta.update(am_meta)
        graph_meta["n_author_meta_edges_directed"] = int(am_adj._nnz() + ma_adj._nnz())

    anchors, positives, pool_idx, excluded = build_training_pairs(author_to_idx, "train")
    graph_meta["n_train_anchors"] = len(anchors)
    graph_meta["n_train_positive_pairs"] = int(sum(len(x) for x in positives))

    return GraphData(
        author_ids=author_ids,
        n_authors=n_authors,
        n_meta=n_meta,
        x=torch.tensor(x_np, dtype=torch.float32, device=device),
        relation_adj=relation_adj,
        train_anchor_idx=anchors,
        train_positive_idx=positives,
        train_negative_pool_idx=pool_idx,
        train_excluded_idx=excluded,
        meta=graph_meta,
    )


def sample_negatives(
    pool_idx: np.ndarray,
    excluded: set[int],
    n_negatives: int,
    rng: np.random.Generator,
) -> np.ndarray:
    out: list[int] = []
    # Rejection sampling is fine here: train split is large and exclusions are small.
    while len(out) < n_negatives:
        need = n_negatives - len(out)
        draw = rng.choice(pool_idx, size=min(len(pool_idx), need * 2), replace=False)
        for item in draw.tolist():
            if item not in excluded:
                out.append(int(item))
                if len(out) >= n_negatives:
                    break
    return np.asarray(out, dtype=np.int64)


def graph_infonce_loss(
    z: torch.Tensor,
    graph: GraphData,
    config: GraphRetrievalConfig,
    rng: np.random.Generator,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    order = rng.permutation(len(graph.train_anchor_idx))
    max_pos = max(1, min(config.max_positives, max(len(x) for x in graph.train_positive_idx)))
    for start in range(0, len(order), config.loss_batch_size):
        batch_ids = order[start : start + config.loss_batch_size]
        anchors_np = np.asarray([graph.train_anchor_idx[i] for i in batch_ids], dtype=np.int64)
        pos_np = np.full((len(batch_ids), max_pos), -1, dtype=np.int64)
        pos_mask_np = np.zeros((len(batch_ids), max_pos), dtype=bool)
        neg_np = np.zeros((len(batch_ids), config.n_negatives), dtype=np.int64)
        for row, row_id in enumerate(batch_ids.tolist()):
            pos = graph.train_positive_idx[row_id]
            if len(pos) > max_pos:
                chosen = rng.choice(np.asarray(pos, dtype=np.int64), size=max_pos, replace=False)
            else:
                chosen = np.asarray(pos, dtype=np.int64)
            pos_np[row, : len(chosen)] = chosen
            pos_mask_np[row, : len(chosen)] = True
            neg_np[row] = sample_negatives(
                graph.train_negative_pool_idx,
                graph.train_excluded_idx[row_id],
                config.n_negatives,
                rng,
            )

        anchors = torch.tensor(anchors_np, dtype=torch.long, device=z.device)
        pos = torch.tensor(np.maximum(pos_np, 0), dtype=torch.long, device=z.device)
        pos_mask = torch.tensor(pos_mask_np, dtype=torch.bool, device=z.device)
        neg = torch.tensor(neg_np, dtype=torch.long, device=z.device)

        anchor_z = z[anchors]
        pos_scores = torch.einsum("bpd,bd->bp", z[pos], anchor_z) / config.tau
        neg_scores = torch.einsum("bnd,bd->bn", z[neg], anchor_z) / config.tau
        pos_scores = pos_scores.masked_fill(~pos_mask, torch.finfo(pos_scores.dtype).min)
        log_pos = torch.logsumexp(pos_scores, dim=1)
        log_neg = torch.logsumexp(neg_scores, dim=1)
        losses.append((-log_pos + torch.logaddexp(log_pos, log_neg)).mean())
    return torch.stack(losses).mean()


def evaluate_graph_embeddings(
    z: np.ndarray,
    author_ids: list[str],
    split: str,
    max_history: int,
) -> dict[str, float]:
    author_to_idx = {aid: i for i, aid in enumerate(author_ids)}
    cutoff = cutoff_year_for_split(split)
    candidate_ids, _ = cutoff_examples(cutoff, max_history, split=split)
    examples = split_examples(split)
    focal_idx = [author_to_idx[ex["author_id"]] for ex in examples if ex["author_id"] in author_to_idx]
    filtered_examples = [ex for ex in examples if ex["author_id"] in author_to_idx]
    candidate_idx = [author_to_idx[aid] for aid in candidate_ids if aid in author_to_idx]
    filtered_candidate_ids = [aid for aid in candidate_ids if aid in author_to_idx]
    return evaluate_coauthor_from_embeddings(
        z[focal_idx],
        filtered_examples,
        filtered_candidate_ids,
        z[candidate_idx],
        split,
    )


def rows_from_metrics(model_name: str, split: str, metrics: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for k in (10, 50, 100):
        rows.append(
            {
                "split": split,
                "model": model_name,
                "K": k,
                "hit": metrics[f"hit@{k}"],
                "mrr": metrics[f"mrr@{k}"],
                "ndcg": metrics[f"ndcg@{k}"],
                "n_examples": int(metrics["n_examples"]),
                "n_with_relevant": int(metrics.get("n_with_relevant", metrics["n_examples"])),
            }
        )
    return rows


def train_graph_model(args: argparse.Namespace) -> None:
    if args.variant not in GRAPH_MODEL_NAMES:
        raise ValueError(f"unknown graph variant: {args.variant}")
    model_name = args.model_name or GRAPH_MODEL_NAMES[args.variant]
    out_dir = author_model_dir(model_name)
    if (out_dir / "best.pt").exists() and not args.force:
        print(f"[skip] {model_name} best.pt exists; use --force")
        return

    seed_everything(args.seed)
    config = GraphRetrievalConfig(
        variant=args.variant,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        tau=args.tau,
        n_negatives=args.n_negatives,
        max_positives=args.max_positives,
        loss_batch_size=args.loss_batch_size,
        res_scale_init=args.res_scale_init,
        feature_source=args.feature_source,
        feature_model=args.feature_model,
        max_history=args.max_history,
        meta_min_weight=args.meta_min_weight,
        seed=args.seed,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"[info] build graph: variant={config.variant} device={device}")
    graph = build_graph_data(config, device)
    config.emb_dim = int(graph.x.shape[1])
    save_json(out_dir / "config.json", config.to_dict())
    save_json(out_dir / "graph_meta.json", graph.meta)
    print(f"[info] graph meta: {json.dumps(graph.meta, ensure_ascii=False)}")

    model = RelGraphSAGE(config, list(graph.relation_adj)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed)

    log: list[dict[str, Any]] = []
    best_metric = -1.0
    best_epoch = 0
    patience_left = config.patience
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        z = model(graph.x, graph.relation_adj, graph.n_authors)
        loss = graph_infonce_loss(z, graph, config, rng)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z_eval = model(graph.x, graph.relation_adj, graph.n_authors).detach().cpu().numpy()
        val_metrics = evaluate_graph_embeddings(
            z_eval,
            graph.author_ids,
            "val",
            config.max_history,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(loss.detach().cpu().item()),
            **{f"val_{k}": v for k, v in val_metrics.items() if k != "n_examples"},
            "val_n_examples": int(val_metrics["n_examples"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        log.append(row)
        save_json(out_dir / "train_log.json", log)
        save_training_plots(out_dir, log)
        score = float(row["val_ndcg@50"])
        print(
            f"[epoch] {model_name} {epoch}: loss={row['train_loss']:.4f} "
            f"val_ndcg@10={row['val_ndcg@10']:.5f} val_ndcg@50={score:.5f}"
        )
        payload = {
            "state_dict": model.state_dict(),
            "model_config": config.to_dict(),
            "graph_meta": graph.meta,
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(payload, out_dir / "last.pt")
        if score > best_metric + 1e-8:
            best_metric = score
            best_epoch = epoch
            patience_left = config.patience
            torch.save(payload, out_dir / "best.pt")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(
                    f"[early-stop] {model_name}: best_epoch={best_epoch} "
                    f"best_val_ndcg@50={best_metric:.5f}"
                )
                break

    best_payload = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best_payload["state_dict"])
    model.eval()
    with torch.no_grad():
        z = model(graph.x, graph.relation_adj, graph.n_authors).detach().cpu().numpy()
    np.save(out_dir / "author_embeddings.npy", z.astype(np.float32, copy=False))
    rows: list[dict[str, Any]] = []
    for split in ("val", "test"):
        metrics = evaluate_graph_embeddings(z, graph.author_ids, split, config.max_history)
        print(
            f"[metrics] {split} {model_name}: "
            f"ndcg@10={metrics['ndcg@10']:.5f} ndcg@50={metrics['ndcg@50']:.5f} "
            f"hit@10={metrics['hit@10']:.5f}"
        )
        rows.extend(rows_from_metrics(model_name, split, metrics))
    metrics_path = out_dir / "metrics.csv"
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f"[done] wrote {metrics_path}")


def write_comparison() -> None:
    rows = []
    retrieval_dir = DATA.parent / "results" / "retrieval"
    baseline_path = retrieval_dir / "test_metrics.csv"
    if baseline_path.exists():
        base = pd.read_csv(baseline_path)
        base.insert(0, "split", "test")
        rows.append(base)
    for model_name in GRAPH_MODEL_NAMES.values():
        metrics_path = author_model_dir(model_name) / "metrics.csv"
        if metrics_path.exists():
            rows.append(pd.read_csv(metrics_path))
    if not rows:
        return
    out = pd.concat(rows, ignore_index=True)
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    out_path = retrieval_dir / "graph_comparison_metrics.csv"
    out.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train graph coauthor retriever")
    parser.add_argument("--variant", choices=sorted(GRAPH_MODEL_NAMES), required=True)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--n-negatives", type=int, default=512)
    parser.add_argument("--max-positives", type=int, default=8)
    parser.add_argument("--loss-batch-size", type=int, default=128)
    parser.add_argument("--res-scale-init", type=float, default=0.1)
    parser.add_argument("--feature-source", choices=["mean", "transformer"], default="mean")
    parser.add_argument("--feature-model", default=COAUTHOR_INFONCE_MODEL_NAME)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--meta-min-weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    train_graph_model(args)
    write_comparison()
    return 0


if __name__ == "__main__":
    sys.exit(main())
