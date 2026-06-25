from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

from common.compat import RESULTS, save_json
from evaluation.hybrid_retrieval import (
    ensure_transformer_author_embeddings,
    evaluate_method_embeddings,
    l2_normalize,
    save_fusion_model,
)
from recommendation.graph_retrieval import GraphRetrievalConfig, build_graph_data
from recommendation.training_utils import COAUTHOR_INFONCE_MODEL_NAME, pick_device


def lightgcn_embeddings(
    x: torch.Tensor,
    adj: torch.Tensor,
    n_authors: int,
    n_layers: int,
    decay: float,
) -> np.ndarray:
    h = x[:n_authors]
    layers = [h]
    cur = h
    author_adj = adj.coalesce()
    for _ in range(n_layers):
        cur = torch.sparse.mm(author_adj, cur)
        cur = torch.nn.functional.normalize(cur, p=2, dim=1)
        layers.append(cur)
    weights = torch.tensor([decay**i for i in range(n_layers + 1)], dtype=torch.float32, device=x.device)
    weights = weights / weights.sum()
    out = sum(w * layer for w, layer in zip(weights, layers, strict=False))
    return l2_normalize(out.detach().cpu().numpy().astype(np.float32, copy=False))


def appnp_embeddings(
    x: torch.Tensor,
    adj: torch.Tensor,
    n_authors: int,
    n_steps: int,
    teleport: float,
) -> np.ndarray:
    h0 = x[:n_authors]
    h = h0
    author_adj = adj.coalesce()
    for _ in range(n_steps):
        h = (1.0 - teleport) * torch.sparse.mm(author_adj, h) + teleport * h0
        h = torch.nn.functional.normalize(h, p=2, dim=1)
    return l2_normalize(h.detach().cpu().numpy().astype(np.float32, copy=False))


def rows_for_method(method: str, author_ids: list[str], emb: np.ndarray, max_history: int) -> list[dict[str, Any]]:
    rows = evaluate_method_embeddings(method, author_ids, emb, max_history)
    return rows


def best_by_val50(rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = None
    for row in rows:
        if row["split"] == "val" and int(row["K"]) == 50:
            if best is None or float(row["ndcg"]) > float(best["ndcg"]):
                best = row
    if best is None:
        raise RuntimeError("no val K=50 rows")
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate LightGCN/APPNP-style graph propagation")
    parser.add_argument("--feature-model", default=COAUTHOR_INFONCE_MODEL_NAME)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force-transformer-cache", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    author_ids, transformer_emb = ensure_transformer_author_embeddings(
        args.feature_model,
        args.max_history,
        device,
        args.force_transformer_cache,
    )
    # build_graph_data gives the normalized coauthor adjacency in the same author-id order.
    graph = build_graph_data(
        GraphRetrievalConfig(
            variant="author",
            feature_source="transformer",
            feature_model=args.feature_model,
            max_history=args.max_history,
        ),
        device,
    )
    graph_author_ids = graph.author_ids
    graph_pos = {aid: i for i, aid in enumerate(graph_author_ids)}
    keep_ids = [aid for aid in author_ids if aid in graph_pos]
    keep_graph_idx = [graph_pos[aid] for aid in keep_ids]
    # Rebuild adjacency over the aligned subset by evaluating on the full graph and
    # slicing the propagated embeddings through keep_graph_idx.
    all_rows: list[dict[str, Any]] = []
    candidates: dict[str, tuple[list[str], np.ndarray, dict[str, Any]]] = {}

    for n_layers in (1, 2, 3, 4):
        for decay in (0.5, 0.7, 0.9, 1.0):
            emb_all = lightgcn_embeddings(graph.x, graph.relation_adj["author_author"], graph.n_authors, n_layers, decay)
            emb = emb_all[keep_graph_idx]
            method = f"prop_lightgcn_l{n_layers}_d{str(decay).replace('.', 'p')}"
            rows = rows_for_method(method, keep_ids, emb, args.max_history)
            for row in rows:
                row.update({"family": "lightgcn", "layers": n_layers, "decay": decay, "teleport": ""})
            all_rows.extend(rows)
            candidates[method] = (keep_ids, emb, {"family": "lightgcn", "layers": n_layers, "decay": decay})

    for n_steps in (2, 4, 8, 12):
        for teleport in (0.05, 0.1, 0.2, 0.3):
            emb_all = appnp_embeddings(graph.x, graph.relation_adj["author_author"], graph.n_authors, n_steps, teleport)
            emb = emb_all[keep_graph_idx]
            method = f"prop_appnp_s{n_steps}_a{str(teleport).replace('.', 'p')}"
            rows = rows_for_method(method, keep_ids, emb, args.max_history)
            for row in rows:
                row.update({"family": "appnp", "layers": n_steps, "decay": "", "teleport": teleport})
            all_rows.extend(rows)
            candidates[method] = (keep_ids, emb, {"family": "appnp", "steps": n_steps, "teleport": teleport})

    out_dir = RESULTS / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    metrics_path = out_dir / "propagation_metrics.csv"
    df.to_csv(metrics_path, index=False)
    best = best_by_val50(all_rows)
    method = str(best["method"])
    ids, emb, cfg = candidates[method]
    save_fusion_model(
        method,
        ids,
        emb,
        {
            "kind": "propagation",
            "feature_model": args.feature_model,
            "selected_by": "val_ndcg@50",
            "val_ndcg@50": float(best["ndcg"]),
            "max_history": args.max_history,
            **cfg,
        },
    )
    pd.DataFrame([row for row in all_rows if row["method"] == method]).to_csv(
        RESULTS.parent / "models" / method / "metrics.csv",
        index=False,
    )
    save_json(out_dir / "propagation_winner.json", {"method": method, "val_ndcg@50": float(best["ndcg"]), **cfg})
    print(f"[winner] {method} val_ndcg@50={float(best['ndcg']):.5f}")
    print(f"[done] wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
