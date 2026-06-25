from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common.author_splits import history_cutoff
from common.compat import AUTHORS, MODELS, RESULTS, load_json, save_json
from evaluation.coauthor_retrieval import (
    cutoff_examples,
    evaluate_coauthor_from_embeddings,
    model_candidate_embeddings,
)
from recommendation.training_utils import (
    COAUTHOR_INFONCE_MODEL_NAME,
    load_author_arrays,
    load_model_checkpoint,
    load_paper_years,
    model_dir,
    pick_device,
    split_examples,
)


GRAPH_METHODS = ["graphsage_author", "graphsage_author_metacluster"]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norm, 1e-12, out=norm)
    return x / norm


def sorted_author_ids() -> list[str]:
    rows = load_json(AUTHORS / "author_index.json")
    rows.sort(key=lambda x: x["author_id"])
    return [row["author_id"] for row in rows]


def load_embedding_method(method: str) -> tuple[list[str], np.ndarray]:
    ids_path = model_dir(method) / "author_ids.json"
    emb_path = model_dir(method) / "author_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"missing embeddings for {method}: {emb_path}")
    if ids_path.exists():
        ids = load_json(ids_path)
    else:
        ids = sorted_author_ids()
    emb = l2_normalize(np.load(emb_path).astype(np.float32, copy=False))
    if len(ids) != emb.shape[0]:
        raise ValueError(f"{method}: ids={len(ids)} embeddings={emb.shape[0]}")
    return ids, emb


@torch.inference_mode()
def ensure_transformer_author_embeddings(
    model_name: str,
    max_history: int,
    device: torch.device,
    force: bool,
) -> tuple[list[str], np.ndarray]:
    out_dir = model_dir(model_name)
    emb_path = out_dir / "author_embeddings.npy"
    ids_path = out_dir / "author_ids.json"
    if emb_path.exists() and ids_path.exists() and not force:
        return load_embedding_method(model_name)

    cutoff = history_cutoff()
    embeddings, q = load_author_arrays()
    years = load_paper_years()
    model, _ = load_model_checkpoint(out_dir / "best.pt", device)
    author_ids, author_emb = model_candidate_embeddings(
        model,
        cutoff,
        embeddings,
        q,
        years,
        device,
        max_history=max_history,
        split=None,
    )
    author_emb = l2_normalize(author_emb.astype(np.float32, copy=False))
    np.save(emb_path, author_emb)
    save_json(ids_path, author_ids)
    return author_ids, author_emb


def align_embeddings(
    left_ids: list[str],
    left_emb: np.ndarray,
    right_ids: list[str],
    right_emb: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    right_pos = {aid: i for i, aid in enumerate(right_ids)}
    keep_ids = [aid for aid in left_ids if aid in right_pos]
    left_pos = {aid: i for i, aid in enumerate(left_ids)}
    left_aligned = left_emb[[left_pos[aid] for aid in keep_ids]]
    right_aligned = right_emb[[right_pos[aid] for aid in keep_ids]]
    return keep_ids, left_aligned, right_aligned


def fused_embeddings(transformer: np.ndarray, graph: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(alpha)
    beta = 1.0 - alpha
    out = np.concatenate(
        [
            np.sqrt(max(alpha, 0.0)) * transformer,
            np.sqrt(max(beta, 0.0)) * graph,
        ],
        axis=1,
    )
    return l2_normalize(out)


def evaluate_method_embeddings(
    method: str,
    author_ids: list[str],
    author_emb: np.ndarray,
    max_history: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pos = {aid: i for i, aid in enumerate(author_ids)}
    for split in ("val", "test"):
        cutoff = history_cutoff()
        candidate_ids, _ = cutoff_examples(cutoff, max_history, split=split)
        examples = split_examples(split)
        filtered_examples = [ex for ex in examples if ex["author_id"] in pos]
        focal_idx = [pos[ex["author_id"]] for ex in filtered_examples]
        kept_candidates = [aid for aid in candidate_ids if aid in pos]
        candidate_idx = [pos[aid] for aid in kept_candidates]
        metrics = evaluate_coauthor_from_embeddings(
            author_emb[focal_idx],
            filtered_examples,
            kept_candidates,
            author_emb[candidate_idx],
            split,
        )
        for k in (10, 50, 100):
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "K": k,
                    "hit": metrics[f"hit@{k}"],
                    "mrr": metrics[f"mrr@{k}"],
                    "ndcg": metrics[f"ndcg@{k}"],
                    "n_examples": int(metrics["n_examples"]),
                    "n_with_relevant": int(metrics.get("n_with_relevant", metrics["n_examples"])),
                }
            )
    return rows


def save_fusion_model(
    name: str,
    author_ids: list[str],
    emb: np.ndarray,
    config: dict[str, Any],
) -> None:
    out_dir = model_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "author_embeddings.npy", emb.astype(np.float32, copy=False))
    save_json(out_dir / "author_ids.json", author_ids)
    save_json(out_dir / "config.json", config)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate score/embedding fusion for retrieval")
    parser.add_argument("--transformer", default=COAUTHOR_INFONCE_MODEL_NAME)
    parser.add_argument("--graph-methods", nargs="*", default=GRAPH_METHODS)
    parser.add_argument("--alphas", nargs="*", type=float, default=[i / 20 for i in range(21)])
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force-transformer-cache", action="store_true")
    args = parser.parse_args()

    device = pick_device(args.device)
    transformer_ids, transformer_emb = ensure_transformer_author_embeddings(
        args.transformer,
        args.max_history,
        device,
        args.force_transformer_cache,
    )

    all_rows: list[dict[str, Any]] = []
    best_by_graph: dict[str, dict[str, Any]] = {}
    for graph_method in args.graph_methods:
        graph_ids, graph_emb = load_embedding_method(graph_method)
        author_ids, t_aligned, g_aligned = align_embeddings(
            transformer_ids,
            transformer_emb,
            graph_ids,
            graph_emb,
        )
        for alpha in args.alphas:
            method = f"fusion_{args.transformer}_{graph_method}_a{alpha:.2f}".replace(".", "p")
            emb = fused_embeddings(t_aligned, g_aligned, alpha)
            rows = evaluate_method_embeddings(method, author_ids, emb, args.max_history)
            for row in rows:
                row["alpha"] = alpha
                row["transformer"] = args.transformer
                row["graph_method"] = graph_method
            all_rows.extend(rows)
            val50 = next(row for row in rows if row["split"] == "val" and row["K"] == 50)
            current = best_by_graph.get(graph_method)
            if current is None or float(val50["ndcg"]) > float(current["val_ndcg@50"]):
                best_by_graph[graph_method] = {
                    "method": method,
                    "alpha": alpha,
                    "val_ndcg@50": float(val50["ndcg"]),
                    "author_ids": author_ids,
                    "emb": emb,
                }

    out_dir = RESULTS / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_rows)
    metrics_path = out_dir / "hybrid_fusion_metrics.csv"
    df.to_csv(metrics_path, index=False)

    winners = []
    for graph_method, best in best_by_graph.items():
        config = {
            "kind": "fusion",
            "transformer": args.transformer,
            "graph_method": graph_method,
            "alpha_transformer": best["alpha"],
            "alpha_graph": 1.0 - best["alpha"],
            "selected_by": "val_ndcg@50",
            "val_ndcg@50": best["val_ndcg@50"],
            "max_history": args.max_history,
        }
        save_fusion_model(best["method"], best["author_ids"], best["emb"], config)
        method_rows = df[df["method"] == best["method"]].to_dict(orient="records")
        pd.DataFrame(method_rows).to_csv(model_dir(best["method"]) / "metrics.csv", index=False)
        winners.append({k: v for k, v in config.items() if k != "kind"} | {"method": best["method"]})
        print(f"[winner] {graph_method}: {best['method']} val_ndcg@50={best['val_ndcg@50']:.5f}")
    save_json(out_dir / "hybrid_fusion_winners.json", {"winners": winners})
    print(f"[done] wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
