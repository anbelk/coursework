from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from author_model_utils import (
    CONFIGS,
    AuthorExampleDataset,
    collate_examples,
    evaluate_retrieval,
    load_author_arrays,
    load_model_checkpoint,
    load_paper_years,
    model_dir,
    pick_device,
    split_examples,
)
from pipeline_common import RESULTS, save_json


def rows_from_metrics(model_name: str, metrics: dict[str, float]) -> list[dict]:
    rows = []
    for k in (10, 50, 100):
        rows.append(
            {
                "model": model_name,
                "K": k,
                "hit": metrics[f"hit@{k}"],
                "mrr": metrics[f"mrr@{k}"],
                "ndcg": metrics[f"ndcg@{k}"],
                "n_examples": int(metrics["n_examples"]),
            }
        )
    return rows


def select_best(val_path: Path) -> dict:
    df = pd.read_csv(val_path)
    pivot = df.pivot(index="model", columns="K", values=["ndcg", "mrr", "hit"])
    scored = []
    for model in pivot.index:
        scored.append(
            {
                "model": model,
                "ndcg@10": float(pivot.loc[model, ("ndcg", 10)]),
                "ndcg@100": float(pivot.loc[model, ("ndcg", 100)]),
                "mrr@10": float(pivot.loc[model, ("mrr", 10)]),
            }
        )
    scored.sort(key=lambda x: (x["ndcg@10"], x["ndcg@100"], x["mrr@10"]), reverse=True)
    best = {
        "best_model": scored[0]["model"],
        "selection_rule": "max val nDCG@10; tie-break nDCG@100 then MRR@10",
        "ranking": scored,
    }
    save_json(RESULTS / "retrieval" / "best_model.json", best)
    return best


def plot_val_curves() -> None:
    out_dir = RESULTS / "retrieval"
    logs = []
    for name in CONFIGS:
        log_path = model_dir(name) / "train_log.json"
        if log_path.exists():
            rows = pd.read_json(log_path)
            rows["model"] = name
            logs.append(rows)
    if not logs:
        return
    df = pd.concat(logs, ignore_index=True)
    plt.figure(figsize=(8, 5))
    for name, part in df.groupby("model"):
        plt.plot(part["epoch"], part["val_ndcg@10"], marker="o", label=name)
    plt.xlabel("epoch")
    plt.ylabel("val nDCG@10")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "val_metric_curves.png", dpi=160)
    plt.close()


def evaluate_model(name: str, split: str, embeddings: np.ndarray, q: np.ndarray, years: np.ndarray, device: torch.device) -> dict[str, float]:
    examples = split_examples(split)
    ds = AuthorExampleDataset(examples, embeddings, q, years)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_examples, num_workers=0)
    model, _ = load_model_checkpoint(model_dir(name) / "best.pt", device)
    candidate_year = 2025 if split == "val" else 2026
    candidates = np.flatnonzero(years == candidate_year).astype(np.int64)
    return evaluate_retrieval(model, loader, examples, embeddings, candidates, device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--models", nargs="*", default=["all"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_dir = RESULTS / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    embeddings, q = load_author_arrays()
    years = load_paper_years()

    if args.split == "test":
        if args.model == "best" or args.model is None:
            best_path = out_dir / "best_model.json"
            best = pd.read_json(best_path, typ="series").to_dict()
            names = [best["best_model"]]
        else:
            names = [args.model]
    elif args.models == ["all"]:
        names = list(CONFIGS)
    else:
        names = args.models

    all_rows = []
    for name in names:
        metrics = evaluate_model(name, args.split, embeddings, q, years, device)
        print(f"[metrics] {args.split} {name}: {metrics}")
        all_rows.extend(rows_from_metrics(name, metrics))

    out_path = out_dir / f"{args.split}_metrics.csv"
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"[done] wrote {out_path}")
    if args.split == "val":
        best = select_best(out_path)
        plot_val_curves()
        print(f"[done] best_model={best['best_model']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
