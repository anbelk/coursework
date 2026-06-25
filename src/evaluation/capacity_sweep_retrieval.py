from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from common.compat import MODELS, RANDOM_STATE, RESULTS, save_json
from common.author_splits import HISTORY_CUTOFF, split_author_set
from recommendation.model import AuthorTransformer, AuthorTransformerConfig
from recommendation.training_utils import (
    AuthorExampleDataset,
    collate_examples,
    evaluate_mean_history_retrieval,
    evaluate_retrieval,
    load_author_arrays,
    load_paper_years,
    move_batch,
    pick_device,
    save_checkpoint,
    seed_everything,
    split_examples,
    total_losses,
    warmup_cosine_lambda,
)


CAPACITY_PRESETS: dict[str, dict[str, int]] = {
    "micro": {"d_model": 64, "n_layers": 1, "n_heads": 2, "dim_feedforward": 128},
    "tiny": {"d_model": 128, "n_layers": 2, "n_heads": 4, "dim_feedforward": 256},
    "small": {"d_model": 256, "n_layers": 2, "n_heads": 4, "dim_feedforward": 512},
    "medium": {"d_model": 384, "n_layers": 4, "n_heads": 6, "dim_feedforward": 1024},
    "large": {"d_model": 512, "n_layers": 6, "n_heads": 8, "dim_feedforward": 1536},
}


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval_loss(
    model,
    loader,
    embeddings_t,
    embeddings,
    q,
    years,
    max_history,
    device,
    tau,
    loss_type: str,
    n_negatives: int,
    author_pool: set[str] | None = None,
) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_batch(batch, device)
            loss, _ = total_losses(
                model,
                batch,
                embeddings_t,
                embeddings,
                q,
                years,
                max_history,
                device,
                tau,
                "coauthor_infonce",
                n_negatives,
                author_pool=author_pool,
            )
            losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def train_and_eval(
    name: str,
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    test_examples: list[dict[str, Any]],
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    model_cfg: AuthorTransformerConfig,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    train_pool = split_author_set("train")
    val_pool = split_author_set("val")
    test_pool = split_author_set("test")

    train_ds = AuthorExampleDataset(
        train_examples, embeddings, q, years, max_history=args.max_history, author_pool=train_pool
    )
    val_ds = AuthorExampleDataset(
        val_examples, embeddings, q, years, max_history=args.max_history, author_pool=val_pool
    )
    test_ds = AuthorExampleDataset(
        test_examples, embeddings, q, years, max_history=args.max_history, author_pool=test_pool
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_examples, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_examples, num_workers=0)
    val_retrieval_loader = DataLoader(
        val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_examples, num_workers=0
    )
    test_retrieval_loader = DataLoader(
        test_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_examples, num_workers=0
    )

    model = AuthorTransformer(model_cfg).to(device)
    n_params = count_parameters(model)
    embeddings_t = torch.tensor(embeddings, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_lambda(step, total_steps, warmup_steps),
    )

    val_cutoff = HISTORY_CUTOFF

    def snapshot(epoch: int, train_loss: float) -> dict[str, Any]:
        val_r = evaluate_retrieval(
            model, val_retrieval_loader, val_examples, embeddings, q, years,
            val_cutoff, device, "val", args.max_history,
        )
        test_r = evaluate_retrieval(
            model, test_retrieval_loader, test_examples, embeddings, q, years,
            val_cutoff, device, "test", args.max_history,
        )
        return {
            "preset": name,
            "best_epoch": epoch,
            "train_loss": train_loss,
            "n_params": n_params,
            "val_ndcg@10": float(val_r["ndcg@10"]),
            "val_ndcg@50": float(val_r["ndcg@50"]),
            "val_hit@10": float(val_r["hit@10"]),
            "test_ndcg@10": float(test_r["ndcg@10"]),
            "test_ndcg@50": float(test_r["ndcg@50"]),
            "test_hit@10": float(test_r["hit@10"]),
            "select_score": float(val_r["ndcg@50"]),
        }

    best_score = -1.0
    best_state: dict[str, Any] = {}
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = total_losses(
                model,
                batch,
                embeddings_t,
                embeddings,
                q,
                years,
                args.max_history,
                device,
                args.tau,
                "coauthor_infonce",
                args.n_negatives,
                author_pool=train_pool,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))

        state = snapshot(epoch, float(np.mean(train_losses)))
        # Select on val nDCG@50: counts more positives than @10 -> less noisy on the
        # tiny coauthor eval set.
        if state["select_score"] > best_score + 1e-8:
            best_score = state["select_score"]
            best_state = state
            patience_left = args.patience
            save_checkpoint(
                out_dir / "best.pt",
                model,
                {"epoch": epoch, "metrics": best_state, "model_config": model_cfg.to_dict(), "preset": name},
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if not best_state:
        best_state = snapshot(args.epochs, float(np.mean(train_losses)))

    best_state["checkpoint"] = str(out_dir / "best.pt")
    print(
        f"[done] {name} params={n_params:,} epoch={best_state['best_epoch']} "
        f"val_ndcg@10={best_state['val_ndcg@10']:.4f} val_ndcg@50={best_state['val_ndcg@50']:.4f} "
        f"test_ndcg@10={best_state['test_ndcg@10']:.4f}"
    )
    return best_state


def plot_capacity(rows: list[dict[str, Any]], mean_val_ndcg: float, out_path: Path) -> None:
    df = pd.DataFrame(rows).sort_values("n_params")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["n_params"], df["val_ndcg@10"], marker="s", label="val nDCG@10")
    ax.plot(df["n_params"], df["test_ndcg@10"], marker="o", label="test nDCG@10")
    ax.axhline(mean_val_ndcg, color="C2", linestyle="--", label=f"mean baseline ({mean_val_ndcg:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Capacity sweep: quality vs model size")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capacity sweep to detect memorization")
    parser.add_argument("--presets", nargs="*", default=list(CAPACITY_PRESETS))
    parser.add_argument("--loss", choices=["logsumexp", "infonce"], default="infonce")
    parser.add_argument("--n-negatives", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = RESULTS / "retrieval" / f"capacity_sweep_{args.loss}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    plot_path = out_dir / "capacity_ndcg10.png"

    if summary_path.exists() and not args.force:
        print(f"[skip] {summary_path} exists; use --force")
        return 0

    seed_everything(args.seed)
    embeddings, q = load_author_arrays()
    years = load_paper_years()
    train_examples = split_examples("train")
    val_examples = split_examples("val")
    test_examples = split_examples("test")
    val_pool = split_author_set("val")

    val_ds = AuthorExampleDataset(
        val_examples, embeddings, q, years, max_history=args.max_history, author_pool=val_pool
    )
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_examples)
    val_cutoff = HISTORY_CUTOFF
    mean_metrics = evaluate_mean_history_retrieval(
        val_loader, val_examples, embeddings, q, years, val_cutoff, "val", args.max_history
    )
    mean_val_ndcg = float(mean_metrics["ndcg@10"])
    print(
        f"[info] train={len(train_examples)} val={len(val_examples)} test={len(test_examples)} "
        f"cutoff={val_cutoff} mean_val_ndcg@10={mean_val_ndcg:.5f}"
    )

    rows: list[dict[str, Any]] = []
    for preset in args.presets:
        if preset not in CAPACITY_PRESETS:
            print(f"[skip] unknown preset: {preset}")
            continue
        hp = CAPACITY_PRESETS[preset]
        model_cfg = AuthorTransformerConfig(
            emb_dim=embeddings.shape[1],
            q_dim=q.shape[1],
            max_history=args.max_history,
            dropout=args.dropout,
            **hp,
        )
        ckpt_dir = MODELS / f"capacity_sweep_{args.loss}" / preset
        print(f"[train] preset={preset} {hp}")
        row = train_and_eval(
            preset,
            train_examples,
            val_examples,
            test_examples,
            embeddings,
            q,
            years,
            model_cfg,
            ckpt_dir,
            args,
        )
        row["loss"] = args.loss
        row["mean_baseline_val_ndcg@10"] = mean_val_ndcg
        rows.append(row)

    payload = {
        "loss": args.loss,
        "history_cutoff": HISTORY_CUTOFF,
        "mean_baseline_val": mean_metrics,
        "presets": {k: CAPACITY_PRESETS[k] for k in args.presets if k in CAPACITY_PRESETS},
        "rows": rows,
    }
    save_json(summary_path, payload)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    plot_capacity(rows, mean_val_ndcg, plot_path)
    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
