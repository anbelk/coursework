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
    MEAN_BASELINE_NAME,
    move_batch,
    pick_device,
    save_checkpoint,
    seed_everything,
    split_examples,
    total_losses,
    warmup_cosine_lambda,
)


def subsample_train_by_author(
    examples: list[dict[str, Any]],
    fraction: float,
    seed: int = RANDOM_STATE,
) -> tuple[list[dict[str, Any]], int, int]:
    authors = sorted({ex["author_id"] for ex in examples})
    rng = np.random.default_rng(seed)
    n_authors = max(1, int(round(len(authors) * fraction)))
    chosen = set(rng.choice(authors, size=n_authors, replace=False).tolist())
    subset = [ex for ex in examples if ex["author_id"] in chosen]
    return subset, len(subset), n_authors


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
) -> dict[str, float]:
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
    return {"val_loss": float(np.mean(losses)) if losses else 0.0}


def train_fraction(
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    train_pool = split_author_set("train")
    val_pool = split_author_set("val")
    val_ds = AuthorExampleDataset(
        val_examples, embeddings, q, years, max_history=args.max_history, author_pool=val_pool
    )
    train_ds = AuthorExampleDataset(
        train_examples, embeddings, q, years, max_history=args.max_history, author_pool=train_pool
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_examples,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_examples,
        num_workers=0,
    )
    val_retrieval_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_examples,
        num_workers=0,
    )

    model_cfg = AuthorTransformerConfig(
        emb_dim=embeddings.shape[1],
        q_dim=q.shape[1],
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_history=args.max_history,
    )
    model = AuthorTransformer(model_cfg).to(device)
    embeddings_t = torch.tensor(embeddings, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_lambda(step, total_steps, warmup_steps),
    )

    val_cutoff = HISTORY_CUTOFF
    best_metric = -1.0
    best_epoch = 0
    best_row: dict[str, Any] = {}
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

        val_losses = eval_loss(
            model,
            val_loader,
            embeddings_t,
            embeddings,
            q,
            years,
            args.max_history,
            device,
            args.tau,
            args.loss,
            args.n_negatives,
            author_pool=val_pool,
        )
        retrieval = evaluate_retrieval(
            model,
            val_retrieval_loader,
            val_examples,
            embeddings,
            q,
            years,
            val_cutoff,
            device,
            "val",
            args.max_history,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **val_losses,
            **{f"val_{k}": v for k, v in retrieval.items() if k != "n_examples"},
        }
        score = float(row["val_ndcg@10"])
        if score > best_metric + 1e-8:
            best_metric = score
            best_epoch = epoch
            best_row = row.copy()
            patience_left = args.patience
            save_checkpoint(
                out_dir / "best.pt",
                model,
                {
                    "epoch": epoch,
                    "metrics": row,
                    "model_config": model_cfg.to_dict(),
                    "n_train_examples": len(train_examples),
                },
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    return {
        "best_epoch": best_epoch,
        "best_val_ndcg@10": best_metric,
        "best_val_hit@10": float(best_row.get("val_hit@10", 0.0)),
        "best_val_mrr@10": float(best_row.get("val_mrr@10", 0.0)),
        "best_val_ndcg@50": float(best_row.get("val_ndcg@50", 0.0)),
        "best_val_hit@50": float(best_row.get("val_hit@50", 0.0)),
        "best_val_loss": float(best_row.get("val_loss", 0.0)),
        "checkpoint": str(out_dir / "best.pt"),
    }


def plot_learning_curve(
    rows: list[dict[str, Any]], mean_ndcg10: float, out_path: Path, loss: str
) -> None:
    df = pd.DataFrame(rows).sort_values("train_fraction")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["train_fraction"], df["best_val_ndcg@10"], marker="o", label="Transformer")
    ax.axhline(mean_ndcg10, color="C1", linestyle="--", label=f"mean baseline ({mean_ndcg10:.3f})")
    ax.set_xlabel("train fraction (by author)")
    ax.set_ylabel("val nDCG@10")
    ax.set_title(f"Retrieval learning curve ({loss})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def output_paths(loss: str) -> tuple[Path, Path, Path, Path, Path]:
    suffix = "" if loss == "logsumexp" else f"_{loss}"
    results_dir = RESULTS / "retrieval" / f"learning_curve{suffix}"
    models_dir = MODELS / f"learning_curve{suffix}"
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    return (
        models_dir,
        results_dir,
        results_dir / "summary.json",
        results_dir / "summary.csv",
        results_dir / "val_ndcg10.png",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Learning curve: subsample train by author, eval on val")
    parser.add_argument("--loss", choices=["logsumexp", "infonce"], default="logsumexp")
    parser.add_argument("--n-negatives", type=int, default=256)
    parser.add_argument("--fractions", nargs="*", type=float, default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    models_root, _, summary_path, csv_path, plot_path = output_paths(args.loss)

    if summary_path.exists() and not args.force:
        print(f"[skip] {summary_path} exists; use --force")
        return 0

    seed_everything(args.seed)
    embeddings, q = load_author_arrays()
    years = load_paper_years()
    train_all = split_examples("train")
    val_examples = split_examples("val")

    val_pool = split_author_set("val")
    val_ds = AuthorExampleDataset(
        val_examples, embeddings, q, years, max_history=args.max_history, author_pool=val_pool
    )
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_examples)
    val_cutoff = HISTORY_CUTOFF
    mean_metrics = evaluate_mean_history_retrieval(
        val_loader, val_examples, embeddings, q, years, val_cutoff, "val", args.max_history
    )
    mean_ndcg10 = float(mean_metrics["ndcg@10"])
    print(f"[baseline] val {MEAN_BASELINE_NAME} ndcg@10={mean_ndcg10:.5f} loss={args.loss}")

    total_authors = len({ex["author_id"] for ex in train_all})
    rows: list[dict[str, Any]] = []
    for fraction in sorted(set(args.fractions)):
        train_subset, n_examples, n_authors = subsample_train_by_author(train_all, fraction, args.seed)
        tag = f"frac_{fraction:.2f}".replace(".", "p")
        ckpt_dir = models_root / tag
        print(
            f"[train] loss={args.loss} fraction={fraction:.2f} authors={n_authors}/{total_authors} "
            f"examples={n_examples}/{len(train_all)}"
        )
        metrics = train_fraction(train_subset, val_examples, embeddings, q, years, ckpt_dir, args)
        row = {
            "loss": args.loss,
            "n_negatives": args.n_negatives if args.loss == "infonce" else 0,
            "train_fraction": fraction,
            "n_train_authors": n_authors,
            "n_train_examples": n_examples,
            "total_train_authors": total_authors,
            "total_train_examples": len(train_all),
            "mean_baseline_ndcg@10": mean_ndcg10,
            **metrics,
        }
        rows.append(row)
        print(
            f"[done] fraction={fraction:.2f} best_epoch={metrics['best_epoch']} "
            f"val_ndcg@10={metrics['best_val_ndcg@10']:.5f} "
            f"val_hit@10={metrics['best_val_hit@10']:.5f}"
        )

    payload = {
        "loss": args.loss,
        "n_negatives": args.n_negatives,
        "fractions": args.fractions,
        "mean_baseline_val": mean_metrics,
        "rows": rows,
    }
    save_json(summary_path, payload)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    plot_learning_curve(rows, mean_ndcg10, plot_path, args.loss)
    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
