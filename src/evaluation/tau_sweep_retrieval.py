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

from evaluation.capacity_sweep_retrieval import CAPACITY_PRESETS, count_parameters, eval_loss
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


def train_one_tau(
    tau: float,
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
    args.tau = tau
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

    cutoff = HISTORY_CUTOFF

    best_val_ndcg = -1.0
    best_epoch = 0
    best_state: dict[str, Any] = {}
    patience_left = args.patience
    train_losses_last: list[float] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = total_losses(
                model, batch, embeddings_t, embeddings, q, years, args.max_history, device,
                tau, "coauthor_infonce", args.n_negatives, author_pool=train_pool,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))
        train_losses_last = train_losses

        val_loss = eval_loss(
            model,
            val_loader,
            embeddings_t,
            embeddings,
            q,
            years,
            args.max_history,
            device,
            tau,
            args.loss,
            args.n_negatives,
            author_pool=val_pool,
        )
        val_retrieval = evaluate_retrieval(
            model, val_retrieval_loader, val_examples, embeddings, q, years, cutoff, device, "val", args.max_history
        )
        score = float(val_retrieval["ndcg@10"])
        if score > best_val_ndcg + 1e-8:
            best_val_ndcg = score
            best_epoch = epoch
            patience_left = args.patience
            test_retrieval = evaluate_retrieval(
                model, test_retrieval_loader, test_examples, embeddings, q, years, cutoff, device, "test", args.max_history
            )
            best_state = {
                "tau": tau,
                "best_epoch": epoch,
                "n_params": n_params,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "val_hit@10": float(val_retrieval["hit@10"]),
                "val_ndcg@10": score,
                "val_mrr@10": float(val_retrieval["mrr@10"]),
                "val_ndcg@50": float(val_retrieval["ndcg@50"]),
                "test_hit@10": float(test_retrieval["hit@10"]),
                "test_ndcg@10": float(test_retrieval["ndcg@10"]),
                "test_mrr@10": float(test_retrieval["mrr@10"]),
            }
            save_checkpoint(
                out_dir / "best.pt",
                model,
                {"epoch": epoch, "metrics": best_state, "model_config": model_cfg.to_dict(), "tau": tau},
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if not best_state:
        val_retrieval = evaluate_retrieval(
            model, val_retrieval_loader, val_examples, embeddings, q, years, cutoff, device, "val", args.max_history
        )
        test_retrieval = evaluate_retrieval(
            model, test_retrieval_loader, test_examples, embeddings, q, years, cutoff, device, "test", args.max_history
        )
        best_state = {
            "tau": tau,
            "best_epoch": args.epochs,
            "n_params": n_params,
            "train_loss": float(np.mean(train_losses_last)) if train_losses_last else 0.0,
            "val_loss": eval_loss(
                model,
                val_loader,
                embeddings_t,
                embeddings,
                q,
                years,
                args.max_history,
                device,
                tau,
                args.loss,
                args.n_negatives,
            ),
            "val_hit@10": float(val_retrieval["hit@10"]),
            "val_ndcg@10": float(val_retrieval["ndcg@10"]),
            "val_mrr@10": float(val_retrieval["mrr@10"]),
            "val_ndcg@50": float(val_retrieval["ndcg@50"]),
            "test_hit@10": float(test_retrieval["hit@10"]),
            "test_ndcg@10": float(test_retrieval["ndcg@10"]),
            "test_mrr@10": float(test_retrieval["mrr@10"]),
        }

    best_state["checkpoint"] = str(out_dir / "best.pt")
    print(
        f"[done] tau={tau:.3f} epoch={best_state['best_epoch']} "
        f"val_ndcg@10={best_state['val_ndcg@10']:.5f} test_ndcg@10={best_state['test_ndcg@10']:.5f}"
    )
    return best_state


def plot_tau(rows: list[dict[str, Any]], mean_val_ndcg: float, out_path: Path) -> None:
    df = pd.DataFrame(rows).sort_values("tau")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["tau"], df["val_ndcg@10"], marker="o", label="val nDCG@10")
    ax.plot(df["tau"], df["test_ndcg@10"], marker="s", label="test nDCG@10")
    ax.axhline(mean_val_ndcg, color="C2", linestyle="--", label=f"mean baseline val ({mean_val_ndcg:.3f})")
    ax.set_xlabel("temperature τ")
    ax.set_ylabel("nDCG@10")
    ax.set_title("InfoNCE temperature sweep")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep contrastive temperature τ")
    parser.add_argument("--taus", nargs="*", type=float, default=[0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3])
    parser.add_argument("--preset", default="medium", choices=list(CAPACITY_PRESETS))
    parser.add_argument("--loss", choices=["logsumexp", "infonce"], default="infonce")
    parser.add_argument("--n-negatives", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--tag", default="", help="Output suffix, e.g. 'low' -> tau_sweep_infonce_low")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.tau = 0.1

    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = RESULTS / "retrieval" / f"tau_sweep_{args.loss}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"
    plot_path = out_dir / "tau_ndcg10.png"

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
    cutoff = HISTORY_CUTOFF
    mean_metrics = evaluate_mean_history_retrieval(
        val_loader, val_examples, embeddings, q, years, cutoff, "val", args.max_history
    )
    mean_val_ndcg = float(mean_metrics["ndcg@10"])
    print(f"[info] preset={args.preset} loss={args.loss} mean_val_ndcg@10={mean_val_ndcg:.5f}")

    hp = CAPACITY_PRESETS[args.preset]
    model_cfg = AuthorTransformerConfig(
        emb_dim=embeddings.shape[1],
        q_dim=q.shape[1],
        max_history=args.max_history,
        dropout=args.dropout,
        **hp,
    )

    rows: list[dict[str, Any]] = []
    for tau in sorted(set(args.taus)):
        tag = f"tau_{tau:.3f}".replace(".", "p")
        ckpt_dir = MODELS / f"tau_sweep_{args.loss}{suffix}" / tag
        print(f"[train] tau={tau:.3f}")
        row = train_one_tau(
            tau, train_examples, val_examples, test_examples, embeddings, q, years, model_cfg, ckpt_dir, args
        )
        row["loss"] = args.loss
        row["preset"] = args.preset
        row["mean_baseline_val_ndcg@10"] = mean_val_ndcg
        rows.append(row)

    payload = {"loss": args.loss, "preset": args.preset, "taus": args.taus, "rows": rows}
    save_json(summary_path, payload)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    plot_tau(rows, mean_val_ndcg, plot_path)
    print(f"[done] wrote {summary_path}")
    print(f"[done] wrote {csv_path}")
    print(f"[done] wrote {plot_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
