from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from author_model_utils import (
    CONFIGS,
    AuthorExampleDataset,
    collate_examples,
    evaluate_retrieval,
    load_author_arrays,
    load_paper_years,
    model_dir,
    move_batch,
    pick_device,
    save_checkpoint,
    save_training_plots,
    seed_everything,
    split_examples,
    total_losses,
    warmup_cosine_lambda,
)
from model import AuthorTransformer, AuthorTransformerConfig
from pipeline_common import RANDOM_STATE, save_json


@torch.inference_mode()
def eval_loss(model, loader, embeddings_t, device, lambda_cluster, lambda_emb, tau) -> dict[str, float]:
    model.eval()
    losses, lcs, les = [], [], []
    for batch in loader:
        batch = move_batch(batch, device)
        loss, lc, le, _ = total_losses(model, batch, embeddings_t, lambda_cluster, lambda_emb, tau)
        losses.append(float(loss.item()))
        lcs.append(float(lc.item()))
        les.append(float(le.item()))
    return {
        "val_loss": float(np.mean(losses)) if losses else 0.0,
        "val_L_cluster": float(np.mean(lcs)) if lcs else 0.0,
        "val_L_emb": float(np.mean(les)) if les else 0.0,
    }


def train_one(config_name: str, args: argparse.Namespace) -> None:
    if config_name not in CONFIGS:
        raise ValueError(f"unknown config: {config_name}")
    out_dir = model_dir(config_name)
    if (out_dir / "best.pt").exists() and not args.force:
        print(f"[skip] {config_name} best.pt exists; use --force")
        return

    seed_everything(RANDOM_STATE)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"[info] {config_name}: device={device}")

    embeddings, q = load_author_arrays()
    years = load_paper_years()
    train_examples = split_examples("train")
    val_examples = split_examples("val")
    train_ds = AuthorExampleDataset(train_examples, embeddings, q, years, max_history=args.max_history)
    val_ds = AuthorExampleDataset(val_examples, embeddings, q, years, max_history=args.max_history)
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
    lambdas = CONFIGS[config_name]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_lambda(step, total_steps, warmup_steps),
    )

    paper_years = np.array(years)
    val_candidates = np.flatnonzero(paper_years == 2025).astype(np.int64)
    log = []
    best_metric = -1.0
    best_epoch = 0
    patience_left = args.patience
    config_payload = {
        "config_name": config_name,
        **lambdas,
        "tau": args.tau,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "model_config": model_cfg.to_dict(),
        "n_train_examples": len(train_examples),
        "n_val_examples": len(val_examples),
    }
    save_json(out_dir / "config.json", config_payload)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses, train_lcs, train_les = [], [], []
        pbar = tqdm(train_loader, desc=f"{config_name} epoch {epoch}", leave=False)
        for batch in pbar:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, lc, le, _ = total_losses(
                model,
                batch,
                embeddings_t,
                lambdas["lambda_cluster"],
                lambdas["lambda_emb"],
                args.tau,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))
            train_lcs.append(float(lc.item()))
            train_les.append(float(le.item()))
            pbar.set_postfix(loss=f"{train_losses[-1]:.4f}")

        val_losses = eval_loss(
            model,
            val_loader,
            embeddings_t,
            device,
            lambdas["lambda_cluster"],
            lambdas["lambda_emb"],
            args.tau,
        )
        retrieval = evaluate_retrieval(
            model,
            val_retrieval_loader,
            val_examples,
            embeddings,
            val_candidates,
            device,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_L_cluster": float(np.mean(train_lcs)),
            "train_L_emb": float(np.mean(train_les)),
            **val_losses,
            **{f"val_{k}": v for k, v in retrieval.items() if k != "n_examples"},
            "val_n_examples": int(retrieval["n_examples"]),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        # aliases used by plot helper
        row["val_ndcg@10"] = row["val_ndcg@10"]
        log.append(row)
        save_json(out_dir / "train_log.json", log)
        save_checkpoint(out_dir / "last.pt", model, {"epoch": epoch, "metrics": row, **config_payload})
        save_training_plots(out_dir, log)
        score = float(row["val_ndcg@10"])
        print(
            f"[epoch] {config_name} {epoch}: train={row['train_loss']:.4f} "
            f"val_ndcg@10={score:.5f} val_mrr@10={row['val_mrr@10']:.5f}"
        )
        if score > best_metric + 1e-8:
            best_metric = score
            best_epoch = epoch
            patience_left = args.patience
            save_checkpoint(out_dir / "best.pt", model, {"epoch": epoch, "metrics": row, **config_payload})
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early-stop] {config_name}: best_epoch={best_epoch}, best_val_ndcg@10={best_metric:.5f}")
                break
    print(f"[done] {config_name}: best_epoch={best_epoch}, best_val_ndcg@10={best_metric:.5f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="all", choices=["all", *CONFIGS.keys()])
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
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    names = list(CONFIGS) if args.config == "all" else [args.config]
    for name in names:
        train_one(name, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
