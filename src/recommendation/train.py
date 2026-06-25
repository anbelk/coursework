from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from recommendation.model import AuthorTransformer, AuthorTransformerConfig
from recommendation.training_utils import (
    COAUTHOR_INFONCE_MODEL_NAME,
    AuthorExampleDataset,
    MultiPositiveBatchSampler,
    collate_examples,
    evaluate_retrieval,
    examples_with_positives,
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
from common.author_splits import HISTORY_CUTOFF, split_author_set
from common.compat import RANDOM_STATE, save_json


LOSS_NAME = "coauthor_infonce"


def free_accelerator_memory() -> None:
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
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
    n_negatives,
    author_pool,
) -> dict[str, float]:
    model.eval()
    losses = []
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
            LOSS_NAME,
            n_negatives,
            author_pool=author_pool,
        )
        losses.append(float(loss.item()))
    return {"val_loss": float(np.mean(losses)) if losses else 0.0}


def train_one(args: argparse.Namespace) -> None:
    out_dir = model_dir(args.model_name)
    if (out_dir / "best.pt").exists() and not args.force:
        print(f"[skip] {args.model_name} best.pt exists; use --force")
        return

    seed_everything(RANDOM_STATE)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    print(f"[info] {args.model_name}: device={device}, loss={LOSS_NAME}")

    embeddings, q = load_author_arrays(args.q_mode)
    years = load_paper_years()
    train_pool = split_author_set("train")
    val_pool = split_author_set("val")
    train_examples = examples_with_positives(split_examples("train"), author_pool=train_pool)
    val_examples = split_examples("val")
    train_ds = AuthorExampleDataset(
        train_examples, embeddings, q, years, max_history=args.max_history, author_pool=train_pool
    )
    val_ds = AuthorExampleDataset(
        val_examples, embeddings, q, years, max_history=args.max_history, author_pool=val_pool
    )
    train_batch_sampler = MultiPositiveBatchSampler(
        train_ds,
        batch_size=args.batch_size,
        extra_positive_rows=args.extra_positive_rows,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_batch_sampler,
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
    # coauthor_infonce does not use a full on-device paper-embedding tensor (it encodes
    # candidates via the model); avoid wasting ~1.5GB of MPS memory on it.
    embeddings_t = torch.zeros(1, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_lambda(step, total_steps, warmup_steps),
    )

    val_cutoff = HISTORY_CUTOFF
    log = []
    best_metric = -1.0
    best_epoch = 0
    patience_left = args.patience
    config_payload = {
        "config_name": args.model_name,
        "loss": LOSS_NAME,
        "loss_type": LOSS_NAME,
        "n_negatives": args.n_negatives,
        "tau": args.tau,
        "optimizer": "AdamW",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "q_mode": args.q_mode,
        "epochs": args.epochs,
        "patience": args.patience,
        "model_config": model_cfg.to_dict(),
        "n_train_examples": len(train_examples),
        "n_val_examples": len(val_examples),
    }
    save_json(out_dir / "config.json", config_payload)

    for epoch in range(1, args.epochs + 1):
        train_batch_sampler.set_epoch(epoch)
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"{args.model_name} epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar):
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
                LOSS_NAME,
                args.n_negatives,
                author_pool=train_pool,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            train_losses.append(float(loss.item()))
            pbar.set_postfix(loss=f"{train_losses[-1]:.4f}")
            # MPS accumulates memory across steps; release periodically to avoid OOM.
            if (step + 1) % 20 == 0:
                free_accelerator_memory()
        free_accelerator_memory()

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
            args.n_negatives,
            val_pool,
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
            max_history=args.max_history,
        )
        free_accelerator_memory()
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **val_losses,
            **{f"val_{k}": v for k, v in retrieval.items() if k != "n_examples"},
            "val_n_examples": int(retrieval["n_examples"]),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        log.append(row)
        save_json(out_dir / "train_log.json", log)
        save_checkpoint(out_dir / "last.pt", model, {"epoch": epoch, "metrics": row, **config_payload})
        save_training_plots(out_dir, log)
        # Select on nDCG@50 by default: it counts more positives than @10, so it is a
        # less noisy early-stopping signal on the tiny coauthor eval set.
        score = float(row[f"val_{args.select_metric}"])
        print(
            f"[epoch] {args.model_name} {epoch}: train={row['train_loss']:.4f} "
            f"val_{args.select_metric}={score:.5f} val_ndcg@10={row['val_ndcg@10']:.5f}"
        )
        if score > best_metric + 1e-8:
            best_metric = score
            best_epoch = epoch
            patience_left = args.patience
            save_checkpoint(out_dir / "best.pt", model, {"epoch": epoch, "metrics": row, **config_payload})
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early-stop] {args.model_name}: best_epoch={best_epoch}, best_val_{args.select_metric}={best_metric:.5f}")
                break
    print(f"[done] {args.model_name}: best_epoch={best_epoch}, best_val_{args.select_metric}={best_metric:.5f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train author retriever (coauthor InfoNCE)")
    parser.add_argument("--model-name", default=COAUTHOR_INFONCE_MODEL_NAME)
    parser.add_argument("--n-negatives", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--extra-positive-rows", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--q-mode", choices=["none", "fine", "metacluster", "fine_metacluster"], default="fine")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--select-metric", default="ndcg@50", help="val metric for early stopping")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    train_one(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
