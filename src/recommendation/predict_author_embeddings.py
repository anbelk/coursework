from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from recommendation.training_utils import (
    AuthorExampleDataset,
    MODEL_NAME,
    collate_examples,
    load_author_arrays,
    load_model_checkpoint,
    load_paper_years,
    model_dir,
    pick_device,
    predict_embeddings,
)
from common.compat import AUTHORS, PREDICTIONS, load_json, save_json


def build_inference_examples(cutoff_year: int, max_history: int) -> tuple[list[dict], dict]:
    author_index = load_json(AUTHORS / "author_index.json")
    examples = []
    history_index = {}
    for author in author_index:
        hist = [
            int(idx)
            for idx, year in zip(author["paper_idxs"], author["years"], strict=False)
            if int(year) <= cutoff_year
        ]
        if len(hist) < 10:
            continue
        examples.append(
            {
                "author_id": author["author_id"],
                "cutoff_year": cutoff_year,
                "history_paper_idxs": hist[-max_history:],
                "future_paper_idxs": [hist[-1]],  # unused placeholder for dataset collation
            }
        )
        history_index[author["author_id"]] = {
            "name": author.get("name", ""),
            "history_paper_idxs": hist[-max_history:],
            "last5_paper_idxs": hist[-5:],
        }
    return examples, history_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--cutoff-year", type=int, default=2024)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--q-mode", choices=["none", "fine", "metacluster", "fine_metacluster"], default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    if (
        (PREDICTIONS / "author_pred_emb.npy").exists()
        and (PREDICTIONS / "author_ids.json").exists()
        and not args.force
    ):
        print("[skip] author predictions exist; use --force")
        return 0

    model_name = args.model
    q_mode = args.q_mode
    if q_mode is None:
        config_path = model_dir(model_name) / "config.json"
        q_mode = load_json(config_path).get("q_mode", "fine") if config_path.exists() else "fine"
    device = pick_device(args.device)
    embeddings, q = load_author_arrays(q_mode)
    years = load_paper_years()
    examples, history_index = build_inference_examples(args.cutoff_year, args.max_history)
    ds = AuthorExampleDataset(examples, embeddings, q, years, max_history=args.max_history)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_examples, num_workers=0)
    model, payload = load_model_checkpoint(model_dir(model_name) / "best.pt", device)
    pred, _ = predict_embeddings(model, loader, device)
    pred = pred.astype(np.float32)
    np.save(PREDICTIONS / "author_pred_emb.npy", pred)
    author_ids = [ex["author_id"] for ex in examples]
    save_json(PREDICTIONS / "author_ids.json", author_ids)
    save_json(PREDICTIONS / "author_history_index.json", history_index)
    save_json(
        PREDICTIONS / "prediction_meta.json",
        {
            "model": model_name,
            "q_mode": q_mode,
            "checkpoint_epoch": payload.get("epoch"),
            "cutoff_year": args.cutoff_year,
            "n_authors": len(author_ids),
            "shape": list(pred.shape),
        },
    )
    print(f"[done] wrote predictions for {len(author_ids)} authors using {model_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
