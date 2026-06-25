from __future__ import annotations

import argparse
import sys

import pandas as pd
from torch.utils.data import DataLoader

from common.compat import RESULTS, save_json
from common.author_splits import split_author_set
from evaluation.coauthor_retrieval import cutoff_year_for_split
from recommendation.training_utils import (
    COAUTHOR_INFONCE_MODEL_NAME,
    MEAN_BASELINE_NAME,
    AuthorExampleDataset,
    collate_examples,
    evaluate_mean_history_retrieval,
    evaluate_model,
    evaluate_random_retrieval,
    load_author_arrays,
    load_paper_years,
    pick_device,
    split_examples,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare retrieval metrics across loss variants")
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[COAUTHOR_INFONCE_MODEL_NAME, MEAN_BASELINE_NAME, "random"],
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    out_dir = RESULTS / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    embeddings, q = load_author_arrays()
    years = load_paper_years()
    splits = ["val", "test"] if args.split == "both" else [args.split]
    summary: dict[str, list[dict]] = {}

    for split in splits:
        all_rows = []
        examples = split_examples(split)
        author_pool = split_author_set(split)
        ds = AuthorExampleDataset(examples, embeddings, q, years, author_pool=author_pool)
        loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_examples, num_workers=0)
        cutoff_year = cutoff_year_for_split(split)

        for model_name in args.models:
            if model_name == MEAN_BASELINE_NAME:
                metrics = evaluate_mean_history_retrieval(
                    loader, examples, embeddings, q, years, cutoff_year, split
                )
            elif model_name == "random":
                metrics = evaluate_random_retrieval(len(examples), examples, cutoff_year, split)
            else:
                metrics = evaluate_model(model_name, split, embeddings, q, years, device)
            print(f"[metrics] {split} {model_name}: ndcg@10={metrics['ndcg@10']:.5f} hit@10={metrics['hit@10']:.5f}")
            all_rows.extend(rows_from_metrics(model_name, metrics))

        out_path = out_dir / f"{split}_loss_comparison.csv"
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        summary[split] = all_rows
        print(f"[done] wrote {out_path}")

    save_json(out_dir / "loss_comparison_summary.json", summary)
    print(f"[done] wrote {out_dir / 'loss_comparison_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
