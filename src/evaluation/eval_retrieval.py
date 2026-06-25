from __future__ import annotations

import argparse
import sys

import pandas as pd
from torch.utils.data import DataLoader

from common.compat import RESULTS
from common.author_splits import HISTORY_CUTOFF, split_author_set
from evaluation.coauthor_retrieval import cutoff_year_for_split
from recommendation.training_utils import (
    MEAN_BASELINE_NAME,
    COAUTHOR_INFONCE_MODEL_NAME,
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


def infer_q_mode(model_name: str | None) -> str:
    if not model_name:
        return "fine"
    path = RESULTS.parent / "models" / model_name / "config.json"
    if path.exists():
        import json

        with path.open("r", encoding="utf-8") as f:
            return json.load(f).get("q_mode", "fine")
    return "fine"


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
                "n_with_relevant": int(metrics.get("n_with_relevant", metrics["n_examples"])),
            }
        )
    return rows


def evaluate_baselines(
    split: str,
    embeddings,
    q,
    years,
    max_history: int = 20,
) -> list[dict]:
    examples = split_examples(split)
    author_pool = split_author_set(split)
    ds = AuthorExampleDataset(
        examples, embeddings, q, years, max_history=max_history, author_pool=author_pool
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_examples, num_workers=0)
    cutoff_year = cutoff_year_for_split(split)
    rows = []
    mean_metrics = evaluate_mean_history_retrieval(
        loader, examples, embeddings, q, years, cutoff_year, split, max_history
    )
    print(f"[metrics] {split} {MEAN_BASELINE_NAME}: {mean_metrics}")
    rows.extend(rows_from_metrics(MEAN_BASELINE_NAME, mean_metrics))
    random_metrics = evaluate_random_retrieval(len(examples), examples, cutoff_year, split, max_history)
    print(f"[metrics] {split} random: {random_metrics}")
    rows.extend(rows_from_metrics("random", random_metrics))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=COAUTHOR_INFONCE_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--q-mode", choices=["none", "fine", "metacluster", "fine_metacluster"], default=None)
    parser.add_argument("--baselines-only", action="store_true")
    args = parser.parse_args()

    out_dir = RESULTS / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    q_mode = args.q_mode if args.q_mode is not None else infer_q_mode(args.model)
    embeddings, q = load_author_arrays(q_mode)
    years = load_paper_years()

    # Transformer is evaluated on val + test; fixed baselines (mean, random) only on
    # test, where the final transformer-vs-baseline comparison happens. (Train-split
    # retrieval would build a 37k x 86k score matrix -> OOM; the overfitting check is
    # done separately on a capped subset.)
    for split in ("val", "test"):
        all_rows: list[dict] = []
        if not args.baselines_only:
            metrics = evaluate_model(args.model, split, embeddings, q, years, device)
            print(f"[metrics] {split} {args.model}: ndcg@10={metrics['ndcg@10']:.5f} hit@10={metrics['hit@10']:.5f}")
            all_rows.extend(rows_from_metrics(args.model, metrics))
        if split == "test":
            all_rows.extend(evaluate_baselines(split, embeddings, q, years, args.max_history))
        out_path = out_dir / f"{split}_metrics.csv"
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        print(f"[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
