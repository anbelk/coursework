from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from common.compat import MODELS, RESULTS
from recommendation.training_utils import evaluate_model, load_author_arrays, load_paper_years, pick_device


def infer_q_mode(model_name: str) -> str:
    config_path = MODELS / model_name / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing model config: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f).get("q_mode", "fine")


def rows_from_metrics(model_name: str, q_mode: str, split: str, metrics: dict[str, float]) -> list[dict]:
    rows = []
    for k in (10, 50, 100):
        rows.append(
            {
                "method": model_name,
                "artifact": model_name,
                "q_mode": q_mode,
                "split": split,
                "K": k,
                "hit": metrics[f"hit@{k}"],
                "mrr": metrics[f"mrr@{k}"],
                "ndcg": metrics[f"ndcg@{k}"],
                "recall": metrics[f"recall@{k}"],
                "n_examples": int(metrics["n_examples"]),
                "n_with_relevant": int(metrics.get("n_with_relevant", metrics["n_examples"])),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained transformer variants without overwriting legacy metrics")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default=str(RESULTS / "retrieval" / "transformer_variants_metrics.csv"))
    args = parser.parse_args()

    device = pick_device(args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for model_name in args.models:
        q_mode = infer_q_mode(model_name)
        embeddings, q = load_author_arrays(q_mode)
        years = load_paper_years()
        for split in args.splits:
            metrics = evaluate_model(model_name, split, embeddings, q, years, device)
            print(
                f"[metrics] {split} {model_name}: q_mode={q_mode} "
                f"hit@10={metrics['hit@10']:.6f} ndcg@10={metrics['ndcg@10']:.6f}"
            )
            all_rows.extend(rows_from_metrics(model_name, q_mode, split, metrics))

    new = pd.DataFrame(all_rows)
    if out_path.exists():
        old = pd.read_csv(out_path)
        old = old[~old[["method", "split", "K"]].apply(tuple, axis=1).isin(new[["method", "split", "K"]].apply(tuple, axis=1))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out_path, index=False)
    print(f"[done] wrote {out_path}")
    print(new[(new["split"] == "test") & (new["K"] == 10)].to_markdown(index=False, floatfmt=".6f"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
