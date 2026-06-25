from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.compat import (
    ALL_CLUSTERING_EVAL_VARIANTS,
    BASELINE_VARIANT_NAMES,
    RESULTS,
    load_json,
    save_json,
    topic_dir,
)


def load_variant_row(name: str) -> dict:
    coherence = load_json(topic_dir(name) / "coherence.json")
    distinct = load_json(topic_dir(name) / "distinctness.json")
    quant = load_json(topic_dir(name) / "metrics_quant.json")
    return {
        "variant": name,
        "K": int(quant["n_clusters"]),
        "coherence_weighted": float(coherence["weighted_mean"]) if coherence.get("weighted_mean") is not None else None,
        "coherence_unweighted": float(coherence["unweighted_mean"]) if coherence.get("unweighted_mean") is not None else None,
        "dup_pair_rate": float(distinct["duplicate_pair_rate"]) if distinct.get("duplicate_pair_rate") is not None else None,
        "n_pairs_checked": int(distinct["n_pairs"]),
        "mean_entropy_norm": float(quant["mean_assignment_entropy_norm"]),
        "size_p25": float(quant["size_p25"]),
        "size_p50": float(quant["size_p50"]),
        "size_p75": float(quant["size_p75"]),
        "top5_concentration": float(quant["top5_concentration"]),
        "noise_ratio": float(quant["noise_ratio"]),
    }


def rank_variants(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        ["coherence_weighted", "dup_pair_rate"],
        ascending=[False, True],
        na_position="last",
    )


def pick_winner(df: pd.DataFrame) -> str | None:
    eligible = df[~df["variant"].isin(BASELINE_VARIANT_NAMES)]
    if eligible.empty:
        return None
    return str(eligible.iloc[0]["variant"])


def make_size_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(df))
    ax.bar([i - 0.25 for i in x], df["size_p25"], width=0.25, label="P25")
    ax.bar(list(x), df["size_p50"], width=0.25, label="P50")
    ax.bar([i + 0.25 for i in x], df["size_p75"], width=0.25, label="P75")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["variant"], rotation=45, ha="right")
    ax.set_ylabel("papers per cluster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(df: pd.DataFrame, winner: str | None, out_path: Path, plot_path: Path) -> None:
    display_cols = [
        "variant",
        "K",
        "coherence_weighted",
        "dup_pair_rate",
        "mean_entropy_norm",
        "size_p50",
        "top5_concentration",
        "noise_ratio",
    ]
    report = [
        "# BERTopic Clustering Selection Report",
        "",
        f"Best variant: `{winner}`" if winner else "Best variant: none",
        "",
        f"Size distribution plot: `{plot_path.relative_to(out_path.parent)}`",
        "",
        df[display_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Winner selection: highest `coherence_weighted` (tie-break: lower `dup_pair_rate`).",
        "Baselines (`random_umap10`, `kmeans_umap10`) are included for comparison but excluded from winner selection.",
    ]
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def refresh_winner_links(winner: str) -> None:
    winner_dir = RESULTS / "winner"
    winner_dir.mkdir(parents=True, exist_ok=True)
    for child in winner_dir.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    targets = {
        "clustering": Path("../../data/clustering") / winner,
        "topics": Path("../../data/topics") / winner,
    }
    for name, target in targets.items():
        os.symlink(target, winner_dir / name)
    save_json(winner_dir / "winner.json", {"variant": winner})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=[v.name for v in ALL_CLUSTERING_EVAL_VARIANTS])
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = [load_variant_row(name) for name in args.variants]
    df = rank_variants(pd.DataFrame(rows))
    summary_path = RESULTS / "summary_metrics.csv"
    df.to_csv(summary_path, index=False)
    plot_path = RESULTS / "size_distribution.png"
    make_size_plot(df, plot_path)
    winner = pick_winner(df)
    write_report(df, winner, RESULTS / "selection_report.md", plot_path)
    if winner:
        refresh_winner_links(str(winner))
    print(f"[done] wrote {summary_path}")
    print(f"[done] winner={winner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
