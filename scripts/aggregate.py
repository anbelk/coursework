from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline_common import ALL_VARIANTS, RESULTS, load_json, minmax_score, save_json, topic_dir


def load_variant_row(name: str) -> dict:
    coherence = load_json(topic_dir(name) / "coherence.json")
    distinct = load_json(topic_dir(name) / "distinctness.json")
    quant = load_json(topic_dir(name) / "metrics_quant.json")
    return {
        "variant": name,
        "K": int(quant["n_clusters"]),
        "coherence_weighted": float(coherence["weighted_mean"]) if coherence.get("weighted_mean") is not None else np.nan,
        "coherence_unweighted": float(coherence["unweighted_mean"]) if coherence.get("unweighted_mean") is not None else np.nan,
        "dup_pair_rate": float(distinct["duplicate_pair_rate"]) if distinct.get("duplicate_pair_rate") is not None else np.nan,
        "n_pairs_checked": int(distinct["n_pairs"]),
        "mean_entropy_norm": float(quant["mean_assignment_entropy_norm"]),
        "size_p25": float(quant["size_p25"]),
        "size_p50": float(quant["size_p50"]),
        "size_p75": float(quant["size_p75"]),
        "top5_concentration": float(quant["top5_concentration"]),
        "noise_ratio": float(quant["noise_ratio"]),
    }


def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dup = out["dup_pair_rate"].fillna(0.0).tolist()
    out["score_coherence"] = minmax_score(out["coherence_weighted"].tolist(), True)
    out["score_distinct"] = minmax_score((1.0 - np.array(dup)).tolist(), True)
    out["score_top5"] = minmax_score((1.0 - out["top5_concentration"]).tolist(), True)
    out["score_noise"] = minmax_score((1.0 - out["noise_ratio"]).tolist(), True)
    out["score_entropy"] = minmax_score((1.0 - out["mean_entropy_norm"]).tolist(), True)
    out["passes_guards"] = (
        (out["size_p50"] >= 5)
        & (out["top5_concentration"] <= 0.6)
        & (out["noise_ratio"] <= 0.5)
    )
    out["composite_score"] = (
        0.40 * out["score_coherence"]
        + 0.25 * out["score_distinct"]
        + 0.15 * out["score_top5"]
        + 0.10 * out["score_noise"]
        + 0.10 * out["score_entropy"]
    )
    out.loc[~out["passes_guards"], "composite_score"] = -np.inf
    return out.sort_values("composite_score", ascending=False)


def make_size_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.25, df["size_p25"], width=0.25, label="P25")
    ax.bar(x, df["size_p50"], width=0.25, label="P50")
    ax.bar(x + 0.25, df["size_p75"], width=0.25, label="P75")
    ax.set_xticks(x)
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
        "composite_score",
        "passes_guards",
    ]
    report = [
        "# BERTopic Clustering Selection Report",
        "",
        f"Best variant: `{winner}`" if winner else "Best variant: none (all variants failed hard guards)",
        "",
        f"Size distribution plot: `{plot_path.relative_to(out_path.parent)}`",
        "",
        df[display_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Composite score weights: coherence 0.40, distinctness 0.25, top-5 balance 0.15, noise 0.10, entropy 0.10.",
        "Hard guards: size_p50 >= 5, top5_concentration <= 0.6, noise_ratio <= 0.5.",
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
    parser.add_argument("--variants", nargs="*", default=[v.name for v in ALL_VARIANTS])
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = [load_variant_row(name) for name in args.variants]
    df = compute_scores(pd.DataFrame(rows))
    summary_path = RESULTS / "summary_metrics.csv"
    df.replace([np.inf, -np.inf], np.nan).to_csv(summary_path, index=False)
    plot_path = RESULTS / "size_distribution.png"
    make_size_plot(df, plot_path)
    winner = df.loc[df["passes_guards"], "variant"].iloc[0] if df["passes_guards"].any() else None
    write_report(df.replace([np.inf, -np.inf], np.nan), winner, RESULTS / "selection_report.md", plot_path)
    if winner:
        refresh_winner_links(str(winner))
    print(f"[done] wrote {summary_path}")
    print(f"[done] winner={winner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
