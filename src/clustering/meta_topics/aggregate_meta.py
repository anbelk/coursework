from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.compat import DATA, META_ALL_VARIANTS, RESULTS, is_meta_baseline, load_json, save_json, topic_dir, variant_dir


META_DIR = DATA / "meta"


def load_variant_row(name: str) -> dict:
    coherence = load_json(topic_dir(name) / "coherence.json")
    distinct = load_json(topic_dir(name) / "distinctness.json")
    quant = load_json(topic_dir(name) / "metrics_quant.json")
    return {
        "variant": name,
        "K": int(quant["n_clusters"]),
        "coherence_proxy": bool(coherence.get("proxy", False)),
        "distinctness_proxy": bool(distinct.get("proxy", False)),
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


def rank_variants(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["metrics_are_real"] = ~(out["coherence_proxy"] | out["distinctness_proxy"])
    return out.sort_values(
        ["coherence_weighted", "dup_pair_rate"],
        ascending=[False, True],
        na_position="last",
    )


def pick_winner(df: pd.DataFrame, allow_proxy: bool) -> str | None:
    eligible = df[~df["variant"].map(is_meta_baseline)]
    eligible = eligible if allow_proxy else eligible[eligible["metrics_are_real"]]
    if eligible.empty:
        return None
    return str(eligible.iloc[0]["variant"])


def make_size_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    ax.bar(x - 0.25, df["size_p25"], width=0.25, label="P25")
    ax.bar(x, df["size_p50"], width=0.25, label="P50")
    ax.bar(x + 0.25, df["size_p75"], width=0.25, label="P75")
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"], rotation=45, ha="right")
    ax.set_ylabel("fine topics per meta-cluster")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_report(df: pd.DataFrame, winner: str | None, out_path: Path, plot_path: Path) -> None:
    display_cols = [
        "variant",
        "K",
        "coherence_weighted",
        "coherence_proxy",
        "dup_pair_rate",
        "distinctness_proxy",
        "mean_entropy_norm",
        "size_p50",
        "top5_concentration",
        "noise_ratio",
        "metrics_are_real",
    ]
    report = [
        "# Meta-Clustering Selection Report",
        "",
        f"Best variant: `{winner}`" if winner else "Best variant: none",
        "",
        f"Size distribution plot: `{plot_path.relative_to(out_path.parent)}`",
        "",
        df[display_cols].to_markdown(index=False, floatfmt=".4f"),
        "",
        "Winner selection: highest `coherence_weighted` (tie-break: lower `dup_pair_rate`).",
        "Proxy LLM metrics are excluded unless aggregate_meta.py is run with --allow-proxy.",
    ]
    out_path.write_text("\n".join(report) + "\n", encoding="utf-8")


def refresh_winner_links(winner: str) -> None:
    winner_dir = RESULTS / "meta_winner"
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


def clear_winner_links() -> None:
    winner_dir = RESULTS / "meta_winner"
    winner_dir.mkdir(parents=True, exist_ok=True)
    for child in winner_dir.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
    save_json(winner_dir / "winner.json", {"variant": None})


def write_manifest(out_dir: Path, df: pd.DataFrame, winner: str | None, allow_proxy: bool) -> None:
    save_json(
        out_dir / "manifest.json",
        {
            "kind": "meta_clustering_selection",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "allow_proxy": allow_proxy,
            "winner": winner,
            "variants": df.to_dict(orient="records"),
        },
    )


def write_assignments(winner: str, allow_proxy: bool) -> None:
    labels = np.load(variant_dir(winner) / "labels.npy")
    proba = np.load(variant_dir(winner) / "proba.npy")
    hard = labels.copy()
    hard[hard < 0] = np.argmax(proba[hard < 0], axis=1) if (hard < 0).any() else hard[hard < 0]
    meta_labels = load_json(topic_dir(winner) / "llm_label.json")
    fine_sizes = load_json(variant_dir("hdbscan_fine") / "sizes.json")
    meta_paper_count: dict[str, int] = {}
    for fine_id, meta_id in enumerate(hard):
        key = str(int(meta_id))
        meta_paper_count[key] = meta_paper_count.get(key, 0) + int(fine_sizes.get(str(fine_id), 0))
    save_json(
        META_DIR / "meta_assignments.json",
        {
            "winner": winner,
            "allow_proxy_selection": allow_proxy,
            "fine_to_meta": {str(i): int(v) for i, v in enumerate(hard.tolist())},
            "meta_label": meta_labels,
            "meta_paper_count": meta_paper_count,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=[v.name for v in META_ALL_VARIANTS])
    parser.add_argument("--allow-proxy", action="store_true")
    args = parser.parse_args()

    out_dir = RESULTS / "meta_clustering"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [load_variant_row(name) for name in args.variants]
    df = rank_variants(pd.DataFrame(rows))
    summary_path = out_dir / "summary_metrics.csv"
    df.to_csv(summary_path, index=False)
    plot_path = out_dir / "size_distribution.png"
    make_size_plot(df, plot_path)
    winner = pick_winner(df, args.allow_proxy)
    write_report(df, winner, out_dir / "selection_report.md", plot_path)
    write_manifest(out_dir, df, winner, args.allow_proxy)
    if winner:
        refresh_winner_links(str(winner))
        write_assignments(str(winner), args.allow_proxy)
    else:
        clear_winner_links()
    print(f"[done] wrote {summary_path}")
    print(f"[done] winner={winner}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
