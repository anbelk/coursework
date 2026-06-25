from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.compat import RESULTS, load_json, save_json, topic_dir, variant_dir


DEFAULT_VARIANTS = [
    ("Baseline", "kmeans_umap_baseline", "fine", "kmeans_umap_baseline_fine"),
    ("BERTopic Style", "bertopic_style", "fine", "bertopic_style_fine"),
    ("Neural Topic Model", "fastopic_neural_topic_model", "fine", "fastopic_fine"),
    ("Baseline", "kmeans_umap_baseline", "medium", "kmeans_umap_baseline_medium"),
    ("BERTopic Style", "bertopic_style", "medium", "bertopic_style_medium"),
    ("Neural Topic Model", "fastopic_neural_topic_model", "medium", "fastopic_medium"),
    ("Metacluster", "bertopic_style_metacluster", "metacluster", "bertopic_style_metacluster"),
]


def paper_level_labels(name: str, base_variant: str) -> np.ndarray:
    if "metacluster" not in name and not name.startswith("meta_"):
        return np.asarray(np.load(variant_dir(name) / "labels.npy", mmap_mode="r"), dtype=np.int32)

    base_labels = np.asarray(np.load(variant_dir(base_variant) / "labels.npy", mmap_mode="r"), dtype=np.int32)
    meta_labels = np.asarray(np.load(variant_dir(name) / "labels.npy", mmap_mode="r"), dtype=np.int32)
    out = np.full((len(base_labels),), -1, dtype=np.int32)
    good = (base_labels >= 0) & (base_labels < len(meta_labels))
    out[good] = meta_labels[base_labels[good]]
    return out


def summarize_sizes(labels: np.ndarray, k: int | None = None) -> tuple[dict[str, Any], np.ndarray]:
    non_noise = labels[labels >= 0]
    if k is None:
        k = int(non_noise.max()) + 1 if len(non_noise) else 0
    sizes = np.bincount(non_noise, minlength=k).astype(np.float64)
    sizes = sizes[sizes > 0]
    total = float(sizes.sum())
    p = sizes / total if total > 0 else np.array([], dtype=np.float64)
    size_entropy = float(-(p * np.log(p)).sum() / np.log(len(sizes))) if len(sizes) > 1 else 0.0
    out = {
        "K_nonempty": int(len(sizes)),
        "n_assigned_papers": int(total),
        "noise_ratio": float((labels < 0).mean()),
        "size_mean": float(sizes.mean()) if len(sizes) else None,
        "size_p25": float(np.percentile(sizes, 25)) if len(sizes) else None,
        "size_p50": float(np.percentile(sizes, 50)) if len(sizes) else None,
        "size_p75": float(np.percentile(sizes, 75)) if len(sizes) else None,
        "size_p90": float(np.percentile(sizes, 90)) if len(sizes) else None,
        "top5_concentration": float(np.sort(sizes)[-5:].sum() / total) if total > 0 else None,
        "size_balance_entropy": size_entropy,
    }
    return out, sizes


def load_optional_metrics(variant: str) -> dict[str, Any]:
    diagnostics = load_json(topic_dir(variant) / "intrinsic_diagnostics.json") if (topic_dir(variant) / "intrinsic_diagnostics.json").exists() else {}
    assigned = load_json(topic_dir(variant) / "assigned_llm_metrics.json") if (topic_dir(variant) / "assigned_llm_metrics.json").exists() else {}
    return {
        "embedding_coherence_lift": diagnostics.get("embedding_coherence_lift"),
        "assignment_confidence_mean": diagnostics.get("assignment_confidence_mean"),
        "llm_assigned_fit@10": assigned.get("llm_assigned_fit@10"),
        "llm_intruder_accuracy@10": assigned.get("llm_intruder_accuracy@10"),
        "llm_sample_clusters": assigned.get("n_clusters_evaluated"),
    }


def build_tables(base_variant: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    dist_rows = []
    for method_class, method, level, variant in DEFAULT_VARIANTS:
        if not (variant_dir(variant) / "labels.npy").exists():
            continue
        labels = paper_level_labels(variant, base_variant)
        k = None
        proba_path = variant_dir(variant) / "proba.npy"
        if proba_path.exists():
            k = int(np.load(proba_path, mmap_mode="r").shape[1])
        summary, sizes = summarize_sizes(labels, k=k)
        summary_rows.append(
            {
                "method_class": method_class,
                "method": method,
                "level": level,
                "variant": variant,
                "base_variant": base_variant if variant.startswith("meta_") else None,
                **summary,
                **load_optional_metrics(variant),
            }
        )
        for size in sizes.tolist():
            dist_rows.append(
                {
                    "method": method,
                    "level": level,
                    "variant": variant,
                    "cluster_size": float(size),
                    "log10_cluster_size": float(np.log10(max(size, 1.0))),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(dist_rows)


def plot_size_distribution(dist: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = []
    data = []
    for (level, method), group in dist.groupby(["level", "method"], sort=False):
        labels.append(f"{level}\n{method}")
        data.append(group["log10_cluster_size"].to_numpy())
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("log10 papers per cluster")
    ax.set_title("Cluster granularity by method")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_granularity_scatter(summary: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"fine": "o", "medium": "s", "metacluster": "^"}
    for _, row in summary.iterrows():
        x = float(row["size_p50"])
        y = float(row["K_nonempty"])
        ax.scatter(x, y, marker=markers.get(row["level"], "o"), s=90)
        ax.annotate(row["variant"], (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("median papers per cluster")
    ax.set_ylabel("number of non-empty clusters")
    ax.set_title("Granularity levels")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build granularity diagnostics for clustering methods")
    parser.add_argument("--base-variant", default="bertopic_style_fine")
    parser.add_argument("--out-dir", default=str(RESULTS / "final_tables"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, dist = build_tables(args.base_variant)
    summary.to_csv(out_dir / "cluster_granularity_summary.csv", index=False)
    dist.to_csv(out_dir / "cluster_granularity_distribution.csv", index=False)
    plot_size_distribution(dist, out_dir / "cluster_granularity_distribution.png")
    plot_granularity_scatter(summary, out_dir / "cluster_granularity_scatter.png")
    save_json(
        out_dir / "cluster_granularity_manifest.json",
        {
            "base_variant_for_metacluster": args.base_variant,
            "outputs": [
                "cluster_granularity_summary.csv",
                "cluster_granularity_distribution.csv",
                "cluster_granularity_distribution.png",
                "cluster_granularity_scatter.png",
            ],
        },
    )
    print(summary.to_markdown(index=False, floatfmt=".6f"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
