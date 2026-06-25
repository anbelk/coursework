from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TOPICS = ROOT / "data" / "topics"
CLUSTERING = ROOT / "data" / "clustering"
OUT_DIR = RESULTS / "final_tables"


CLUSTERING_VARIANTS = [
    {
        "method_class": "Baseline",
        "method": "kmeans_umap_baseline",
        "granularity": "fine",
        "variant": "kmeans_umap_baseline_fine",
        "status": "computed",
    },
    {
        "method_class": "Baseline",
        "method": "kmeans_umap_baseline",
        "granularity": "medium",
        "variant": "kmeans_umap_baseline_medium",
        "status": "computed",
    },
    {
        "method_class": "BERTopic Style",
        "method": "bertopic_style",
        "granularity": "fine",
        "variant": "bertopic_style_fine",
        "status": "computed",
    },
    {
        "method_class": "BERTopic Style",
        "method": "bertopic_style",
        "granularity": "medium",
        "variant": "bertopic_style_medium",
        "status": "computed",
    },
    {
        "method_class": "Neural Topic Model",
        "method": "fastopic_neural_topic_model",
        "granularity": "fine",
        "variant": "fastopic_fine",
        "status": "not_run",
    },
    {
        "method_class": "Neural Topic Model",
        "method": "fastopic_neural_topic_model",
        "granularity": "medium",
        "variant": "fastopic_medium",
        "status": "not_run",
    },
]


META_VARIANTS = [
    {
        "method_class": "Coarse / Metacluster",
        "method": "bertopic_style_metacluster",
        "granularity": "metacluster",
        "variant": "bertopic_style_metacluster",
        "base_variant": "bertopic_style_fine",
        "status": "computed",
    },
]


RECOMMENDATION_ALIASES = {
    "mean_author_embedding": "mean_author_embedding",
    "author_retriever": "transformer_author_fine",
    "graphsage_author": "graphsage_author_no_cluster",
    "graphsage_author_metacluster": "graphsage_author_metacluster",
    "graphsage_transformer_author": "graphsage_transformer_fine",
}

RECOMMENDATION_MAIN_METHODS = [
    ("Baseline", "mean_author_embedding", "mean_author_embedding", "computed"),
    ("Transformer", "transformer_author_no_cluster", "transformer_author_no_cluster", "not_run"),
    ("Transformer", "transformer_author_fine", "author_retriever", "computed_legacy_alias"),
    ("Transformer", "transformer_author_metacluster", "transformer_author_metacluster", "not_run"),
    ("Transformer", "transformer_author_fine_metacluster", "transformer_author_fine_metacluster", "not_run"),
    ("GraphSAGE", "graphsage_author_no_cluster", "graphsage_author", "computed_legacy_alias"),
    ("GraphSAGE", "graphsage_author_fine", None, "not_run"),
    ("GraphSAGE", "graphsage_author_metacluster", "graphsage_author_metacluster", "computed_legacy_alias"),
    ("GraphSAGE", "graphsage_author_fine_metacluster", None, "not_run"),
    ("Semantic + Graph", "graphsage_transformer_fine", "graphsage_transformer_author", "computed_legacy_alias"),
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(path: Path, key: str) -> Any:
    return load_json(path).get(key)


def clustering_row(row: dict[str, str]) -> dict[str, Any]:
    variant = row["variant"]
    quant = load_json(TOPICS / variant / "metrics_quant.json")
    coherence = load_json(TOPICS / variant / "coherence.json")
    distinct = load_json(TOPICS / variant / "distinctness.json")
    diagnostics = load_json(TOPICS / variant / "intrinsic_diagnostics.json")
    assigned = load_json(TOPICS / variant / "assigned_llm_metrics.json")
    artifact_present = (CLUSTERING / variant / "labels.npy").exists()
    status = "computed" if artifact_present else row["status"]
    return {
        **row,
        "status": status,
        "K": quant.get("n_clusters"),
        "llm_assigned_fit@10": assigned.get("llm_assigned_fit@10"),
        "llm_intruder_accuracy@10": assigned.get("llm_intruder_accuracy@10"),
        "assigned_fit_n_clusters": assigned.get("n_clusters_evaluated"),
        "dup_pair_rate": distinct.get("duplicate_pair_rate"),
        "embedding_coherence_lift": diagnostics.get("embedding_coherence_lift"),
        "embedding_coherence_lift_weighted": diagnostics.get("embedding_coherence_lift_weighted"),
        "assignment_confidence_mean": diagnostics.get("assignment_confidence_mean"),
        "assignment_entropy_norm": quant.get("mean_assignment_entropy_norm"),
        "noise_ratio": quant.get("noise_ratio"),
        "size_balance_entropy": diagnostics.get("size_balance_entropy"),
        "size_p50": quant.get("size_p50"),
        "top5_concentration": quant.get("top5_concentration"),
    }


def build_clustering_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    clustering = pd.DataFrame([clustering_row(row) for row in CLUSTERING_VARIANTS])
    meta = pd.DataFrame([clustering_row(row) for row in META_VARIANTS])
    return clustering, meta


def build_recommendation_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_path = RESULTS / "retrieval" / "consistency" / "retrieval_metrics_at_k.csv"
    transformer_path = RESULTS / "retrieval" / "transformer_variants_metrics.csv"
    llm_path = RESULTS / "retrieval" / "consistency" / "llm_threshold_metrics_at_k.csv"
    observed = pd.read_csv(observed_path)
    if transformer_path.exists():
        transformer = pd.read_csv(transformer_path)
        if "recall" not in transformer.columns:
            transformer["recall"] = None
        transformer["method"] = transformer["method"].astype(str)
        observed = pd.concat([observed, transformer[observed.columns]], ignore_index=True)
    observed = observed[(observed["split"] == "test") & (observed["K"] == 10)].copy()
    observed["canonical_method"] = observed["method"].map(RECOMMENDATION_ALIASES).fillna(observed["method"])
    observed_rows = []
    for method_class, canonical, legacy, status in RECOMMENDATION_MAIN_METHODS:
        match = observed[observed["method"] == legacy] if legacy else pd.DataFrame()
        if match.empty:
            observed_rows.append(
                {
                    "method_class": method_class,
                    "method": canonical,
                    "artifact": legacy,
                    "status": status,
                    "K": 10,
                    "hit": None,
                    "mrr": None,
                    "ndcg": None,
                    "recall": None,
                    "n_examples": None,
                    "n_with_relevant": None,
                }
            )
        else:
            row = match.iloc[0]
            computed_status = status if status != "not_run" else "computed"
            observed_rows.append(
                {
                    "method_class": method_class,
                    "method": canonical,
                    "artifact": legacy,
                    "status": computed_status,
                    "K": 10,
                    "hit": row["hit"],
                    "mrr": row["mrr"],
                    "ndcg": row["ndcg"],
                    "recall": row["recall"],
                    "n_examples": row["n_examples"],
                    "n_with_relevant": row["n_with_relevant"],
                }
            )
    observed = pd.DataFrame(observed_rows)

    llm = pd.read_csv(llm_path)
    llm = llm[llm["K"] == 10].copy()
    llm["canonical_method"] = llm["method"].map(RECOMMENDATION_ALIASES).fillna(llm["method"])
    llm_rows = []
    for method_class, canonical, legacy, status in RECOMMENDATION_MAIN_METHODS:
        match = llm[llm["method"] == legacy] if legacy else pd.DataFrame()
        if match.empty:
            llm_rows.append(
                {
                    "method_class": method_class,
                    "method": canonical,
                    "artifact": legacy,
                    "status": "llm_not_run" if status != "not_run" else status,
                    "K": 10,
                    "n_users": None,
                    "llm_has_3_ge2": None,
                    "llm_selected3_mean_score": None,
                    "llm_hit_ge3": None,
                    "llm_mrr_ge3": None,
                    "llm_has_3_ge3": None,
                    "llm_relevant_rate_ge2": None,
                }
            )
        else:
            row = match.iloc[0]
            llm_rows.append(
                {
                    "method_class": method_class,
                    "method": canonical,
                    "artifact": legacy,
                    "status": status,
                    "K": 10,
                    "n_users": row["n_users"],
                    "llm_has_3_ge2": row["llm_has_3_ge2"],
                    "llm_selected3_mean_score": row["llm_selected3_mean_score"],
                    "llm_hit_ge3": row["llm_hit_ge3"],
                    "llm_mrr_ge3": row["llm_mrr_ge3"],
                    "llm_has_3_ge3": row["llm_has_3_ge3"],
                    "llm_relevant_rate_ge2": row["llm_relevant_rate_ge2"],
                }
            )
    llm = pd.DataFrame(llm_rows)
    return observed, llm


def write_markdown(
    clustering: pd.DataFrame,
    meta: pd.DataFrame,
    observed: pd.DataFrame,
    llm: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sections = [
        "# Final Experiment Tables",
        "",
        "## Clustering intrinsic metrics",
        "",
        clustering.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Metacluster metrics",
        "",
        meta.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Recommendation observed future-coauthor metrics, K=10",
        "",
        observed.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Recommendation LLM metrics, K=10",
        "",
        llm.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Notes:",
        "- Clustering LLM metrics use random assigned papers, not representative core papers.",
        "- Metacluster is treated as the coarse level built over `hdbscan_fine`.",
        "- FASTopic rows are filled when `data/clustering/fastopic_*` artifacts and topic metrics exist.",
        "- Recommendation rows with `not_run` are fixed-design methods that still require training/evaluation.",
    ]
    (OUT_DIR / "tables.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clustering, meta = build_clustering_tables()
    observed, llm = build_recommendation_tables()
    clustering.to_csv(OUT_DIR / "clustering_intrinsic.csv", index=False)
    meta.to_csv(OUT_DIR / "clustering_metacluster.csv", index=False)
    observed.to_csv(OUT_DIR / "recommendation_observed_k10.csv", index=False)
    llm.to_csv(OUT_DIR / "recommendation_llm_k10.csv", index=False)
    write_markdown(clustering, meta, observed, llm)
    print(f"[done] wrote {OUT_DIR}")
    print("\n## clustering_intrinsic")
    print(clustering.to_markdown(index=False, floatfmt=".6f"))
    print("\n## clustering_metacluster")
    print(meta.to_markdown(index=False, floatfmt=".6f"))
    print("\n## recommendation_observed_k10")
    print(observed.to_markdown(index=False, floatfmt=".6f"))
    print("\n## recommendation_llm_k10")
    print(llm.to_markdown(index=False, floatfmt=".6f"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
