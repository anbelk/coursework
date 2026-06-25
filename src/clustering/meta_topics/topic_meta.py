from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from common.compat import META_ALL_VARIANTS, DATA, compact_abstract, load_json, save_json, topic_dir, variant_dir
from clustering.paper_topics.topic_terms import ctfidf, mmr_terms


META_DIR = DATA / "meta"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cluster_documents(meta_docs: list[dict[str, Any]], labels: np.ndarray, k: int) -> list[str]:
    docs = []
    for cluster_id in range(k):
        parts = [
            str(meta_docs[int(idx)]["document"])
            for idx in np.flatnonzero(labels == cluster_id)
        ]
        docs.append("\n\n".join(parts))
    return docs


def representative_fine_clusters(proba: np.ndarray, labels: np.ndarray, k: int, top_n: int) -> dict[str, list[dict[str, Any]]]:
    reps: dict[str, list[dict[str, Any]]] = {}
    for cluster_id in range(k):
        scores = proba[:, cluster_id].copy()
        scores[labels == -1] = -1.0
        order = [int(i) for i in np.argsort(-scores) if scores[int(i)] > 0]
        reps[str(cluster_id)] = [
            {
                "fine_cluster_id": idx,
                "probability": float(scores[idx]),
            }
            for idx in order[:top_n]
        ]
    return reps


def representative_papers(
    fine_reps: dict[str, list[dict[str, Any]]],
    fine_labels: dict[str, dict[str, str]],
    fine_papers: dict[str, list[dict[str, Any]]],
    max_papers: int,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for meta_id, fine_rows in fine_reps.items():
        papers = []
        for fine_row in fine_rows[:4]:
            fine_id = str(fine_row["fine_cluster_id"])
            fine_label = fine_labels.get(fine_id, {}).get("name", f"Fine topic {fine_id}")
            for paper in fine_papers.get(fine_id, [])[:4]:
                papers.append(
                    {
                        "paper_id": paper["paper_id"],
                        "title": paper.get("title", ""),
                        "abstract": compact_abstract(paper.get("abstract", ""), 900),
                        "fine_cluster_id": int(fine_id),
                        "fine_cluster_label": fine_label,
                        "probability": float(fine_row["probability"]),
                    }
                )
                if len(papers) >= max_papers:
                    break
            if len(papers) >= max_papers:
                break
        out[meta_id] = papers
    return out


def selected_variants(names: list[str]) -> list[str]:
    all_names = [variant.name for variant in META_ALL_VARIANTS]
    return all_names if names == ["all"] else names


def run_variant(name: str, args: argparse.Namespace, meta_docs: list[dict[str, Any]], fine_labels: dict, fine_papers: dict) -> None:
    out_dir = topic_dir(name)
    top_terms_path = out_dir / "top_terms.json"
    reps_path = out_dir / "representative_papers.json"
    fine_reps_path = out_dir / "representative_fine_clusters.json"
    if top_terms_path.exists() and reps_path.exists() and fine_reps_path.exists() and not args.force:
        print(f"[skip] {name} meta topics exist; use --force")
        return

    labels = np.load(variant_dir(name) / "labels.npy")
    proba = np.load(variant_dir(name) / "proba.npy")
    k = int(proba.shape[1])
    docs = cluster_documents(meta_docs, labels, k)
    scores, terms = ctfidf(docs, args.min_df, args.max_features)
    top_terms = {
        str(cluster_id): mmr_terms(scores, terms, cluster_id, args.top_candidates, args.top_n, args.mmr_lambda)
        for cluster_id in range(k)
    }
    fine_reps = representative_fine_clusters(proba, labels, k, args.rep_fine_n)
    paper_reps = representative_papers(fine_reps, fine_labels, fine_papers, args.rep_paper_n)

    save_json(top_terms_path, top_terms)
    save_json(fine_reps_path, fine_reps)
    save_json(reps_path, paper_reps)
    print(f"[done] {name}: K={k}, terms={len(terms)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=["all"])
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--rep-fine-n", type=int, default=10)
    parser.add_argument("--rep-paper-n", type=int, default=10)
    parser.add_argument("--top-candidates", type=int, default=30)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    meta_docs = read_jsonl(META_DIR / "cluster_documents.jsonl")
    fine_labels = load_json(topic_dir("hdbscan_fine") / "llm_label.json")
    fine_papers = load_json(topic_dir("hdbscan_fine") / "representative_papers.json")
    for name in tqdm(selected_variants(args.variants), desc="meta topic terms"):
        run_variant(name, args, meta_docs, fine_labels, fine_papers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
