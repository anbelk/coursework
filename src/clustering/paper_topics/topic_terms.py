from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm

from common.compat import (
    ALL_VARIANTS,
    compact_abstract,
    load_papers_in_embedding_order,
    representative_indices,
    save_json,
    topic_dir,
    variant_dir,
)


def cluster_documents(papers: list[dict], labels: np.ndarray, k: int) -> list[str]:
    docs: list[str] = []
    for cluster_id in range(k):
        idxs = np.flatnonzero(labels == cluster_id)
        parts = []
        for i in idxs:
            p = papers[int(i)]
            parts.append(f"{p.get('title', '')}. {p.get('abstract', '')}")
        docs.append("\n".join(parts))
    return docs


def ctfidf(docs: list[str], min_df: int, max_features: int | None) -> tuple[np.ndarray, list[str]]:
    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b",
    )
    counts = vectorizer.fit_transform(docs).astype(np.float64)
    row_sums = np.asarray(counts.sum(axis=1)).ravel()
    avg_words = float(row_sums.mean()) if len(row_sums) else 1.0
    tf = counts.multiply(1.0 / np.maximum(row_sums, 1.0)[:, None])
    term_totals = np.asarray(counts.sum(axis=0)).ravel()
    idf = np.log1p(avg_words / np.maximum(term_totals, 1.0))
    scores = tf.multiply(idf).toarray()
    return scores, vectorizer.get_feature_names_out().tolist()


def mmr_terms(
    scores: np.ndarray,
    terms: list[str],
    cluster_id: int,
    top_candidates: int,
    top_n: int,
    lambda_: float,
) -> list[dict]:
    relevance = scores[cluster_id]
    candidate_idx = np.argsort(-relevance)[:top_candidates]
    candidate_idx = [int(i) for i in candidate_idx if relevance[i] > 0]
    if not candidate_idx:
        return []

    profiles = normalize(scores[:, candidate_idx].T, norm="l2", axis=1)
    profile_sim = profiles @ profiles.T
    selected: list[int] = []
    available = list(range(len(candidate_idx)))
    while available and len(selected) < top_n:
        best_local = max(
            available,
            key=lambda local: lambda_ * float(relevance[candidate_idx[local]])
            - (1.0 - lambda_) * (float(profile_sim[local, selected].max()) if selected else 0.0),
        )
        selected.append(best_local)
        available.remove(best_local)

    return [
        {"term": terms[candidate_idx[local]], "score": float(relevance[candidate_idx[local]])}
        for local in selected
    ]


def run_variant(name: str, papers: list[dict], args: argparse.Namespace) -> None:
    out_dir = topic_dir(name)
    top_terms_path = out_dir / "top_terms.json"
    reps_path = out_dir / "representative_papers.json"
    if top_terms_path.exists() and reps_path.exists() and not args.force:
        print(f"[skip] {name} topics exist; use --force")
        return

    labels = np.load(variant_dir(name) / "labels.npy")
    proba = np.load(variant_dir(name) / "proba.npy")
    k = int(proba.shape[1])
    docs = cluster_documents(papers, labels, k)
    scores, terms = ctfidf(docs, args.min_df, args.max_features)

    top_terms = {
        str(cluster_id): mmr_terms(
            scores,
            terms,
            cluster_id,
            args.top_candidates,
            args.top_n,
            args.mmr_lambda,
        )
        for cluster_id in range(k)
    }

    reps_idx = representative_indices(proba, labels, k, top_n=args.rep_n)
    reps = {}
    for cluster_id, idxs in reps_idx.items():
        reps[cluster_id] = [
            {
                "paper_id": papers[i]["paper_id"],
                "title": papers[i].get("title", ""),
                "abstract": compact_abstract(papers[i].get("abstract", ""), args.abstract_chars),
                "probability": float(proba[i, int(cluster_id)]),
            }
            for i in idxs
        ]

    save_json(top_terms_path, top_terms)
    save_json(reps_path, reps)
    print(f"[done] {name}: K={k}, terms={len(terms)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=[v.name for v in ALL_VARIANTS])
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--rep-n", type=int, default=10)
    parser.add_argument("--top-candidates", type=int, default=30)
    parser.add_argument("--mmr-lambda", type=float, default=0.7)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--abstract-chars", type=int, default=900)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    papers = load_papers_in_embedding_order()
    for name in tqdm(args.variants, desc="topic terms"):
        run_variant(name, papers, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
