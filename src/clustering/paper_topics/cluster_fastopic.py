from __future__ import annotations

import argparse
import sys
from typing import Any

import numpy as np

from common.compat import (
    EMBEDDINGS_PATH,
    RANDOM_STATE,
    compact_abstract,
    l2_normalize,
    load_papers_in_embedding_order,
    save_json,
    topic_dir,
    variant_dir,
    weighted_centroids,
)


def make_docs(papers: list[dict[str, Any]], indices: np.ndarray) -> list[str]:
    docs = []
    for idx in indices.tolist():
        paper = papers[int(idx)]
        docs.append(f"{paper.get('title', '')}. {paper.get('abstract', '')}".strip())
    return docs


def selected_indices(n_docs: int, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit <= 0 or limit >= n_docs:
        return np.arange(n_docs, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_docs, size=limit, replace=False)).astype(np.int64)


def normalize_theta(theta: Any) -> np.ndarray:
    if hasattr(theta, "detach"):
        theta = theta.detach().cpu().numpy()
    theta = np.asarray(theta, dtype=np.float32)
    row_sum = theta.sum(axis=1, keepdims=True)
    return np.where(row_sum > 0, theta / np.maximum(row_sum, 1e-12), theta).astype(np.float32)


class PresetOnlyEmbedder:
    def encode(self, docs: list[str], **_: Any) -> np.ndarray:
        raise RuntimeError("FASTopic should use preset_doc_embeddings for this experiment")


def write_full_artifacts(
    name: str,
    params: dict[str, Any],
    selected: np.ndarray,
    theta: np.ndarray,
    embeddings: np.ndarray,
    n_docs: int,
) -> np.ndarray:
    out_dir = variant_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    k = int(theta.shape[1])
    labels = np.full((n_docs,), -1, dtype=np.int32)
    labels[selected] = np.argmax(theta, axis=1).astype(np.int32)
    np.save(out_dir / "labels.npy", labels)

    proba = np.lib.format.open_memmap(out_dir / "proba.npy", mode="w+", dtype=np.float32, shape=(n_docs, k))
    proba[:] = 0.0
    proba[selected] = theta
    proba.flush()

    centroids = weighted_centroids(embeddings[selected], theta, labels[selected])
    np.save(out_dir / "centroids_qwen.npy", centroids.astype(np.float32, copy=False))
    sizes = {str(i): int((labels[selected] == i).sum()) for i in range(k)}
    save_json(out_dir / "sizes.json", sizes)
    save_json(out_dir / "params.json", params)
    return labels


def write_topics(
    name: str,
    papers: list[dict[str, Any]],
    selected: np.ndarray,
    theta: np.ndarray,
    top_words: Any,
    rep_n: int,
) -> None:
    out_dir = topic_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)
    k = int(theta.shape[1])
    top_terms: dict[str, list[dict[str, Any]]] = {}
    for topic_id in range(k):
        words = top_words[topic_id] if topic_id < len(top_words) else []
        if isinstance(words, str):
            words = words.split()
        top_terms[str(topic_id)] = [
            {"term": str(word), "score": float(len(words) - i)}
            for i, word in enumerate(list(words)[:15])
        ]

    reps: dict[str, list[dict[str, Any]]] = {}
    for topic_id in range(k):
        order = np.argsort(-theta[:, topic_id])[:rep_n]
        rows = []
        for local_idx in order.tolist():
            paper_idx = int(selected[int(local_idx)])
            paper = papers[paper_idx]
            rows.append(
                {
                    "paper_id": paper["paper_id"],
                    "title": paper.get("title", ""),
                    "abstract": compact_abstract(paper.get("abstract", ""), 900),
                    "probability": float(theta[int(local_idx), topic_id]),
                }
            )
        reps[str(topic_id)] = rows
    save_json(out_dir / "top_terms.json", top_terms)
    save_json(out_dir / "representative_papers.json", reps)


def run_variant(args: argparse.Namespace) -> None:
    from fastopic import FASTopic
    from topmost.preprocess import Preprocess

    out_dir = variant_dir(args.name)
    if (out_dir / "labels.npy").exists() and (out_dir / "proba.npy").exists() and not args.force:
        print(f"[skip] {args.name} exists; use --force")
        return

    papers = load_papers_in_embedding_order()
    embeddings = l2_normalize(np.load(EMBEDDINGS_PATH, mmap_mode="r"))
    selected = selected_indices(len(papers), args.limit, args.seed)
    docs = make_docs(papers, selected)
    selected_emb = np.asarray(embeddings[selected], dtype=np.float32)
    preprocess = Preprocess(
        stopwords="English",
        min_doc_count=args.min_doc_count,
        max_doc_freq=args.max_doc_freq,
        min_length=args.min_length,
        vocab_size=args.vocab_size,
        seed=args.seed,
        verbose=args.verbose,
    )
    model = FASTopic(
        num_topics=args.num_topics,
        preprocess=preprocess,
        num_top_words=args.num_top_words,
        device=args.device,
        doc_embed_model=PresetOnlyEmbedder(),
        normalize_embeddings=False,
        low_memory=True,
        low_memory_batch_size=args.batch_size,
        verbose=args.verbose,
        log_interval=max(1, args.log_interval),
    )
    top_words, theta = model.fit_transform(
        docs,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        preset_doc_embeddings=selected_emb,
    )
    theta = normalize_theta(theta)
    params = {
        "method": "fastopic",
        "feature_space": "qwen_l2_preset_doc_embeddings",
        "n_topics": args.num_topics,
        "limit": int(len(selected)),
        "n_docs_total": int(len(papers)),
        "sample_seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "vocab_size": args.vocab_size,
        "min_doc_count": args.min_doc_count,
        "max_doc_freq": args.max_doc_freq,
        "batch_size": args.batch_size,
        "device": args.device,
        "status": "sampled" if len(selected) < len(papers) else "full",
    }
    write_full_artifacts(args.name, params, selected, theta, embeddings, len(papers))
    write_topics(args.name, papers, selected, theta, top_words, args.rep_n)
    print(f"[done] {args.name}: topics={args.num_topics}, docs={len(selected)}, epochs={args.epochs}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FASTopic with existing paper embeddings")
    parser.add_argument("--name", required=True)
    parser.add_argument("--num-topics", type=int, required=True)
    parser.add_argument("--limit", type=int, default=50_000, help="0 means full corpus")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=20_000)
    parser.add_argument("--min-doc-count", type=int, default=5)
    parser.add_argument("--max-doc-freq", type=float, default=0.5)
    parser.add_argument("--min-length", type=int, default=3)
    parser.add_argument("--num-top-words", type=int, default=15)
    parser.add_argument("--rep-n", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_variant(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
