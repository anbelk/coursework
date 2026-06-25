from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.author_splits import HISTORY_CUTOFF, split_author_set
from common.compat import AUTHORS, LLMCache, RESULTS, load_json, save_json
from evaluation.coauthor_retrieval import (
    _coauthor_index,
    cutoff_examples,
    mean_candidate_embeddings,
    model_candidate_embeddings,
    past_coauthors,
    relevant_future_coauthors,
)
from recommendation.pipeline import (
    graded_ndcg,
    normalize_recommendation_type,
    normalize_score,
    openalex_url,
    RELEVANCE_SYSTEM_PROMPT,
)
from recommendation.training_utils import (
    COAUTHOR_INFONCE_MODEL_NAME,
    MEAN_BASELINE_NAME,
    AuthorExampleDataset,
    collate_examples,
    examples_with_positives,
    load_author_arrays,
    load_model_checkpoint,
    load_paper_years,
    mean_history_embeddings,
    model_dir,
    pick_device,
    predict_embeddings,
    split_examples,
)


GRAPH_RETRIEVAL_METHODS = {"graphsage_author", "graphsage_author_metacluster"}
DEFAULT_LLM_EVAL_METHODS = [
    COAUTHOR_INFONCE_MODEL_NAME,
    MEAN_BASELINE_NAME,
    "graphsage_author",
    "graphsage_author_metacluster",
]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norm, 1e-12, out=norm)
    return x / norm


def graph_author_embeddings(method: str) -> tuple[list[str], np.ndarray]:
    author_index = load_json(AUTHORS / "author_index.json")
    author_index.sort(key=lambda x: x["author_id"])
    ids_path = model_dir(method) / "author_ids.json"
    author_ids = load_json(ids_path) if ids_path.exists() else [row["author_id"] for row in author_index]
    emb_path = model_dir(method) / "author_embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"missing graph embeddings: {emb_path}")
    emb = l2_normalize(np.load(emb_path).astype(np.float32, copy=False))
    if emb.shape[0] != len(author_ids):
        raise ValueError(f"{method}: embeddings rows={emb.shape[0]}, author_ids={len(author_ids)}")
    return author_ids, emb


def is_embedding_file_method(method: str) -> bool:
    return (model_dir(method) / "author_embeddings.npy").exists()


def author_recent_papers_for_eval(author_id: str, cutoff: int, max_papers: int = 5) -> list[dict[str, Any]]:
    idx = _coauthor_index()
    entry = idx["author_by_id"].get(author_id)
    papers = idx["papers"]
    if entry is None:
        return []
    hist = [
        int(paper_idx)
        for paper_idx, year in zip(entry["paper_idxs"], entry["years"], strict=False)
        if int(year) <= cutoff
    ]
    from common.compat import compact_abstract

    records = []
    for n, paper_idx in enumerate(hist[-max_papers:], start=1):
        paper = papers[int(paper_idx)]
        records.append(
            {
                "idx": n,
                "paper_id": paper["paper_id"],
                "title": paper.get("title", ""),
                "abstract": compact_abstract(paper.get("abstract", ""), 600),
                "url": openalex_url(paper["paper_id"]),
            }
        )
    return records


def paper_context(records: list[dict[str, Any]]) -> str:
    chunks = []
    for rec in records:
        chunks.append(
            f"{rec['idx']}. Paper id: {rec['paper_id']}\n"
            f"Title: {rec['title']}\n"
            f"Abstract: {rec['abstract']}\n"
            f"OpenAlex: {rec['url']}"
        )
    return "\n\n".join(chunks)


def build_eval_prompt(user_id: str, candidate_id: str, cutoff: int) -> str:
    user_papers = author_recent_papers_for_eval(user_id, cutoff)
    cand_papers = author_recent_papers_for_eval(candidate_id, cutoff)
    return f"""User author id: {user_id}
User name: {author_name_from_index(user_id)}

User recent papers:
{paper_context(user_papers)}

Candidate author id: {candidate_id}
Candidate name: {author_name_from_index(candidate_id)}

Candidate recent papers:
{paper_context(cand_papers)}

Evaluate whether this candidate is a relevant potential collaborator for the user."""


def author_name_from_index(author_id: str) -> str:
    idx = _coauthor_index()
    return str(idx["author_by_id"].get(author_id, {}).get("name", author_id))


@torch.inference_mode()
def retrieve_top_k(
    focal_emb: np.ndarray,
    candidate_ids: list[str],
    candidate_emb: np.ndarray,
    user_id: str,
    cutoff: int,
    top_k: int,
) -> list[dict[str, Any]]:
    scores = candidate_emb @ focal_emb.astype(np.float32, copy=False)
    cid_to_pos = {cid: i for i, cid in enumerate(candidate_ids)}
    excluded = {user_id} | past_coauthors(user_id, cutoff)
    for cid in excluded:
        j = cid_to_pos.get(cid)
        if j is not None:
            scores[j] = -np.inf
    top_local = np.argsort(-scores)[:top_k]
    out = []
    for rank, j in enumerate(top_local, start=1):
        if scores[int(j)] == -np.inf:
            continue
        aid = candidate_ids[int(j)]
        out.append(
            {
                "author_id": aid,
                "name": author_name_from_index(aid),
                "dense_rank": rank,
                "model_cosine": float(scores[int(j)]),
            }
        )
    return out


@torch.inference_mode()
def build_retrieval_cache(
    method: str,
    model_name: str,
    split: str,
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    cutoff: int,
    max_history: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    if method == MEAN_BASELINE_NAME:
        return mean_candidate_embeddings(cutoff, embeddings, q, years, max_history, split=split)
    if method in GRAPH_RETRIEVAL_METHODS or is_embedding_file_method(method):
        author_ids, graph_emb = graph_author_embeddings(method)
        pos = {aid: i for i, aid in enumerate(author_ids)}
        candidate_ids, _ = cutoff_examples(cutoff, max_history, split=split)
        kept_ids = [aid for aid in candidate_ids if aid in pos]
        return kept_ids, graph_emb[[pos[aid] for aid in kept_ids]]
    model, _ = load_model_checkpoint(model_dir(model_name) / "best.pt", device)
    return model_candidate_embeddings(
        model, cutoff, embeddings, q, years, device, max_history, split=split
    )


@torch.inference_mode()
def focal_embeddings_for_examples(
    method: str,
    model_name: str,
    examples: list[dict[str, Any]],
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    max_history: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    if method in GRAPH_RETRIEVAL_METHODS or is_embedding_file_method(method):
        author_ids, graph_emb = graph_author_embeddings(method)
        pos = {aid: i for i, aid in enumerate(author_ids)}
        kept_ids = [ex["author_id"] for ex in examples if ex["author_id"] in pos]
        return kept_ids, graph_emb[[pos[aid] for aid in kept_ids]]

    ds = AuthorExampleDataset(examples, embeddings, q, years, max_history=max_history)
    loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_examples, num_workers=0)
    if method == MEAN_BASELINE_NAME:
        emb = mean_history_embeddings(loader)
        return [ex["author_id"] for ex in examples], emb
    model, _ = load_model_checkpoint(model_dir(model_name) / "best.pt", device)
    emb, _ = predict_embeddings(model, loader, device)
    return [ex["author_id"] for ex in examples], emb.astype(np.float32, copy=False)


def llm_score_candidates(
    user_id: str,
    candidates: list[dict[str, Any]],
    cutoff: int,
    cache: LLMCache,
    workers: int = 1,
) -> list[dict[str, Any]]:
    def score_one(cand: dict[str, Any]) -> dict[str, Any]:
        prompt = build_eval_prompt(user_id, cand["author_id"], cutoff)
        obj = cache.complete_json(RELEVANCE_SYSTEM_PROMPT, prompt)
        return {
            **cand,
            "llm_score": normalize_score(obj.get("score")),
            "recommendation_type": normalize_recommendation_type(obj.get("recommendation_type")),
            "llm_reason": str(obj.get("reason", "")).strip(),
        }

    if workers <= 1 or len(candidates) <= 1:
        return [score_one(cand) for cand in candidates]
    scored_by_id: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(score_one, cand) for cand in candidates]
        for future in as_completed(futures):
            row = future.result()
            scored_by_id[row["author_id"]] = row
    return [scored_by_id[cand["author_id"]] for cand in candidates if cand["author_id"] in scored_by_id]


def summarize_llm_rows(rows: list[dict[str, Any]], top_n: int) -> dict[str, float]:
    metric_prefix = f"@{top_n}"
    if not rows:
        return {
            f"llm_ndcg{metric_prefix}": 0.0,
            f"llm_relevant_rate{metric_prefix}_ge2": 0.0,
            f"mean_llm_score{metric_prefix}": 0.0,
        }
    by_user: dict[str, list[int]] = {}
    for row in rows:
        by_user.setdefault(row["user_id"], []).append(int(row["llm_score"]))
    per_user = []
    for scores in by_user.values():
        ordered = scores[:top_n]
        while len(ordered) < top_n:
            ordered.append(0)
        per_user.append(
            {
                f"llm_ndcg{metric_prefix}": graded_ndcg(ordered),
                f"llm_relevant_rate{metric_prefix}_ge2": float(np.mean([s >= 2 for s in ordered])),
                f"mean_llm_score{metric_prefix}": float(np.mean(ordered)),
            }
        )
    df = pd.DataFrame(per_user)
    return {
        f"llm_ndcg{metric_prefix}": float(df[f"llm_ndcg{metric_prefix}"].mean()),
        f"llm_relevant_rate{metric_prefix}_ge2": float(df[f"llm_relevant_rate{metric_prefix}_ge2"].mean()),
        f"mean_llm_score{metric_prefix}": float(df[f"mean_llm_score{metric_prefix}"].mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM-eval raw retrieval candidates")
    parser.add_argument("--model", default=COAUTHOR_INFONCE_MODEL_NAME)
    parser.add_argument("--methods", nargs="*", default=DEFAULT_LLM_EVAL_METHODS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-users", type=int, default=50)
    parser.add_argument("--top-n", type=int, default=None, help="Number of retrieval candidates scored by LLM")
    parser.add_argument("--top-k", type=int, default=10, help="Deprecated alias for --top-n")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--q-mode", choices=["none", "fine", "metacluster", "fine_metacluster"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    top_n = int(args.top_n if args.top_n is not None else args.top_k)

    out_dir = RESULTS / "retrieval" / "llm_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        print("[skip] retrieval LLM eval exists; use --force")
        return 0

    device = pick_device(args.device)
    if args.q_mode is None:
        cfg_path = model_dir(args.model) / "config.json"
        args.q_mode = load_json(cfg_path).get("q_mode", "fine") if cfg_path.exists() else "fine"
    embeddings, q = load_author_arrays(args.q_mode)
    years = load_paper_years()
    cutoff = HISTORY_CUTOFF
    author_pool = split_author_set(args.split)
    examples = examples_with_positives(split_examples(args.split), author_pool=author_pool)
    rng = random.Random(args.seed)
    sample = rng.sample(examples, min(args.n_users, len(examples)))
    save_json(out_dir / "user_pool.json", {"seed": args.seed, "user_ids": [ex["author_id"] for ex in sample]})

    methods = args.methods
    caches: dict[str, tuple[list[str], np.ndarray]] = {}
    focal_maps: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        cand_ids, cand_emb = build_retrieval_cache(
            method, args.model, args.split, embeddings, q, years, cutoff, args.max_history, device
        )
        caches[method] = (cand_ids, cand_emb)
        focal_ids, focal_emb = focal_embeddings_for_examples(
            method, args.model, sample, embeddings, q, years, args.max_history, device
        )
        focal_maps[method] = {aid: focal_emb[i] for i, aid in enumerate(focal_ids)}

    llm_cache = LLMCache()
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, float]] = {}

    for method in tqdm(methods, desc="methods"):
        cand_ids, cand_emb = caches[method]
        method_rows: list[dict[str, Any]] = []
        for ex in tqdm(sample, desc=method, leave=False):
            uid = ex["author_id"]
            if uid not in focal_maps[method]:
                continue
            candidates = retrieve_top_k(
                focal_maps[method][uid],
                cand_ids,
                cand_emb,
                uid,
                cutoff,
                top_n,
            )
            scored = llm_score_candidates(uid, candidates, cutoff, llm_cache, workers=args.workers)
            for row in scored:
                method_rows.append(
                    {
                        "method": method,
                        "user_id": uid,
                        "candidate_id": row["author_id"],
                        "dense_rank": row["dense_rank"],
                        "model_cosine": row["model_cosine"],
                        "llm_score": row["llm_score"],
                        "recommendation_type": row["recommendation_type"],
                        "llm_reason": row["llm_reason"],
                    }
                )
        all_rows.extend(method_rows)
        summaries[method] = summarize_llm_rows(method_rows, top_n)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "ratings.csv", index=False)
    summary = {
        "split": args.split,
        "n_users": len(sample),
        "top_n": top_n,
        "top_k_deprecated_arg": args.top_k,
        "methods": summaries,
    }
    save_json(summary_path, summary)

    plt.figure(figsize=(7, 4))
    for method in methods:
        sub = df[df["method"] == method]
        sub["llm_score"].value_counts().sort_index().reindex([0, 1, 2, 3], fill_value=0).plot(
            kind="bar", alpha=0.6, label=method
        )
    plt.xlabel("LLM relevance score")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "score_histogram.png", dpi=160)
    plt.close()

    print(f"[done] {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
