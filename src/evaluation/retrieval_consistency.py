from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common.author_splits import HISTORY_CUTOFF, split_author_set
from common.compat import RESULTS, save_json
from evaluation.coauthor_retrieval import past_coauthors, relevant_future_coauthors
from evaluation.eval_retrieval_llm import (
    build_retrieval_cache,
    focal_embeddings_for_examples,
)
from recommendation.pipeline import graded_ndcg
from recommendation.training_utils import (
    COAUTHOR_INFONCE_MODEL_NAME,
    examples_with_positives,
    load_author_arrays,
    load_paper_years,
    pick_device,
    split_examples,
)


DEFAULT_METHODS = [
    "mean_author_embedding",
    "author_retriever",
    "graphsage_author",
    "graphsage_author_metacluster",
    "graphsage_transformer_author",
    "graphsage_transformer_author_metacluster",
]


@dataclass
class MethodRun:
    method: str
    user_ids: list[str]
    candidate_ids: list[str]
    scores: np.ndarray
    relevant_positions: list[set[int]]


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    np.maximum(norm, 1e-12, out=norm)
    return x / norm


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2:
        return float("nan")
    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    a = a - float(a.mean())
    b = b - float(b.mean())
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return float("nan")
    return float((a @ b) / denom)


def rank_vector_desc(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(scores), dtype=np.float32)
    return ranks


def retrieval_metrics_from_relevance(
    scores: np.ndarray,
    relevant_positions: list[set[int]],
    ks: tuple[int, ...],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    order = np.argsort(-scores, axis=1)
    out: dict[str, float] = {
        "n_examples": float(len(relevant_positions)),
        "n_with_relevant": float(sum(bool(rel) for rel in relevant_positions)),
    }
    per_user: dict[str, np.ndarray] = {}
    for k in ks:
        hits = np.zeros(len(relevant_positions), dtype=np.float32)
        mrrs = np.zeros(len(relevant_positions), dtype=np.float32)
        ndcgs = np.zeros(len(relevant_positions), dtype=np.float32)
        recalls = np.zeros(len(relevant_positions), dtype=np.float32)
        discounts = 1.0 / np.log2(np.arange(2, k + 2))
        for i, (row, rel) in enumerate(zip(order[:, :k], relevant_positions, strict=False)):
            if not rel:
                continue
            flags = np.array([idx in rel for idx in row], dtype=np.float32)
            hits[i] = float(flags.any())
            recalls[i] = float(flags.sum() / len(rel))
            if flags.any():
                first = int(np.flatnonzero(flags)[0]) + 1
                mrrs[i] = 1.0 / first
            dcg = float((flags * discounts).sum())
            ideal_n = min(len(rel), k)
            idcg = float(discounts[:ideal_n].sum()) if ideal_n else 0.0
            ndcgs[i] = dcg / idcg if idcg else 0.0
        out[f"hit@{k}"] = float(hits.mean())
        out[f"mrr@{k}"] = float(mrrs.mean())
        out[f"ndcg@{k}"] = float(ndcgs.mean())
        out[f"recall@{k}"] = float(recalls.mean())
        per_user[f"hit@{k}"] = hits
        per_user[f"mrr@{k}"] = mrrs
        per_user[f"ndcg@{k}"] = ndcgs
        per_user[f"recall@{k}"] = recalls
    return out, per_user


@torch.inference_mode()
def build_method_run(
    method: str,
    split: str,
    max_history: int,
    device: torch.device,
    embeddings: np.ndarray,
    q: np.ndarray,
    years: np.ndarray,
    examples: list[dict[str, Any]],
    author_pool: set[str],
) -> MethodRun:
    cutoff = HISTORY_CUTOFF
    candidate_ids, candidate_emb = build_retrieval_cache(
        method,
        COAUTHOR_INFONCE_MODEL_NAME,
        split,
        embeddings,
        q,
        years,
        cutoff,
        max_history,
        device,
    )
    focal_ids, focal_emb = focal_embeddings_for_examples(
        method,
        COAUTHOR_INFONCE_MODEL_NAME,
        examples,
        embeddings,
        q,
        years,
        max_history,
        device,
    )
    focal_pos = {aid: i for i, aid in enumerate(focal_ids)}
    filtered_examples = [ex for ex in examples if ex["author_id"] in focal_pos]
    user_ids = [ex["author_id"] for ex in filtered_examples]
    focal_idx = [focal_pos[aid] for aid in user_ids]

    focal = l2_normalize(focal_emb[focal_idx])
    candidates = l2_normalize(candidate_emb)
    scores = focal @ candidates.T

    cid_to_pos = {cid: i for i, cid in enumerate(candidate_ids)}
    relevant_positions: list[set[int]] = []
    for i, ex in enumerate(filtered_examples):
        uid = ex["author_id"]
        excluded = {uid} | past_coauthors(uid, int(ex["cutoff_year"]))
        for cid in excluded:
            pos = cid_to_pos.get(cid)
            if pos is not None:
                scores[i, pos] = -np.inf
        rel = relevant_future_coauthors(ex, author_pool=author_pool)
        relevant_positions.append({cid_to_pos[aid] for aid in rel if aid in cid_to_pos})
    return MethodRun(method, user_ids, candidate_ids, scores.astype(np.float32), relevant_positions)


def top_ids(run: MethodRun, user_idx: int, k: int) -> list[str]:
    order = np.argsort(-run.scores[user_idx])[:k]
    return [run.candidate_ids[int(j)] for j in order if np.isfinite(run.scores[user_idx, int(j)])]


def pairwise_correlations(
    runs: dict[str, MethodRun],
    ks: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for left_name, right_name in itertools.combinations(runs, 2):
        left = runs[left_name]
        right = runs[right_name]
        right_user_pos = {uid: i for i, uid in enumerate(right.user_ids)}
        right_cand_pos = {cid: i for i, cid in enumerate(right.candidate_ids)}
        common_candidates = [cid for cid in left.candidate_ids if cid in right_cand_pos]
        left_cand_idx = np.array([left.candidate_ids.index(cid) for cid in common_candidates], dtype=np.int64)
        right_cand_idx = np.array([right_cand_pos[cid] for cid in common_candidates], dtype=np.int64)

        pearsons = []
        spearmans = []
        overlaps = {k: [] for k in ks}
        jaccards = {k: [] for k in ks}
        n_users = 0
        for i, uid in enumerate(left.user_ids):
            j = right_user_pos.get(uid)
            if j is None:
                continue
            left_scores = left.scores[i, left_cand_idx]
            right_scores = right.scores[j, right_cand_idx]
            finite = np.isfinite(left_scores) & np.isfinite(right_scores)
            if finite.sum() >= 2:
                pearsons.append(pearson(left_scores[finite], right_scores[finite]))
                left_ranks = rank_vector_desc(left_scores[finite])
                right_ranks = rank_vector_desc(right_scores[finite])
                spearmans.append(pearson(left_ranks, right_ranks))
            for k in ks:
                left_top = set(top_ids(left, i, k))
                right_top = set(top_ids(right, j, k))
                if left_top and right_top:
                    inter = len(left_top & right_top)
                    overlaps[k].append(inter / float(k))
                    jaccards[k].append(inter / float(len(left_top | right_top)))
            n_users += 1

        row: dict[str, Any] = {
            "method_left": left_name,
            "method_right": right_name,
            "n_users": n_users,
            "n_common_candidates": len(common_candidates),
            "mean_score_pearson": float(np.nanmean(pearsons)) if pearsons else float("nan"),
            "mean_rank_spearman": float(np.nanmean(spearmans)) if spearmans else float("nan"),
        }
        for k in ks:
            row[f"top{k}_overlap_rate"] = float(np.mean(overlaps[k])) if overlaps[k] else 0.0
            row[f"top{k}_jaccard"] = float(np.mean(jaccards[k])) if jaccards[k] else 0.0
        rows.append(row)
    return rows


def paired_bootstrap(
    per_user: dict[str, dict[str, np.ndarray]],
    baseline: str,
    metrics: list[str],
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    base = per_user[baseline]
    n = len(next(iter(base.values())))
    for method, method_metrics in per_user.items():
        if method == baseline:
            continue
        for metric in metrics:
            diff = method_metrics[metric] - base[metric]
            boot = np.empty(n_boot, dtype=np.float32)
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                boot[b] = float(diff[idx].mean())
            lo, hi = np.quantile(boot, [0.025, 0.975])
            rows.append(
                {
                    "baseline": baseline,
                    "method": method,
                    "metric": metric,
                    "mean_diff": float(diff.mean()),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "significant_95": bool(lo > 0 or hi < 0),
                    "n_examples": n,
                    "n_boot": n_boot,
                }
            )
    return rows


def summarize_existing_llm(k_values: tuple[int, ...], out_dir: Path) -> pd.DataFrame | None:
    ratings_path = RESULTS / "retrieval" / "llm_eval" / "ratings.csv"
    if not ratings_path.exists():
        return None
    df = pd.read_csv(ratings_path)
    rows: list[dict[str, Any]] = []
    for method, method_df in df.groupby("method"):
        max_available_k = int(method_df["dense_rank"].max()) if len(method_df) else 0
        for k in k_values:
            if k > max_available_k:
                continue
            per_user = []
            for _, user_df in method_df.sort_values("dense_rank").groupby("user_id"):
                scores = [int(x) for x in user_df.sort_values("dense_rank")["llm_score"].head(k)]
                while len(scores) < k:
                    scores.append(0)
                top_selected = sorted(scores, reverse=True)[:3]
                while len(top_selected) < 3:
                    top_selected.append(0)
                per_user.append(
                    {
                        "llm_ndcg": graded_ndcg(scores),
                        "llm_relevant_rate_ge2": float(np.mean([s >= 2 for s in scores])),
                        "llm_mean_score": float(np.mean(scores)),
                        "llm_has_1_ge2": float(any(s >= 2 for s in scores)),
                        "llm_has_3_ge2": float(sum(s >= 2 for s in scores) >= 3),
                        "llm_has_1_eq3": float(any(s == 3 for s in scores)),
                        "llm_selected3_mean_score": float(np.mean(top_selected)),
                    }
                )
            if not per_user:
                continue
            user_metrics = pd.DataFrame(per_user)
            rows.append(
                {
                    "method": method,
                    "K": k,
                    "n_users": int(len(user_metrics)),
                    "llm_ndcg": float(user_metrics["llm_ndcg"].mean()),
                    "llm_relevant_rate_ge2": float(user_metrics["llm_relevant_rate_ge2"].mean()),
                    "llm_mean_score": float(user_metrics["llm_mean_score"].mean()),
                    "llm_has_1_ge2": float(user_metrics["llm_has_1_ge2"].mean()),
                    "llm_has_3_ge2": float(user_metrics["llm_has_3_ge2"].mean()),
                    "llm_has_1_eq3": float(user_metrics["llm_has_1_eq3"].mean()),
                    "llm_selected3_mean_score": float(user_metrics["llm_selected3_mean_score"].mean()),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "existing_llm_metrics_at_k.csv", index=False)
    return out


def write_markdown_report(
    out_dir: Path,
    metrics_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    llm_df: pd.DataFrame | None,
    primary_k: int,
) -> None:
    lines = [
        "# Retrieval consistency report",
        "",
        "## Target metric decision",
        "",
        f"Primary online budget: `top_n = {primary_k}` retrieval candidates scored by LLM, then `top_k = 3` final recommendations.",
        "",
        "Primary offline proxy metrics:",
        f"- observed coauthor `Hit@{primary_k}`: at least one future new coauthor appears in the LLM candidate set;",
        f"- observed coauthor `MRR@{primary_k}`: rank of the first future new coauthor;",
        f"- observed coauthor `nDCG@{primary_k}`: ordering of all observed future new coauthors;",
        f"- LLM `has_3_ge2@{primary_k}`: enough candidates with LLM score >= 2 to fill three final recommendations;",
        f"- LLM `has_1_eq3@{primary_k}` and `selected3_mean_score@{primary_k}`: quality of the best candidates after LLM scoring.",
        "",
        "`@5` and `@20` are diagnostic cost-sensitivity checks, not the main target.",
        "",
        "## Method policy",
        "",
        "Keep methods that test different information sources:",
        "- `mean_author_embedding`: article semantics, no training;",
        "- `author_retriever`: trained semantic author encoder;",
        "- one simple graph method: coauthor graph / metacluster graph ablation;",
        "- one combined method: semantic author vector plus graph message passing.",
        "",
        "Do not foreground graph variants that only change implementation detail and do not produce a stable improvement.",
        "",
        "## Current observed retrieval metrics",
        "",
        metrics_df.to_markdown(index=False),
        "",
        "## Pairwise signal correlation",
        "",
        corr_df.to_markdown(index=False),
        "",
        "## Paired bootstrap versus baseline",
        "",
        bootstrap_df.to_markdown(index=False),
        "",
    ]
    if llm_df is not None and not llm_df.empty:
        lines.extend(
            [
                "## Existing LLM ratings re-aggregated at K",
                "",
                "These rows reuse the existing `results/retrieval/llm_eval/ratings.csv`; they do not add new LLM calls.",
                "",
                llm_df.to_markdown(index=False),
                "",
            ]
        )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure retrieval consistency and method correlation")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--ks", nargs="*", type=int, default=[5, 10, 20])
    parser.add_argument("--primary-k", type=int, default=10)
    parser.add_argument("--baseline", default=COAUTHOR_INFONCE_MODEL_NAME)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    ks = tuple(sorted(set(args.ks)))
    out_dir = RESULTS / "retrieval" / "consistency"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    embeddings, q = load_author_arrays()
    years = load_paper_years()
    author_pool = split_author_set(args.split)
    examples = split_examples(args.split)
    runs: dict[str, MethodRun] = {}
    metrics_rows: list[dict[str, Any]] = []
    per_user: dict[str, dict[str, np.ndarray]] = {}
    for method in args.methods:
        run = build_method_run(
            method,
            args.split,
            args.max_history,
            device,
            embeddings,
            q,
            years,
            examples,
            author_pool,
        )
        runs[method] = run
        metrics, user_metrics = retrieval_metrics_from_relevance(run.scores, run.relevant_positions, ks)
        per_user[method] = user_metrics
        for k in ks:
            metrics_rows.append(
                {
                    "split": args.split,
                    "method": method,
                    "K": k,
                    "hit": metrics[f"hit@{k}"],
                    "mrr": metrics[f"mrr@{k}"],
                    "ndcg": metrics[f"ndcg@{k}"],
                    "recall": metrics[f"recall@{k}"],
                    "n_examples": int(metrics["n_examples"]),
                    "n_with_relevant": int(metrics["n_with_relevant"]),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "retrieval_metrics_at_k.csv", index=False)

    corr_df = pd.DataFrame(pairwise_correlations(runs, ks))
    corr_df.to_csv(out_dir / "method_correlations.csv", index=False)

    bootstrap_metrics = [f"hit@{args.primary_k}", f"mrr@{args.primary_k}", f"ndcg@{args.primary_k}"]
    bootstrap_df = pd.DataFrame(
        paired_bootstrap(per_user, args.baseline, bootstrap_metrics, args.bootstrap, args.seed)
    )
    bootstrap_df.to_csv(out_dir / "paired_bootstrap_vs_baseline.csv", index=False)

    llm_df = summarize_existing_llm(ks, out_dir)
    write_markdown_report(out_dir, metrics_df, corr_df, bootstrap_df, llm_df, args.primary_k)
    save_json(
        out_dir / "config.json",
        {
            "methods": args.methods,
            "split": args.split,
            "ks": list(ks),
            "primary_k": args.primary_k,
            "baseline": args.baseline,
            "max_history": args.max_history,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
    )
    print(f"[done] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
