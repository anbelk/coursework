from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
from tqdm import tqdm

from common.compat import LLMCache, compact_abstract, load_json, load_papers_in_embedding_order, save_json, topic_dir, variant_dir


DEFAULT_VARIANTS = [
    "hdbscan_fine",
    "hdbscan_medium",
    "kmeans_umap10_k979",
    "kmeans_umap10_k242",
    "meta_hdbscan_medium",
]


FIT_SYSTEM_PROMPT = """You are a strict evaluator of scientific topic assignments.
Return only valid JSON:
{
  "paper_scores": [0, 1, 2, 3],
  "reason": "brief explanation"
}
Use exactly this scale for each paper:
0 = not related to the topic
1 = weak/general relation only
2 = related to the topic
3 = clearly fits the topic and the required granularity
Be conservative. Judge only from the supplied title and abstract."""


INTRUDER_SYSTEM_PROMPT = """You evaluate scientific topic coherence.
Return only valid JSON:
{
  "intruder_index": 1,
  "reason": "brief explanation"
}
Exactly one listed paper is an intruder that should not belong to the topic.
Return its 1-based index. Judge only from the supplied title and abstract."""


def granularity_for_variant(name: str) -> str:
    if "fine" in name or name.endswith("_k979") or name.endswith("_k1000"):
        return "fine"
    if "medium" in name or name.endswith("_k242") or name.endswith("_k200"):
        return "medium"
    if name.startswith("meta_") or "metacluster" in name:
        return "metacluster"
    return "coarse"


def normalize_score(value: Any) -> int:
    try:
        score = int(value)
    except Exception:
        score = 0
    return max(0, min(3, score))


def assignment_confidence(labels: np.ndarray, proba: np.ndarray, chunk_size: int = 65_536) -> np.ndarray:
    confidence = np.zeros((len(labels),), dtype=np.float32)
    for start in range(0, len(labels), chunk_size):
        end = min(start + chunk_size, len(labels))
        lab = labels[start:end]
        good = (lab >= 0) & (lab < proba.shape[1])
        if not np.any(good):
            continue
        rows = np.arange(start, end, dtype=np.int64)[good]
        cols = lab[good].astype(np.int64, copy=False)
        confidence[start:end][good] = np.asarray(proba[rows, cols], dtype=np.float32)
    return confidence


def topic_context(name: str, cluster_id: int) -> str:
    top_terms_path = topic_dir(name) / "top_terms.json"
    label_path = topic_dir(name) / "llm_label.json"
    parts = [f"Cluster id: {cluster_id}"]
    if label_path.exists():
        labels = load_json(label_path)
        row = labels.get(str(cluster_id), {})
        if row:
            parts.append(f"Cluster label: {row.get('name', '')}")
            parts.append(f"Cluster description: {row.get('description', '')}")
    if top_terms_path.exists():
        terms = load_json(top_terms_path).get(str(cluster_id), [])
        if terms:
            parts.append("Top terms: " + ", ".join(str(t.get("term", "")) for t in terms[:15]))
    return "\n".join(parts)


def paper_block(papers: list[dict[str, Any]]) -> str:
    rows = []
    for i, paper in enumerate(papers, start=1):
        rows.append(
            f"{i}. Title: {paper.get('title', '')}\n"
            f"Abstract: {compact_abstract(paper.get('abstract', ''), 700)}"
        )
    return "\n\n".join(rows)


def fit_prompt(name: str, cluster_id: int, granularity: str, papers: list[dict[str, Any]]) -> str:
    return f"""Required granularity: {granularity}

{topic_context(name, cluster_id)}

Random papers assigned to this cluster:
{paper_block(papers)}

Score each paper independently. The output array must contain exactly {len(papers)} scores."""


def intruder_prompt(name: str, cluster_id: int, granularity: str, papers: list[dict[str, Any]]) -> str:
    return f"""Required granularity: {granularity}

{topic_context(name, cluster_id)}

Papers:
{paper_block(papers)}

Find the one paper that least belongs to this cluster."""


def choose_cluster_ids(labels: np.ndarray, k: int, sample_clusters: int | None, rng: np.random.Generator) -> list[int]:
    sizes = np.bincount(labels[labels >= 0], minlength=k)
    cluster_ids = np.flatnonzero(sizes > 0)
    if sample_clusters is not None and len(cluster_ids) > sample_clusters:
        cluster_ids = np.sort(rng.choice(cluster_ids, size=sample_clusters, replace=False))
    return [int(x) for x in cluster_ids.tolist()]


def paper_level_labels_for_variant(name: str, base_variant: str) -> tuple[np.ndarray, np.ndarray]:
    if not name.startswith("meta_") and "metacluster" not in name:
        labels = np.asarray(np.load(variant_dir(name) / "labels.npy", mmap_mode="r"))
        proba = np.load(variant_dir(name) / "proba.npy", mmap_mode="r")
        return labels, proba

    base_labels = np.asarray(np.load(variant_dir(base_variant) / "labels.npy", mmap_mode="r"))
    base_proba = np.load(variant_dir(base_variant) / "proba.npy", mmap_mode="r")
    meta_proba = np.asarray(np.load(variant_dir(name) / "proba.npy", mmap_mode="r"), dtype=np.float32)
    meta_hard = np.asarray(np.load(variant_dir(name) / "labels.npy", mmap_mode="r"), dtype=np.int32)
    k = int(meta_proba.shape[1])

    paper_labels = np.full((len(base_labels),), -1, dtype=np.int32)
    good = (base_labels >= 0) & (base_labels < len(meta_hard))
    paper_labels[good] = meta_hard[base_labels[good]]

    paper_proba = np.zeros((len(base_labels), k), dtype=np.float32)
    good_meta = good & (paper_labels >= 0)
    if np.any(good_meta):
        base_ids = base_labels[good_meta].astype(np.int64, copy=False)
        base_conf = np.asarray(base_proba[np.flatnonzero(good_meta), base_ids], dtype=np.float32)
        paper_proba[np.flatnonzero(good_meta)] = meta_proba[base_ids] * base_conf[:, None]
    return paper_labels, paper_proba


def choose_assigned(labels: np.ndarray, cluster_id: int, n: int, rng: np.random.Generator) -> list[int]:
    idxs = np.flatnonzero(labels == cluster_id)
    if len(idxs) < n:
        return []
    return rng.choice(idxs, size=n, replace=False).astype(int).tolist()


def choose_intruder(
    labels: np.ndarray,
    proba: np.ndarray,
    confidence: np.ndarray,
    cluster_id: int,
    epsilon: float,
    min_other_confidence: float,
    rng: np.random.Generator,
) -> int | None:
    candidates = np.flatnonzero((labels >= 0) & (labels != cluster_id))
    if len(candidates) == 0:
        return None
    if cluster_id < proba.shape[1]:
        cluster_weight = np.asarray(proba[candidates, cluster_id], dtype=np.float32)
        candidates = candidates[cluster_weight <= epsilon]
    if len(candidates) == 0:
        return None
    confident = candidates[confidence[candidates] >= min_other_confidence]
    if len(confident):
        candidates = confident
    return int(rng.choice(candidates))


def _evaluate_cluster(
    name: str,
    cluster_id: int,
    labels: np.ndarray,
    proba: np.ndarray,
    confidence: np.ndarray,
    papers: list[dict[str, Any]],
    cache: LLMCache,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    seed = args.seed + cluster_id * 104_729 + sum(ord(c) for c in name)
    rng = np.random.default_rng(seed)
    granularity = granularity_for_variant(name)
    assigned = choose_assigned(labels, cluster_id, args.papers_per_cluster, rng)
    if len(assigned) < args.papers_per_cluster:
        return None

    fit_papers = [papers[i] for i in assigned]
    fit_obj = cache.complete_json(FIT_SYSTEM_PROMPT, fit_prompt(name, cluster_id, granularity, fit_papers))
    raw_scores = list(fit_obj.get("paper_scores", []))
    scores = [normalize_score(x) for x in raw_scores[: args.papers_per_cluster]]
    if len(scores) < args.papers_per_cluster:
        scores.extend([0] * (args.papers_per_cluster - len(scores)))

    intruder_idx = choose_intruder(
        labels,
        proba,
        confidence,
        cluster_id,
        args.intruder_epsilon,
        args.min_other_confidence,
        rng,
    )
    intruder_ok = None
    intruder_answer = None
    intruder_reason = ""
    intruder_position = None
    if intruder_idx is not None:
        inlier_idxs = choose_assigned(labels, cluster_id, args.papers_per_cluster - 1, rng)
        if len(inlier_idxs) == args.papers_per_cluster - 1:
            mixed = [{"idx": i, "is_intruder": False} for i in inlier_idxs]
            mixed.append({"idx": intruder_idx, "is_intruder": True})
            py_rng = random.Random(seed)
            py_rng.shuffle(mixed)
            intruder_position = 1 + next(i for i, row in enumerate(mixed) if row["is_intruder"])
            intruder_papers = [papers[row["idx"]] for row in mixed]
            intruder_obj = cache.complete_json(
                INTRUDER_SYSTEM_PROMPT,
                intruder_prompt(name, cluster_id, granularity, intruder_papers),
            )
            try:
                intruder_answer = int(intruder_obj.get("intruder_index"))
            except Exception:
                intruder_answer = None
            intruder_ok = intruder_answer == intruder_position
            intruder_reason = str(intruder_obj.get("reason", "")).strip()

    return {
        "cluster_id": cluster_id,
        "granularity": granularity,
        "assigned_paper_idxs": assigned,
        "assigned_confidence_mean": float(confidence[assigned].mean()),
        "paper_scores": scores,
        "fit_mean": float(np.mean(scores)),
        "fit_reason": str(fit_obj.get("reason", "")).strip(),
        "intruder_paper_idx": intruder_idx,
        "intruder_position": intruder_position,
        "intruder_answer": intruder_answer,
        "intruder_correct": intruder_ok,
        "intruder_reason": intruder_reason,
    }


def run_variant(name: str, cache: LLMCache, args: argparse.Namespace) -> None:
    out_path = topic_dir(name) / "assigned_llm_metrics.json"
    if out_path.exists() and not args.force:
        print(f"[skip] {name} assigned LLM metrics exist; use --force")
        return

    labels, proba = paper_level_labels_for_variant(name, args.base_variant)
    confidence = assignment_confidence(labels, proba)
    papers = load_papers_in_embedding_order()
    rng = np.random.default_rng(args.seed)
    cluster_ids = choose_cluster_ids(labels, int(proba.shape[1]), args.sample_clusters, rng)

    rows: list[dict[str, Any]] = []
    if args.workers <= 1:
        for cluster_id in tqdm(cluster_ids, desc=f"llm assigned {name}"):
            row = _evaluate_cluster(name, cluster_id, labels, proba, confidence, papers, cache, args)
            if row is not None:
                rows.append(row)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(_evaluate_cluster, name, cluster_id, labels, proba, confidence, papers, cache, args)
                for cluster_id in cluster_ids
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"llm assigned {name}"):
                row = future.result()
                if row is not None:
                    rows.append(row)

    fit_scores = [row["fit_mean"] for row in rows]
    intruder_rows = [row for row in rows if row["intruder_correct"] is not None]
    result = {
        "variant": name,
        "base_variant": args.base_variant if name.startswith("meta_") or "metacluster" in name else None,
        "prompt_version": "assigned_fit_intruder_v1",
        "sample_clusters": args.sample_clusters,
        "n_clusters_requested": len(cluster_ids),
        "n_clusters_evaluated": len(rows),
        "papers_per_cluster": args.papers_per_cluster,
        "intruder_epsilon": args.intruder_epsilon,
        "min_other_confidence": args.min_other_confidence,
        "llm_assigned_fit@10": float(np.mean(fit_scores)) if fit_scores else None,
        "llm_intruder_accuracy@10": float(np.mean([bool(row["intruder_correct"]) for row in intruder_rows]))
        if intruder_rows
        else None,
        "n_intruder_clusters_evaluated": len(intruder_rows),
        "clusters": sorted(rows, key=lambda x: int(x["cluster_id"])),
    }
    save_json(out_path, result)
    print(
        f"[done] {name}: assigned_fit={result['llm_assigned_fit@10']} "
        f"intruder={result['llm_intruder_accuracy@10']}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate assigned cluster fit and intruder accuracy with an LLM judge")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--sample-clusters", type=int, default=80)
    parser.add_argument("--papers-per-cluster", type=int, default=10)
    parser.add_argument("--intruder-epsilon", type=float, default=0.02)
    parser.add_argument("--min-other-confidence", type=float, default=0.30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-variant", default="hdbscan_fine", help="Paper-level base clustering for meta variants")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache = LLMCache(model=args.model) if args.model else LLMCache()
    for name in args.variants:
        run_variant(name, cache, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
