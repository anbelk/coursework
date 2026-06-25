from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from common.compat import ALL_VARIANTS, LLMCache, compact_abstract, load_json, save_json, topic_dir, variant_dir


SYSTEM_PROMPT = """You are a strict evaluator of topic coherence for clusters of scientific papers.
Judge ONLY from the paper titles and abstracts shown. Ignore any external labels or assumptions.
Return only valid JSON:
{
  "score": 0,
  "reason": "brief explanation"
}
Use exactly this scale:
0 = incoherent mixture: unrelated tasks, methods, domains, or languages with no shared concrete research direction
1 = too broad or weak: papers share only a vague field (e.g. "NLP", "machine learning", "language models") without a specific shared task, method, dataset type, or problem
2 = partially coherent: a dominant theme exists but several papers are off-topic or belong to clearly different subthemes
3 = tight and specific: almost all papers pursue one concrete research direction (same task/method/problem family)

Be conservative. If you would struggle to name one specific shared research direction in <=8 words, score <=1.
Random-looking mixtures of unrelated papers must score 0."""


def prompt_for_cluster(reps: list[dict]) -> str:
    papers = []
    for i, r in enumerate(reps[:10], start=1):
        papers.append(
            f"{i}. Title: {r['title']}\n"
            f"Abstract: {compact_abstract(r.get('abstract', ''), 600)}"
        )
    return f"""Representative papers assigned to the same cluster:

{chr(10).join(papers)}

Score how well these papers fit ONE concrete, specific research topic. Do not invent a topic that is not clearly supported by the papers."""


def normalize_score(value) -> int:
    try:
        score = int(value)
    except Exception:
        score = 0
    return max(0, min(3, score))


def _score_cluster(cluster_id: str, reps: dict, cache: LLMCache, hard_labels: np.ndarray) -> tuple[str, dict, int]:
    response = cache.complete_json(SYSTEM_PROMPT, prompt_for_cluster(reps[cluster_id]))
    score = normalize_score(response.get("score"))
    size = int((hard_labels == int(cluster_id)).sum())
    return cluster_id, {
        "score": score,
        "reason": str(response.get("reason", "")).strip(),
        "size": size,
    }, score


def run_variant(name: str, cache: LLMCache, force: bool, workers: int = 1) -> None:
    out_path = topic_dir(name) / "coherence.json"
    if out_path.exists() and not force:
        print(f"[skip] {name} coherence exists; use --force")
        return
    reps = load_json(topic_dir(name) / "representative_papers.json")
    hard_labels = np.load(variant_dir(name) / "labels.npy")
    cluster_ids = sorted(reps, key=lambda x: int(x))
    rows: dict[str, dict] = {}
    weighted_sum = 0.0
    weight_total = 0
    scores: list[int] = []
    if workers <= 1:
        for cluster_id in tqdm(cluster_ids, desc=f"coherence {name}", leave=False):
            cid, row, score = _score_cluster(cluster_id, reps, cache, hard_labels)
            rows[cid] = row
            weighted_sum += score * row["size"]
            weight_total += row["size"]
            scores.append(score)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_score_cluster, cid, reps, cache, hard_labels) for cid in cluster_ids
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"coherence {name}", leave=False):
                cid, row, score = future.result()
                rows[cid] = row
                weighted_sum += score * row["size"]
                weight_total += row["size"]
                scores.append(score)
    result = {
        "variant": name,
        "n_clusters": len(cluster_ids),
        "weighted_mean": float(weighted_sum / weight_total) if weight_total else None,
        "unweighted_mean": float(np.mean(scores)) if scores else None,
        "prompt_version": "coherence_strict_v2",
        "clusters": rows,
    }
    save_json(out_path, result)
    print(f"[done] {name}: coherence_weighted={result['weighted_mean']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=[v.name for v in ALL_VARIANTS])
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Parallel LLM calls per variant")
    args = parser.parse_args()

    cache = LLMCache(model=args.model) if args.model else LLMCache()
    for name in args.variants:
        run_variant(name, cache, args.force, workers=max(1, args.workers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
