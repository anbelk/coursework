from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from common.compat import LLMCache, compact_abstract, load_json, save_json, topic_dir, variant_dir


SYSTEM_PROMPT = """You evaluate coherence of a META-CLUSTER: several fine-grained topic clusters grouped into one broader research area (semantic overlay, not a strict hierarchy).
Judge from the fine topic names and representative papers shown. Do NOT apply fine-grain topic strictness.
Return only valid JSON:
{
  "score": 0,
  "reason": "brief explanation"
}
Use exactly this scale:
0 = incoherent umbrella: fine topics/papers span unrelated broad areas; no defensible shared research area
1 = too vague: only a wide field fits (e.g. "AI", "machine learning", "NLP") without a recognizable sub-area or problem family
2 = partially coherent: a dominant broader area exists, but several fine topics are weakly related or clear outliers
3 = coherent broader area: fine topics belong to one recognizable research area, task family, method paradigm, or application domain

Thematic diversity within the area is expected at meta level. Score 3 when the umbrella is meaningful for navigation, not when every paper shares one narrow fine-grained task."""


def prompt_for_meta_cluster(reps: list[dict]) -> str:
    fine_names = []
    seen = set()
    for r in reps:
        name = str(r.get("fine_cluster_label", "")).strip()
        if name and name not in seen:
            seen.add(name)
            fine_names.append(name)
    fine_block = "\n".join(f"- {n}" for n in fine_names[:12])
    papers = []
    for i, r in enumerate(reps[:8], start=1):
        fine = str(r.get("fine_cluster_label", "")).strip()
        papers.append(
            f"{i}. [{fine}] {r['title']}\n"
            f"   Abstract: {compact_abstract(r.get('abstract', ''), 400)}"
        )
    return f"""Fine topics merged into this meta-cluster:
{fine_block or '(none listed)'}

Representative papers (with fine topic):
{chr(10).join(papers)}

Score how well these fine topics and papers form ONE coherent broader research area."""


def normalize_score(value) -> int:
    try:
        score = int(value)
    except Exception:
        score = 0
    return max(0, min(3, score))


def meta_cluster_paper_count(meta_labels: np.ndarray, cluster_id: int, fine_sizes: dict) -> int:
    total = 0
    for fine_id in np.flatnonzero(meta_labels == cluster_id):
        total += int(fine_sizes.get(str(int(fine_id)), 0))
    return total


def _score_cluster(
    cluster_id: str,
    reps: dict,
    cache: LLMCache,
    meta_labels: np.ndarray,
    fine_sizes: dict,
) -> tuple[str, dict, int]:
    response = cache.complete_json(SYSTEM_PROMPT, prompt_for_meta_cluster(reps[cluster_id]))
    score = normalize_score(response.get("score"))
    size = meta_cluster_paper_count(meta_labels, int(cluster_id), fine_sizes)
    return cluster_id, {
        "score": score,
        "reason": str(response.get("reason", "")).strip(),
        "size": size,
        "n_fine_topics": int((meta_labels == int(cluster_id)).sum()),
    }, score


def run_variant(name: str, cache: LLMCache, force: bool, workers: int = 1) -> None:
    out_path = topic_dir(name) / "coherence.json"
    if out_path.exists() and not force:
        print(f"[skip] {name} meta coherence exists; use --force")
        return
    reps = load_json(topic_dir(name) / "representative_papers.json")
    meta_labels = np.load(variant_dir(name) / "labels.npy")
    fine_sizes = load_json(variant_dir("hdbscan_fine") / "sizes.json")
    cluster_ids = sorted(reps, key=lambda x: int(x))
    rows: dict[str, dict] = {}
    weighted_sum = 0.0
    weight_total = 0
    scores: list[int] = []
    if workers <= 1:
        for cluster_id in tqdm(cluster_ids, desc=f"meta coherence {name}", leave=False):
            cid, row, score = _score_cluster(cluster_id, reps, cache, meta_labels, fine_sizes)
            rows[cid] = row
            weighted_sum += score * row["size"]
            weight_total += row["size"]
            scores.append(score)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_score_cluster, cid, reps, cache, meta_labels, fine_sizes) for cid in cluster_ids
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"meta coherence {name}", leave=False):
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
        "prompt_version": "meta_coherence_v1",
        "clusters": rows,
    }
    save_json(out_path, result)
    print(f"[done] {name}: meta coherence_weighted={result['weighted_mean']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    cache = LLMCache(model=args.model) if args.model else LLMCache()
    for name in args.variants:
        run_variant(name, cache, args.force, workers=max(1, args.workers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
