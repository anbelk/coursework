from __future__ import annotations

import argparse
import sys

import numpy as np
from tqdm import tqdm

from pipeline_common import ALL_VARIANTS, LLMCache, compact_abstract, load_json, save_json, topic_dir, variant_dir


SYSTEM_PROMPT = """You evaluate topic coherence for a cluster of scientific papers.
Return only valid JSON:
{
  "score": 0,
  "reason": "brief explanation"
}
Use exactly this scale:
0 = incoherent mixture, no single research theme
1 = too broad; papers share only a broad field
2 = understandable topic but with noticeable noise or mixed subthemes
3 = clear, specific research direction."""


def prompt_for_cluster(label: dict, reps: list[dict]) -> str:
    papers = []
    for i, r in enumerate(reps[:10], start=1):
        papers.append(
            f"{i}. Title: {r['title']}\n"
            f"Abstract: {compact_abstract(r.get('abstract', ''), 600)}"
        )
    return f"""LLM topic label:
Name: {label.get('name', '')}
Description: {label.get('description', '')}

Top representative papers:
{chr(10).join(papers)}

Score how well these papers fit one concrete research topic."""


def normalize_score(value) -> int:
    try:
        score = int(value)
    except Exception:
        score = 0
    return max(0, min(3, score))


def run_variant(name: str, cache: LLMCache, force: bool) -> None:
    out_path = topic_dir(name) / "coherence.json"
    if out_path.exists() and not force:
        print(f"[skip] {name} coherence exists; use --force")
        return
    labels_llm = load_json(topic_dir(name) / "llm_label.json")
    reps = load_json(topic_dir(name) / "representative_papers.json")
    hard_labels = np.load(variant_dir(name) / "labels.npy")
    k = len(labels_llm)
    rows = {}
    weighted_sum = 0.0
    weight_total = 0
    scores = []
    for cluster_id in tqdm(sorted(labels_llm, key=lambda x: int(x)), desc=f"coherence {name}", leave=False):
        response = cache.complete_json(SYSTEM_PROMPT, prompt_for_cluster(labels_llm[cluster_id], reps[cluster_id]))
        score = normalize_score(response.get("score"))
        size = int((hard_labels == int(cluster_id)).sum())
        rows[cluster_id] = {
            "score": score,
            "reason": str(response.get("reason", "")).strip(),
            "size": size,
        }
        weighted_sum += score * size
        weight_total += size
        scores.append(score)
    result = {
        "variant": name,
        "n_clusters": k,
        "weighted_mean": float(weighted_sum / weight_total) if weight_total else None,
        "unweighted_mean": float(np.mean(scores)) if scores else None,
        "clusters": rows,
    }
    save_json(out_path, result)
    print(f"[done] {name}: coherence_weighted={result['weighted_mean']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=[v.name for v in ALL_VARIANTS])
    parser.add_argument("--model", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache = LLMCache(model=args.model) if args.model else LLMCache()
    for name in args.variants:
        run_variant(name, cache, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
