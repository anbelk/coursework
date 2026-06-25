from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from common.compat import META_ALL_VARIANTS, LLMCache, load_json, save_json, topic_dir


SYSTEM_PROMPT = """You label broad NLP research areas formed by grouping fine-grained topic clusters.
Return only valid JSON with:
{
  "name": "short specific broad NLP area name, max 8 words",
  "description": "one or two concise sentences explaining the shared research area"
}
Avoid generic names such as 'NLP' or 'machine learning'. Focus on the shared task family, method paradigm, model type, data setting, or evaluation direction."""


def build_prompt(cluster_id: str, terms: list[dict], reps: list[dict], fine_reps: list[dict]) -> str:
    top_terms = ", ".join(t["term"] for t in terms[:15])
    titles = "\n".join(f"- {r['title']}" for r in reps[:8])
    fine_ids = ", ".join(str(r["fine_cluster_id"]) for r in fine_reps[:8])
    return f"""Meta-cluster id: {cluster_id}

Top c-TF-IDF/MMR terms:
{top_terms}

Representative fine-cluster ids:
{fine_ids}

Representative paper titles:
{titles}

Create a precise label for this broader NLP research area."""


def fallback_label(cluster_id: str, terms: list[dict]) -> dict[str, str]:
    picked = [str(item.get("term", "")).strip() for item in terms[:4] if item.get("term")]
    name = " / ".join(picked[:2]).title() if picked else f"Meta Area {cluster_id}"
    description = f"Broad NLP research area characterized by: {', '.join(picked)}." if picked else ""
    return {"name": name[:80], "description": description}


def label_one(
    cluster_id: str,
    top_terms: dict,
    reps: dict,
    fine_reps: dict,
    cache: LLMCache,
    require_llm: bool,
) -> tuple[str, dict[str, str]]:
    try:
        response = cache.complete_json(
            SYSTEM_PROMPT,
            build_prompt(cluster_id, top_terms[cluster_id], reps[cluster_id], fine_reps[cluster_id]),
            max_retries=2,
        )
        label = {
            "name": str(response.get("name", "")).strip(),
            "description": str(response.get("description", "")).strip(),
        }
        if label["name"]:
            return cluster_id, label
        if require_llm:
            raise RuntimeError(f"empty LLM label for cluster {cluster_id}")
    except Exception:
        if require_llm:
            raise
    return cluster_id, fallback_label(cluster_id, top_terms[cluster_id])


def run_variant(name: str, cache: LLMCache, force: bool, workers: int, fallback_only: bool, require_llm: bool) -> None:
    out_path = topic_dir(name) / "llm_label.json"
    if out_path.exists() and not force:
        print(f"[skip] {name} labels exist; use --force")
        return
    top_terms = load_json(topic_dir(name) / "top_terms.json")
    reps = load_json(topic_dir(name) / "representative_papers.json")
    fine_reps = load_json(topic_dir(name) / "representative_fine_clusters.json")
    labels = {}
    cluster_ids = sorted(top_terms, key=lambda x: int(x))
    if fallback_only:
        labels = {cid: fallback_label(cid, top_terms[cid]) for cid in cluster_ids}
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(label_one, cid, top_terms, reps, fine_reps, cache, require_llm) for cid in cluster_ids]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"meta label {name}", leave=False):
                cluster_id, label = future.result()
                labels[cluster_id] = label
    save_json(out_path, labels)
    print(f"[done] {name}: labels={len(labels)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=[v.name for v in META_ALL_VARIANTS])
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fallback-only", action="store_true")
    parser.add_argument("--require-llm", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache = LLMCache(model=args.model) if args.model else LLMCache()
    for name in args.variants:
        run_variant(name, cache, args.force, args.workers, args.fallback_only, args.require_llm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
