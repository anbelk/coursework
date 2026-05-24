from __future__ import annotations

import argparse
import sys

from tqdm import tqdm

from pipeline_common import ALL_VARIANTS, LLMCache, load_json, save_json, topic_dir


SYSTEM_PROMPT = """You label scientific-paper clusters.
Return only valid JSON with:
{
  "name": "short specific topic name, max 8 words",
  "description": "one or two concise sentences explaining the shared research topic"
}
Avoid generic names such as 'machine learning' or 'neural networks'. Focus on the exact technique, mechanism, task, architecture component, or evaluation target."""


def build_prompt(cluster_id: str, terms: list[dict], reps: list[dict]) -> str:
    top_terms = ", ".join(t["term"] for t in terms[:15])
    titles = "\n".join(f"- {r['title']}" for r in reps[:5])
    return f"""Cluster id: {cluster_id}

Top c-TF-IDF/MMR terms:
{top_terms}

Representative paper titles:
{titles}

Create a precise scientific topic label for this cluster."""


def run_variant(name: str, cache: LLMCache, force: bool) -> None:
    out_path = topic_dir(name) / "llm_label.json"
    if out_path.exists() and not force:
        print(f"[skip] {name} labels exist; use --force")
        return
    top_terms = load_json(topic_dir(name) / "top_terms.json")
    reps = load_json(topic_dir(name) / "representative_papers.json")
    labels = {}
    for cluster_id in tqdm(sorted(top_terms, key=lambda x: int(x)), desc=f"label {name}", leave=False):
        response = cache.complete_json(SYSTEM_PROMPT, build_prompt(cluster_id, top_terms[cluster_id], reps[cluster_id]))
        labels[cluster_id] = {
            "name": str(response.get("name", "")).strip(),
            "description": str(response.get("description", "")).strip(),
        }
    save_json(out_path, labels)
    print(f"[done] {name}: labels={len(labels)}")


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
