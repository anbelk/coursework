from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from tqdm import tqdm

from common.compat import DATA, LLMCache, load_json, save_json


HIERARCHY_DIR = DATA / "hierarchy"

SYSTEM_PROMPT = """You label hierarchical scientific research topics.
Return only valid JSON:
{
  "name": "short specific topic name",
  "description": "one sentence description"
}
Names must be concise, specific, and suitable for a topic map."""


def build_prompt(node: dict[str, Any], terms: list[dict[str, Any]], child_labels: list[str]) -> str:
    term_text = ", ".join(str(item.get("term", "")) for item in terms[:12] if item.get("term"))
    children = "\n".join(f"- {label}" for label in child_labels[:10])
    return f"""Topic node id: {node["id"]}
Approximate paper count: {node["paper_count"]}

Top terms:
{term_text}

Child topic labels:
{children}

Create a short, human-readable label for this aggregated research topic."""


def label_node(node_id: str, tree: dict[str, Any], node_terms: dict[str, list[dict[str, Any]]], cache: LLMCache) -> tuple[str, dict[str, str]]:
    node = tree["nodes"][node_id]
    child_labels = [tree["nodes"][child_id].get("label", child_id) for child_id in node.get("children", [])]
    obj = cache.complete_json(SYSTEM_PROMPT, build_prompt(node, node_terms.get(node_id, []), child_labels))
    name = str(obj.get("name", "")).strip() or str(node.get("label", node_id))
    description = str(obj.get("description", "")).strip()
    return node_id, {"name": name, "description": description}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path = HIERARCHY_DIR / "labels.json"
    if out_path.exists() and not args.force:
        print("[skip] hierarchy labels exist; use --force")
        return 0

    tree = load_json(HIERARCHY_DIR / "tree.json")
    node_terms = load_json(HIERARCHY_DIR / "node_top_terms.json")
    node_ids = list(tree["levels"]["0"]) + list(tree["levels"]["1"])
    labels: dict[str, dict[str, str]] = {}
    cache = LLMCache()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(label_node, node_id, tree, node_terms, cache) for node_id in node_ids]
        for future in tqdm(as_completed(futures), total=len(futures), desc="hierarchy labels"):
            node_id, label = future.result()
            labels[node_id] = label

    save_json(out_path, labels)
    print(f"[done] wrote {out_path} ({len(labels)} labels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
