from __future__ import annotations

import argparse
import sys

from collections.abc import Callable

from clustering.meta_topics import aggregate_meta, build_meta_features, build_ui_mappings, cluster_meta, topic_meta
from common.compat import META_ALL_VARIANTS
from embeddings import embed_meta_clusters
from evaluation import metric_meta_coherence, metric_meta_distinctness, metric_quant
from llm import label_meta_topics


def run_step(name: str, fn: Callable[[], int | None], extra: list[str] | None = None) -> None:
    print(f"[step] {name}")
    old_argv = sys.argv
    sys.argv = [old_argv[0], *(extra or [])]
    try:
        code = fn()
    finally:
        sys.argv = old_argv
    if code not in (None, 0):
        raise SystemExit(int(code))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run meta-clustering pipeline")
    parser.add_argument("--force", action="store_true", help="Recompute all intermediate artifacts")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM meta-topic labeling and metrics")
    parser.add_argument("--workers", type=int, default=6, help="Parallel LLM calls")
    parser.add_argument(
        "--allow-proxy",
        action="store_true",
        help="Allow proxy metrics in meta winner selection (aggregate_meta)",
    )
    args = parser.parse_args()

    force = ["--force"] if args.force else []
    meta_names = [v.name for v in META_ALL_VARIANTS]
    steps: list[tuple[str, Callable[[], int | None], list[str] | None]] = [
        ("embed_meta_clusters", embed_meta_clusters.main, force),
        ("build_meta_features", build_meta_features.main, force),
        ("cluster_meta", cluster_meta.main, force),
        ("topic_meta", topic_meta.main, force),
    ]
    if not args.skip_llm:
        steps.append(
            (
                "label_meta_topics",
                label_meta_topics.main,
                [*force, "--workers", str(max(1, args.workers))],
            )
        )
        metric_base = ["--variants", *meta_names, *force]
        metric_llm = [*metric_base, "--workers", str(max(1, args.workers))]
        steps.extend(
            [
                ("metric_quant", metric_quant.main, metric_base),
                ("metric_meta_coherence", metric_meta_coherence.main, metric_llm),
                ("metric_meta_distinctness", metric_meta_distinctness.main, metric_llm),
            ]
        )
    aggregate_extra = ["--allow-proxy"] if args.allow_proxy else []
    steps.extend(
        [
            ("aggregate_meta", aggregate_meta.main, aggregate_extra or None),
            ("build_ui_mappings", build_ui_mappings.main, None),
        ]
    )
    for name, fn, extra in steps:
        run_step(name, fn, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
