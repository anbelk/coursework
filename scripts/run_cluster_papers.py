from __future__ import annotations

import argparse

from collections.abc import Callable

from clustering.paper_topics import aggregate, cluster_fcm, cluster_gmm, cluster_hdbscan, reduce_pca, reduce_umap, topic_terms


def run_step(name: str, fn: Callable[[], int | None]) -> None:
    print(f"[step] {name}")
    code = fn()
    if code not in (None, 0):
        raise SystemExit(int(code))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run paper clustering pipeline")
    parser.parse_args()
    steps = [
        ("reduce_umap", reduce_umap.main),
        ("reduce_pca", reduce_pca.main),
        ("cluster_hdbscan", cluster_hdbscan.main),
        ("cluster_fcm", cluster_fcm.main),
        ("cluster_gmm", cluster_gmm.main),
        ("topic_terms", topic_terms.main),
        ("aggregate", aggregate.main),
    ]
    for name, fn in steps:
        run_step(name, fn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
