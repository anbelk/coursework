from __future__ import annotations

import argparse
import sys

import numpy as np

from clustering.meta_topics import aggregate_meta, topic_meta
from clustering.meta_topics.cluster_meta import run_kmeans
from common.clustering_variants import Variant
from common.compat import DATA, load_json, topic_dir
from evaluation import metric_meta_coherence, metric_meta_distinctness, metric_quant
from llm import label_meta_topics

META_DIR = DATA / "meta"
META_CANDIDATE = "meta_hdbscan_medium"


def meta_kmeans_name(k: int) -> str:
    return f"meta_kmeans_umap10_k{k}"


def run_llm_metrics(name: str, cache, force: bool, workers: int) -> None:
    label_meta_topics.run_variant(name, cache, force, workers=workers, fallback_only=False, require_llm=False)
    metric_meta_coherence.run_variant(name, cache, force, workers=workers)
    metric_meta_distinctness.run_variant(name, cache, force, tau=0.85, max_pairs=300, workers=workers)


def llm_complete(name: str) -> bool:
    d = topic_dir(name)
    return all((d / f).exists() for f in ("llm_label.json", "coherence.json", "distinctness.json"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM eval for meta_hdbscan_medium + kmeans baseline at same K",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-cluster", action="store_true", help="Only redo meta LLM metrics + aggregate")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    workers = max(1, args.workers)

    k = int(load_json(topic_dir(META_CANDIDATE) / "metrics_quant.json")["n_clusters"])
    baseline = meta_kmeans_name(k)

    if not args.skip_cluster:
        print(f"=== meta candidate {META_CANDIDATE} K={k}, baseline {baseline} ===")
        embeddings = np.load(META_DIR / "cluster_embeddings.npy").astype(np.float32)
        umap_features = np.load(META_DIR / "umap5.npy").astype(np.float32)
        variant = Variant(baseline, "kmeans", k=k)
        run_kmeans(umap_features, embeddings, variant, force=args.force)

        sys.argv = ["topic_meta", "--variants", baseline, *(["--force"] if args.force else [])]
        code = topic_meta.main()
        if code not in (None, 0):
            raise SystemExit(int(code))

        metric_base = ["--variants", baseline, *(["--force"] if args.force else [])]
        sys.argv = ["metric_quant", *metric_base]
        code = metric_quant.main()
        if code not in (None, 0):
            raise SystemExit(int(code))

    from common.llm_cache import LLMCache

    cache = LLMCache()
    for name in [META_CANDIDATE, baseline]:
        if name == baseline and not args.skip_cluster and not args.force and llm_complete(baseline):
            print(f"[skip] {baseline} LLM complete; use --force")
            continue
        if name == META_CANDIDATE:
            print(f"=== meta coherence/distinctness {name} ===")
            metric_meta_coherence.run_variant(name, cache, True, workers=workers)
            metric_meta_distinctness.run_variant(name, cache, True, tau=0.85, max_pairs=300, workers=workers)
        else:
            run_llm_metrics(name, cache, args.force, workers)

    final_variants = [META_CANDIDATE, baseline]
    print(f"=== aggregate {final_variants} ===")
    sys.argv = ["aggregate_meta", "--variants", *final_variants]
    code = aggregate_meta.main()
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
