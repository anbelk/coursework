from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from clustering.paper_topics import aggregate, topic_terms
from clustering.paper_topics.cluster_baselines import run_kmeans
from common.clustering_variants import Variant
from common.compat import HDBSCAN_VARIANTS, load_embeddings, load_json, topic_dir, variant_dir
from common.compat import REDUCED
from evaluation import metric_coherence, metric_distinctness, metric_quant
from llm import label_topics

HDBSCAN_CANDIDATES = [v.name for v in HDBSCAN_VARIANTS[:2]]


def kmeans_name(k: int) -> str:
    return f"kmeans_umap10_k{k}"


def run_llm_metrics(name: str, cache, force: bool, workers: int) -> None:
    label_topics.run_variant(name, cache, force, workers=workers)
    metric_coherence.run_variant(name, cache, force, workers=workers)
    metric_distinctness.run_variant(name, cache, force, tau=0.85, max_pairs=300, workers=workers)


def pick_hdbscan_winner() -> tuple[str, int]:
    rows: list[tuple[str, float, float, int]] = []
    for name in HDBSCAN_CANDIDATES:
        coherence = load_json(topic_dir(name) / "coherence.json")
        distinct = load_json(topic_dir(name) / "distinctness.json")
        quant = load_json(topic_dir(name) / "metrics_quant.json")
        rows.append(
            (
                name,
                float(coherence["weighted_mean"]),
                float(distinct["duplicate_pair_rate"]),
                int(quant["n_clusters"]),
            )
        )
    rows.sort(key=lambda x: (-x[1], x[2]))
    winner, _, _, k = rows[0]
    print(f"[winner] {winner} (K={k}, coherence={rows[0][1]:.4f}, dup_rate={rows[0][2]:.4f})")
    return winner, k


def ensure_kmeans(k: int, force: bool) -> str:
    name = kmeans_name(k)
    out_dir = variant_dir(name)
    if not (out_dir / "labels.npy").exists() or force:
        features = np.load(REDUCED / "umap_n10.npy").astype(np.float32, copy=False)
        embeddings = load_embeddings()
        variant = Variant(name, "kmeans", k=k)
        run_kmeans(features, embeddings, variant, force=True)
    return name


def ensure_quant_and_terms(names: list[str], force: bool) -> None:
    extra = ["--force"] if force else []
    sys.argv = ["metric_quant", "--variants", *names, *extra]
    code = metric_quant.main()
    if code not in (None, 0):
        raise SystemExit(int(code))
    sys.argv = ["topic_terms", "--variants", *names, *extra]
    code = topic_terms.main()
    if code not in (None, 0):
        raise SystemExit(int(code))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM eval for hdbscan_fine/medium, pick winner, kmeans baseline at winner K",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=6, help="Parallel LLM calls per variant")
    parser.add_argument(
        "--parallel-variants",
        action="store_true",
        help="Run hdbscan_fine and hdbscan_medium LLM pipelines concurrently",
    )
    parser.add_argument("--skip-hdbscan", action="store_true", help="Skip HDBSCAN LLM (reuse existing metrics)")
    args = parser.parse_args()
    workers = max(1, args.workers)

    from common.llm_cache import LLMCache

    if not args.skip_hdbscan:
        print(f"=== LLM metrics for {HDBSCAN_CANDIDATES} (workers={workers}) ===")

        def run_one(name: str) -> None:
            cache = LLMCache()
            run_llm_metrics(name, cache, args.force, workers)

        if args.parallel_variants:
            with ThreadPoolExecutor(max_workers=len(HDBSCAN_CANDIDATES)) as pool:
                futures = {pool.submit(run_one, name): name for name in HDBSCAN_CANDIDATES}
                for future in as_completed(futures):
                    name = futures[future]
                    future.result()
                    print(f"[done] {name} LLM pipeline")
        else:
            for name in HDBSCAN_CANDIDATES:
                run_one(name)
                print(f"[done] {name} LLM pipeline")

    winner, k = pick_hdbscan_winner()
    medium_k = int(load_json(topic_dir(HDBSCAN_CANDIDATES[1]) / "metrics_quant.json")["n_clusters"])
    baseline_ks = sorted({k, medium_k})
    baselines: list[str] = []
    for bk in baseline_ks:
        name = ensure_kmeans(bk, force=args.force)
        print(f"=== baseline {name} at K={bk} ===")
        ensure_quant_and_terms([name], force=args.force)
        baselines.append(name)

    for baseline in baselines:
        llm_files = ("llm_label.json", "coherence.json", "distinctness.json")
        if not args.force and all((topic_dir(baseline) / f).exists() for f in llm_files):
            print(f"[skip] {baseline} LLM complete; use --force to recompute")
            continue
        cache = LLMCache()
        run_llm_metrics(baseline, cache, args.force, workers=workers)

    final_variants = [*HDBSCAN_CANDIDATES, *baselines]
    print(f"=== aggregate {final_variants} ===")
    sys.argv = ["aggregate", "--variants", *final_variants]
    code = aggregate.main()
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
