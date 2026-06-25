from __future__ import annotations

import argparse
import sys

from clustering.meta_topics import aggregate_meta
from common.compat import ALL_CLUSTERING_EVAL_VARIANTS, META_ALL_VARIANTS
from evaluation import metric_coherence, metric_meta_coherence


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute strict coherence and refresh aggregates")
    parser.add_argument("--skip-meta", action="store_true")
    parser.add_argument("--skip-fine", action="store_true")
    parser.add_argument("--force", action="store_true", default=True)
    args = parser.parse_args()

    if not args.skip_fine:
        fine_names = [v.name for v in ALL_CLUSTERING_EVAL_VARIANTS]
        sys.argv = [sys.argv[0], "--variants", *fine_names, "--force"]
        code = metric_coherence.main()
        if code not in (None, 0):
            return int(code)

    if not args.skip_meta:
        meta_names = [v.name for v in META_ALL_VARIANTS]
        sys.argv = [sys.argv[0], "--variants", *meta_names, "--force"]
        code = metric_meta_coherence.main()
        if code not in (None, 0):
            return int(code)

    if not args.skip_fine:
        from clustering.paper_topics import aggregate

        sys.argv = [sys.argv[0]]
        code = aggregate.main()
        if code not in (None, 0):
            return int(code)

    if not args.skip_meta:
        sys.argv = [sys.argv[0]]
        code = aggregate_meta.main()
        if code not in (None, 0):
            return int(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
