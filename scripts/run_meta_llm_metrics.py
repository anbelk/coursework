from __future__ import annotations

import argparse
import sys

from clustering.meta_topics import aggregate_meta
from common.compat import META_ALL_VARIANTS
from evaluation import metric_meta_coherence, metric_meta_distinctness, metric_quant


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute meta LLM metrics (meta-specific judges)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    variant_names = [v.name for v in META_ALL_VARIANTS]
    force = ["--force"] if args.force else []
    workers = ["--workers", str(max(1, args.workers))]
    metric_llm = ["--variants", *variant_names, *force, *workers]
    for step in (metric_quant.main, metric_meta_coherence.main, metric_meta_distinctness.main):
        sys.argv = [sys.argv[0], *metric_llm]
        code = step()
        if code not in (None, 0):
            return int(code)

    sys.argv = [sys.argv[0]]
    code = aggregate_meta.main()
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
