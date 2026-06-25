from __future__ import annotations

import argparse
import sys

from clustering.paper_topics import aggregate, cluster_baselines, topic_terms
from common.compat import BASELINE_VARIANTS
from evaluation import metric_coherence, metric_distinctness, metric_quant
from llm import label_topics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run clustering baselines and evaluate them")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM labeling and coherence/distinctness")
    args = parser.parse_args()

    variant_names = [v.name for v in BASELINE_VARIANTS]
    argv = ["--force"] if args.force else []
    sys.argv = [sys.argv[0], *argv]
    code = cluster_baselines.main()
    if code not in (None, 0):
        return int(code)

    sys.argv = [sys.argv[0], *["--variants", *variant_names], *(["--force"] if args.force else [])]
    for step in (metric_quant.main, topic_terms.main):
        code = step()
        if code not in (None, 0):
            return int(code)

    if not args.skip_llm:
        sys.argv = [sys.argv[0], *["--variants", *variant_names], *(["--force"] if args.force else [])]
        for step in (label_topics.main, metric_coherence.main, metric_distinctness.main):
            code = step()
            if code not in (None, 0):
                return int(code)

    code = aggregate.main()
    return int(code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
