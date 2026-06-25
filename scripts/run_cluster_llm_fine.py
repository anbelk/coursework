from __future__ import annotations

import sys

from evaluation import metric_cluster_llm_assigned


def main() -> int:
    sys.argv = [
        sys.argv[0],
        "--variants",
        "hdbscan_fine",
        "kmeans_umap10_k979",
        *sys.argv[1:],
    ]
    return int(metric_cluster_llm_assigned.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
