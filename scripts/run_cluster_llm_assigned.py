from __future__ import annotations

from evaluation import metric_cluster_llm_assigned


def main() -> int:
    return int(metric_cluster_llm_assigned.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
