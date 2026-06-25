from __future__ import annotations

from evaluation import cluster_granularity


def main() -> int:
    return int(cluster_granularity.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
