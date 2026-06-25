from __future__ import annotations

from clustering.paper_topics import cluster_fastopic


def main() -> int:
    return int(cluster_fastopic.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
