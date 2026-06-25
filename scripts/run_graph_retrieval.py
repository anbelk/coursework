from __future__ import annotations

from recommendation import graph_retrieval


def main() -> int:
    return int(graph_retrieval.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
