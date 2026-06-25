from __future__ import annotations

from evaluation import hybrid_retrieval


def main() -> int:
    return int(hybrid_retrieval.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
