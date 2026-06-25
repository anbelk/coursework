from __future__ import annotations

from evaluation import retrieval_consistency


def main() -> int:
    return int(retrieval_consistency.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
