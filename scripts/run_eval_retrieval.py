from __future__ import annotations

import sys

from evaluation import eval_retrieval


def main() -> int:
    return int(eval_retrieval.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
