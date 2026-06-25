from __future__ import annotations

import argparse

from evaluation import measure_recs_coverage


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure recommendation coverage")
    parser.parse_args()
    return int(measure_recs_coverage.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
