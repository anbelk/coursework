"""Filter a raw OpenAlex corpus down to papers of "qualified" authors.

An author is qualified if they have >= --min-papers papers in the corpus. A paper is
kept if at least one of its authors is qualified. This shrinks the embedding workload
to the part of the corpus that actually feeds the coauthor recommender, while keeping
every paper of every qualified author (so their full history and coauthorship edges
among qualified authors are preserved).

Usage:
    uv run python -m data.filter_qualified_corpus \
        --input data/openalex_ai.jsonl --output data/openalex_clean.jsonl --min-papers 5
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/openalex_ai.jsonl")
    parser.add_argument("--output", default="data/openalex_clean.jsonl")
    parser.add_argument("--min-papers", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Only print stats, do not write output")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    author_counts: dict[str, int] = defaultdict(int)
    n_total = 0
    for rec in tqdm(iter_jsonl(in_path), desc="pass 1: count authors"):
        n_total += 1
        seen: set[str] = set()
        for a in rec.get("authors", []):
            aid = a.get("author_id")
            if aid and aid not in seen:
                seen.add(aid)
                author_counts[aid] += 1

    qualified = {aid for aid, c in author_counts.items() if c >= args.min_papers}
    print(
        f"[stats] papers={n_total} distinct_authors={len(author_counts)} "
        f"qualified_authors(>= {args.min_papers})={len(qualified)}"
    )

    n_kept = 0
    if args.dry_run:
        for rec in tqdm(iter_jsonl(in_path), desc="pass 2: count kept"):
            if any(a.get("author_id") in qualified for a in rec.get("authors", [])):
                n_kept += 1
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for rec in tqdm(iter_jsonl(in_path), desc="pass 2: write kept"):
                if any(a.get("author_id") in qualified for a in rec.get("authors", [])):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_kept += 1

    pct = 100.0 * n_kept / n_total if n_total else 0.0
    print(f"[done] kept {n_kept}/{n_total} papers ({pct:.1f}%) with >=1 qualified author")
    if not args.dry_run:
        print(f"[done] wrote -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
