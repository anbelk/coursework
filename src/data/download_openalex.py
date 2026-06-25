"""Download OpenAlex works for a given primary topic and save to JSONL.

Filters:
    primary_topic.id:T10181
    publication_year:>2016
    type:article
    has_abstract:true

Output JSONL fields per record:
    paper_id (str, e.g. "W3174770825" — без URL-префикса)
    title (str)
    abstract (str)
    year (int)
    authors (list of {"author_id": str, "name": str})
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import requests
from tqdm import tqdm

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
DEFAULT_FILTERS = (
    "primary_topic.id:T10181,"
    "publication_year:>2016,"
    "type:article,"
    "has_abstract:true"
)
SELECT_FIELDS = "id,title,publication_year,abstract_inverted_index,authorships"


def reconstruct_abstract(inv_index: dict[str, list[int]] | None) -> str:
    """Restore abstract text from OpenAlex inverted index."""
    if not inv_index:
        return ""
    positions: list[tuple[int, str]] = []
    for token, idxs in inv_index.items():
        for i in idxs:
            positions.append((i, token))
    positions.sort(key=lambda x: x[0])
    return " ".join(tok for _, tok in positions)


def strip_openalex_id(url_id: str | None) -> str:
    """'https://openalex.org/W123' -> 'W123' (без URL)."""
    if not url_id:
        return ""
    return url_id.rsplit("/", 1)[-1]


def extract_authors(authorships: list[dict[str, Any]] | None) -> list[dict[str, str]]:
    if not authorships:
        return []
    out: list[dict[str, str]] = []
    for a in authorships:
        author = a.get("author") or {}
        aid = strip_openalex_id(author.get("id"))
        name = author.get("display_name") or ""
        if not aid and not name:
            continue
        out.append({"author_id": aid, "name": name})
    return out


def iter_works(
    filters: str,
    mailto: str,
    per_page: int = 200,
    max_retries: int = 6,
) -> Iterable[dict[str, Any]]:
    """Iterate OpenAlex works using cursor pagination."""
    cursor: str | None = "*"
    session = requests.Session()
    while cursor:
        params = {
            "filter": filters,
            "per-page": per_page,
            "cursor": cursor,
            "select": SELECT_FIELDS,
            "mailto": mailto,
        }
        for attempt in range(max_retries):
            try:
                r = session.get(OPENALEX_WORKS_URL, params=params, timeout=60)
                if r.status_code == 429 or 500 <= r.status_code < 600:
                    raise requests.HTTPError(f"status {r.status_code}")
                r.raise_for_status()
                payload = r.json()
                break
            except (requests.RequestException, ValueError) as e:
                wait = 2 ** attempt
                print(
                    f"[warn] request failed ({e}); retry in {wait}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
        else:
            raise RuntimeError("OpenAlex API failed after retries")

        meta = payload.get("meta", {})
        results = payload.get("results", [])
        for w in results:
            yield w

        cursor = meta.get("next_cursor")
        if not results:
            break


def to_record(work: dict[str, Any]) -> dict[str, Any] | None:
    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    title = work.get("title") or ""
    if not title or not abstract:
        return None
    return {
        "paper_id": strip_openalex_id(work.get("id")),
        "title": title,
        "abstract": abstract,
        "year": work.get("publication_year"),
        "authors": extract_authors(work.get("authorships")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="data/openalex_clean.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--mailto",
        default="belkinandrey2003@gmail.com",
        help="Email for the OpenAlex polite pool",
    )
    parser.add_argument(
        "--filters",
        default=DEFAULT_FILTERS,
        help="OpenAlex filter string",
    )
    parser.add_argument("--per-page", type=int, default=200)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    written = 0
    skipped = 0

    print(f"[info] filters: {args.filters}")
    print(f"[info] writing -> {out_path}")

    pbar = tqdm(unit="paper")
    with out_path.open("w", encoding="utf-8") as f:
        for work in iter_works(args.filters, args.mailto, per_page=args.per_page):
            rec = to_record(work)
            if rec is None or not rec["paper_id"]:
                skipped += 1
                pbar.update(1)
                continue
            if rec["paper_id"] in seen:
                skipped += 1
                pbar.update(1)
                continue
            seen.add(rec["paper_id"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            pbar.update(1)
            if written % 5000 == 0:
                f.flush()
    pbar.close()

    print(f"[done] written: {written}; skipped: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
