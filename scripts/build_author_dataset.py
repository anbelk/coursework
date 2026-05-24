from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from pipeline_common import (
    AUTHORS,
    CLUSTERING,
    PAPERS_PATH,
    ensure_dirs,
    iter_jsonl,
    load_paper_ids,
    save_json,
)


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_q_with_noise(force: bool) -> tuple[Path, int]:
    out_path = AUTHORS / "q_with_noise.npy"
    if out_path.exists() and not force:
        q = np.load(out_path, mmap_mode="r")
        return out_path, int(q.shape[1])

    proba = np.load(CLUSTERING / "hdbscan_fine" / "proba.npy").astype(np.float32, copy=False)
    noise = np.clip(1.0 - proba.sum(axis=1, keepdims=True), 0.0, 1.0).astype(np.float32)
    q = np.concatenate([proba, noise], axis=1)
    q /= np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
    np.save(out_path, q.astype(np.float32, copy=False))
    return out_path, int(q.shape[1])


def build_author_index(min_papers: int) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    paper_ids = load_paper_ids()
    paper_idx_by_id = {paper_id: i for i, paper_id in enumerate(paper_ids)}
    papers = []
    author_papers: dict[str, list[int]] = defaultdict(list)
    author_names: dict[str, str] = {}

    for rec in tqdm(iter_jsonl(PAPERS_PATH), desc="read papers"):
        idx = paper_idx_by_id.get(rec["paper_id"])
        if idx is None:
            continue
        papers.append(rec)
        for author in rec.get("authors", []):
            aid = author.get("author_id")
            if not aid:
                continue
            author_papers[aid].append(idx)
            author_names.setdefault(aid, author.get("name") or "")

    author_index = []
    for aid, idxs in tqdm(author_papers.items(), desc="build authors"):
        uniq = sorted(set(idxs), key=lambda i: (papers[i].get("year") or 0, papers[i]["paper_id"]))
        if len(uniq) < min_papers:
            continue
        author_index.append(
            {
                "author_id": aid,
                "name": author_names.get(aid, ""),
                "paper_idxs": uniq,
                "years": [int(papers[i]["year"]) for i in uniq],
            }
        )
    author_index.sort(key=lambda x: x["author_id"])
    qualified = [a["author_id"] for a in author_index]
    return author_index, qualified, papers


def rolling_examples(
    author_index: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    max_history: int,
    min_history: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for author in tqdm(author_index, desc="rolling examples"):
        by_year: dict[int, list[int]] = defaultdict(list)
        for idx in author["paper_idxs"]:
            by_year[int(papers[idx]["year"])].append(idx)

        for cutoff in range(2017, 2026):
            future_year = cutoff + 1
            future = sorted(by_year.get(future_year, []), key=lambda i: papers[i]["paper_id"])
            if not future:
                continue
            history = [
                idx
                for idx in author["paper_idxs"]
                if int(papers[idx]["year"]) <= cutoff
            ]
            if len(history) < min_history:
                continue
            row = {
                "author_id": author["author_id"],
                "cutoff_year": cutoff,
                "history_paper_idxs": history[-max_history:],
                "future_paper_idxs": future,
            }
            if future_year <= 2024:
                train.append(row)
            elif future_year == 2025:
                val.append(row)
            elif future_year == 2026:
                test.append(row)
    return train, val, test


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-papers", type=int, default=5)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    expected = [
        AUTHORS / "author_index.json",
        AUTHORS / "qualified_authors.json",
        AUTHORS / "q_with_noise.npy",
        AUTHORS / "train_examples.jsonl",
        AUTHORS / "val_examples.jsonl",
        AUTHORS / "test_examples.jsonl",
    ]
    if all(p.exists() for p in expected) and not args.force:
        print("[skip] author dataset exists; use --force to rebuild")
        return 0

    q_path, q_dim = build_q_with_noise(args.force)
    author_index, qualified, papers = build_author_index(args.min_papers)
    train, val, test = rolling_examples(author_index, papers, args.max_history, args.min_history)

    save_json(AUTHORS / "author_index.json", author_index)
    save_json(AUTHORS / "qualified_authors.json", qualified)
    save_json(
        AUTHORS / "dataset_meta.json",
        {
            "min_papers": args.min_papers,
            "min_history": args.min_history,
            "max_history": args.max_history,
            "n_authors_qualified": len(qualified),
            "q_with_noise_path": str(q_path),
            "q_dim": q_dim,
            "n_train_examples": len(train),
            "n_val_examples": len(val),
            "n_test_examples": len(test),
        },
    )
    save_jsonl(AUTHORS / "train_examples.jsonl", train)
    save_jsonl(AUTHORS / "val_examples.jsonl", val)
    save_jsonl(AUTHORS / "test_examples.jsonl", test)
    print(
        f"[done] authors={len(qualified)} train={len(train)} "
        f"val={len(val)} test={len(test)} q_dim={q_dim}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
