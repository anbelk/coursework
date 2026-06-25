from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from common.author_splits import (
    HISTORY_CUTOFF,
    TRAIN_FRAC,
    VAL_FRAC,
    split_author_ids_list,
)
from common.compat import (
    AUTHORS,
    CLUSTERING,
    PAPERS_PATH,
    RANDOM_STATE,
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


def example_for_author(
    author: dict[str, Any],
    papers: list[dict[str, Any]],
    cutoff: int,
    max_history: int,
    min_history: int,
) -> dict[str, Any] | None:
    history = [
        int(idx)
        for idx in author["paper_idxs"]
        if int(papers[int(idx)]["year"]) <= cutoff
    ]
    future = [
        int(idx)
        for idx in author["paper_idxs"]
        if int(papers[int(idx)]["year"]) > cutoff
    ]
    if len(history) < min_history or not future:
        return None
    return {
        "author_id": author["author_id"],
        "cutoff_year": cutoff,
        "history_paper_idxs": history[-max_history:],
        "future_paper_idxs": future,
    }


def build_split_examples(
    author_index: list[dict[str, Any]],
    papers: list[dict[str, Any]],
    author_ids: list[str],
    cutoff: int,
    max_history: int,
    min_history: int,
) -> list[dict[str, Any]]:
    author_by_id = {a["author_id"]: a for a in author_index}
    examples: list[dict[str, Any]] = []
    for aid in author_ids:
        author = author_by_id.get(aid)
        if author is None:
            continue
        ex = example_for_author(author, papers, cutoff, max_history, min_history)
        if ex is not None:
            examples.append(ex)
    return examples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-papers", type=int, default=10)
    parser.add_argument("--min-history", type=int, default=10)
    parser.add_argument("--max-history", type=int, default=20)
    parser.add_argument("--history-cutoff", type=int, default=HISTORY_CUTOFF)
    parser.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    parser.add_argument("--val-frac", type=float, default=VAL_FRAC)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    expected = [
        AUTHORS / "author_index.json",
        AUTHORS / "qualified_authors.json",
        AUTHORS / "author_splits.json",
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
    train_ids, val_ids, test_ids = split_author_ids_list(
        qualified,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    cutoff = int(args.history_cutoff)
    train = build_split_examples(author_index, papers, train_ids, cutoff, args.max_history, args.min_history)
    val = build_split_examples(author_index, papers, val_ids, cutoff, args.max_history, args.min_history)
    test = build_split_examples(author_index, papers, test_ids, cutoff, args.max_history, args.min_history)

    save_json(AUTHORS / "author_index.json", author_index)
    save_json(AUTHORS / "qualified_authors.json", qualified)
    save_json(
        AUTHORS / "author_splits.json",
        {
            "history_cutoff": cutoff,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "test_frac": 1.0 - args.train_frac - args.val_frac,
            "random_state": args.seed,
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
            "n_train_authors": len(train_ids),
            "n_val_authors": len(val_ids),
            "n_test_authors": len(test_ids),
        },
    )
    save_json(
        AUTHORS / "dataset_meta.json",
        {
            "min_papers": args.min_papers,
            "min_history": args.min_history,
            "max_history": args.max_history,
            "history_cutoff": cutoff,
            "split_strategy": "author_first_then_temporal",
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "random_state": args.seed,
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
        f"[done] authors={len(qualified)} "
        f"split={len(train_ids)}/{len(val_ids)}/{len(test_ids)} "
        f"examples train={len(train)} val={len(val)} test={len(test)} "
        f"cutoff={cutoff} q_dim={q_dim}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
