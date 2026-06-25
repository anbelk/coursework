from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from common.compat import AUTHORS, RANDOM_STATE, load_json

HISTORY_CUTOFF = 2024
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1


def split_author_ids_list(
    author_ids: list[str],
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    seed: int = RANDOM_STATE,
) -> tuple[list[str], list[str], list[str]]:
    ids = sorted(author_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    shuffled = [ids[i] for i in perm]
    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]
    return train, val, test


@lru_cache(maxsize=1)
def load_author_splits() -> dict[str, Any]:
    return load_json(AUTHORS / "author_splits.json")


def split_author_ids(split: str) -> list[str]:
    data = load_author_splits()
    return sorted(data[split])


def split_author_set(split: str) -> set[str]:
    return set(split_author_ids(split))


def history_cutoff() -> int:
    data = load_author_splits()
    return int(data.get("history_cutoff", HISTORY_CUTOFF))
