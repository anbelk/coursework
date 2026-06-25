from __future__ import annotations

import argparse

from recommendation import predict_author_embeddings, train


def main() -> int:
    parser = argparse.ArgumentParser(description="Train author recommender and predict author embeddings")
    parser.parse_args()
    for name, fn in [("train", train.main), ("predict_author_embeddings", predict_author_embeddings.main)]:
        print(f"[step] {name}")
        code = fn()
        if code not in (None, 0):
            return int(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
