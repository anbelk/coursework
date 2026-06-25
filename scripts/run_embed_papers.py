from __future__ import annotations

import argparse

from embeddings.embed_papers import main as embed_main


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper embeddings")
    parser.parse_args()
    return int(embed_main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
