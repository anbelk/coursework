"""Build Qwen3-Embedding-0.6B embeddings for OpenAlex papers.

Reads JSONL with fields {paper_id, title, abstract, ...} and writes:
    --embeddings (default data/embeddings.npy)  : float32, shape (N, 1024), L2-normalized
    --ids        (default data/paper_ids.json)  : list[str], aligned with rows of the .npy

Prompt format (per task):
    Instruct: Represent the scientific paper title and abstract for fine-grained
    semantic clustering. Focus on the specific research problem, studied mechanism,
    method, architecture component, model type, data/task setting, and experimental
    focus. Distinguish papers from the same broad field if they investigate
    different techniques, components, mechanisms, or evaluation targets.
    Query: Title: {title}

    Abstract: {abstract}

Tokenization is left-padded and truncated to --max-length (default 512).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

INSTRUCT = (
    "Represent the scientific paper title and abstract for fine-grained semantic "
    "clustering. Focus on the specific research problem, studied mechanism, method, "
    "architecture component, model type, data/task setting, and experimental focus. "
    "Distinguish papers from the same broad field if they investigate different "
    "techniques, components, mechanisms, or evaluation targets."
)


def build_input(title: str, abstract: str) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    return (
        f"Instruct: {INSTRUCT}\n"
        f"Query: Title: {title}\n\n"
        f"Abstract: {abstract}"
    )


def build_input_plain(title: str, abstract: str) -> str:
    title = (title or "").strip()
    abstract = (abstract or "").strip()
    return f"Title: {title}\n\nAbstract: {abstract}"


def embed_sentence_transformers(
    records: list[dict],
    model_name: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    prefix: str = "query: ",
) -> tuple[np.ndarray, list[str]]:
    """Embed with a sentence-transformers model (fast, lightweight backend).

    e5 models require a task prefix on every text (e.g. "query: "); pass prefix=""
    for models that do not use one (e.g. bge-*-en).
    """
    from sentence_transformers import SentenceTransformer

    print(f"[info] loading sentence-transformers model {model_name} (prefix={prefix!r})")
    model = SentenceTransformer(model_name, device=str(device))
    model.max_seq_length = max_length
    texts = [prefix + build_input_plain(r["title"], r["abstract"]) for r in records]
    ids = [r["paper_id"] for r in records]
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32, copy=False)
    return emb, ids


def last_token_pool(last_hidden: Tensor, attention_mask: Tensor) -> Tensor:
    # При left-padding последний токен = последняя позиция
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    bs = last_hidden.shape[0]
    return last_hidden[torch.arange(bs, device=last_hidden.device), seq_lens]


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/openalex_clean.jsonl")
    parser.add_argument("--embeddings", default="data/embeddings.npy")
    parser.add_argument("--ids", default="data/paper_ids.json")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda / mps / cpu")
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Weights dtype",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Optional cap on number of papers (debug)",
    )
    parser.add_argument(
        "--backend", default="qwen", choices=["qwen", "st"],
        help="qwen = Qwen3-Embedding-0.6B; st = sentence-transformers model",
    )
    parser.add_argument(
        "--st-model", default="intfloat/multilingual-e5-base",
        help="sentence-transformers model name (backend=st)",
    )
    parser.add_argument(
        "--st-prefix", default="query: ",
        help="prefix prepended to each text (e5 models require 'query: ')",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    emb_path = Path(args.embeddings)
    ids_path = Path(args.ids)
    emb_path.parent.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)

    print(f"[info] reading {in_path}")
    records: list[dict] = []
    for rec in iter_jsonl(in_path):
        if not rec.get("title") or not rec.get("abstract"):
            continue
        records.append(rec)
        if args.limit and len(records) >= args.limit:
            break
    n = len(records)
    print(f"[info] papers to embed: {n}")

    if args.backend == "st":
        embeddings, ids = embed_sentence_transformers(
            records, args.st_model, device, args.batch_size, args.max_length,
            prefix=args.st_prefix,
        )
        print(f"[info] saving embeddings -> {emb_path} (shape={embeddings.shape})")
        np.save(emb_path, embeddings)
        with ids_path.open("w", encoding="utf-8") as f:
            json.dump(ids, f, ensure_ascii=False)
        print(f"[info] saving ids -> {ids_path} (n={len(ids)})")
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"[check] L2 norm: mean={norms.mean():.6f}, min={norms.min():.6f}, max={norms.max():.6f}")
        return 0

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32  # на CPU fp16 медленнее, чем fp32
    print(f"[info] device={device}, dtype={dtype}")

    print(f"[info] loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.to(device)
    model.eval()

    embed_dim = model.config.hidden_size
    print(f"[info] embedding dim = {embed_dim}")

    print("[info] tokenizing all inputs (truncated to max_length)")
    texts = [build_input(r["title"], r["abstract"]) for r in records]
    paper_ids_all = [r["paper_id"] for r in records]
    enc_all = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=args.max_length,
    )
    input_ids_all = enc_all["input_ids"]
    lengths = np.array([len(ids_) for ids_ in input_ids_all])
    print(
        f"[info] token length: mean={lengths.mean():.1f}, "
        f"p50={np.percentile(lengths, 50):.0f}, "
        f"p95={np.percentile(lengths, 95):.0f}, max={lengths.max()}"
    )

    # Сортируем по длине -> меньше padding в каждом батче
    order = np.argsort(lengths, kind="stable")
    inv_order = np.argsort(order, kind="stable")

    embeddings = np.empty((n, embed_dim), dtype=np.float32)
    ids = paper_ids_all  # порядок ids = исходный порядок records

    bs = args.batch_size
    pbar = tqdm(total=n, unit="paper")
    with torch.inference_mode():
        for start in range(0, n, bs):
            sel = order[start : start + bs]
            batch_texts = [texts[i] for i in sel]
            batch = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)

            out = model(**batch)
            emb = last_token_pool(out.last_hidden_state, batch["attention_mask"])
            emb = F.normalize(emb.float(), p=2, dim=1)
            emb_np = emb.cpu().numpy()
            for j, src_idx in enumerate(sel):
                embeddings[src_idx] = emb_np[j]
            pbar.update(len(sel))
    pbar.close()
    del inv_order  # not needed: we wrote into original positions directly

    print(f"[info] saving embeddings -> {emb_path} (shape={embeddings.shape})")
    np.save(emb_path, embeddings)
    with ids_path.open("w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)
    print(f"[info] saving ids -> {ids_path} (n={len(ids)})")

    norms = np.linalg.norm(embeddings, axis=1)
    print(
        f"[check] L2 norm: mean={norms.mean():.6f}, "
        f"min={norms.min():.6f}, max={norms.max():.6f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
