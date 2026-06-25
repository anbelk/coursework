"""Embed fine-topic meta-documents for broader NLP area clustering."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from common.compat import DATA, compact_abstract, load_json, save_json, topic_dir, variant_dir


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
META_DIR = DATA / "meta"
INSTRUCT = (
    "Represent the scientific topic cluster for semantic grouping into broader NLP "
    "research areas. Focus on the common research theme, underlying task, method "
    "family, model paradigm, data setting, and shared experimental direction across "
    "the papers in the cluster. Capture broader semantic relatedness between topic "
    "clusters, while preserving meaningful distinctions between different NLP subfields."
)


def build_input(cluster_document: str) -> str:
    return f"Instruct: {INSTRUCT}\nQuery: Cluster content: {cluster_document}"


def last_token_pool(last_hidden: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    bs = last_hidden.shape[0]
    return last_hidden[torch.arange(bs, device=last_hidden.device), seq_lens]


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_cluster_document(
    cluster_id: int,
    label: dict[str, str],
    terms: list[dict[str, Any]],
    reps: list[dict[str, Any]],
    rep_n: int,
    abstract_chars: int,
) -> str:
    term_text = ", ".join(str(item.get("term", "")) for item in terms[:12] if item.get("term"))
    chunks = [
        f"Topic name: {label.get('name', '')}",
        f"Description: {label.get('description', '')}",
        f"Top terms: {term_text}",
        "Representative papers:",
    ]
    for i, paper in enumerate(reps[:rep_n], start=1):
        chunks.append(
            f"{i}. Title: {paper.get('title', '')}\n"
            f"   Abstract: {compact_abstract(paper.get('abstract', ''), abstract_chars)}"
        )
    return "\n".join(chunks)


def load_meta_documents(rep_n: int, abstract_chars: int) -> list[dict[str, Any]]:
    sizes = load_json(variant_dir("hdbscan_fine") / "sizes.json")
    labels = load_json(topic_dir("hdbscan_fine") / "llm_label.json")
    terms = load_json(topic_dir("hdbscan_fine") / "top_terms.json")
    reps = load_json(topic_dir("hdbscan_fine") / "representative_papers.json")
    n_clusters = len(sizes)
    docs = []
    for cid in range(n_clusters):
        doc = build_cluster_document(
            cid,
            labels.get(str(cid), {}),
            terms.get(str(cid), []),
            reps.get(str(cid), []),
            rep_n,
            abstract_chars,
        )
        docs.append(
            {
                "cluster_id": cid,
                "paper_count": int(sizes.get(str(cid), 0)),
                "document": doc,
            }
        )
    return docs


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rep-n", type=int, default=12)
    parser.add_argument("--abstract-chars", type=int, default=600)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    META_DIR.mkdir(parents=True, exist_ok=True)
    out_emb = META_DIR / "cluster_embeddings.npy"
    out_docs = META_DIR / "cluster_documents.jsonl"
    if out_emb.exists() and out_docs.exists() and not args.force:
        print("[skip] meta cluster embeddings exist; use --force")
        return 0

    docs = load_meta_documents(args.rep_n, args.abstract_chars)
    save_jsonl(out_docs, docs)
    save_json(
        META_DIR / "cluster_embedding_meta.json",
        {
            "model": MODEL_NAME,
            "instruct": INSTRUCT,
            "max_length": args.max_length,
            "rep_n": args.rep_n,
            "abstract_chars": args.abstract_chars,
            "n_documents": len(docs),
        },
    )

    device = pick_device(args.device)
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    print(f"[info] device={device}, dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
    model = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.to(device)
    model.eval()

    texts = [build_input(row["document"]) for row in docs]
    enc_all = tokenizer(texts, padding=False, truncation=True, max_length=args.max_length)
    lengths = np.array([len(ids_) for ids_ in enc_all["input_ids"]])
    order = np.argsort(lengths, kind="stable")
    embeddings = np.empty((len(texts), model.config.hidden_size), dtype=np.float32)

    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), args.batch_size), desc="meta embeddings"):
            sel = order[start : start + args.batch_size]
            batch = tokenizer(
                [texts[i] for i in sel],
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
                embeddings[int(src_idx)] = emb_np[j]

    np.save(out_emb, embeddings)
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"[done] {out_emb} shape={embeddings.shape} norm_mean={norms.mean():.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
