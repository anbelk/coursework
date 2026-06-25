from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any

import numpy as np

from common.compat import (
    LLMCache,
    PREDICTIONS,
    compact_abstract,
    load_json,
    load_papers_in_embedding_order,
)


RELEVANCE_SYSTEM_PROMPT = """You evaluate potential scientific collaboration relevance.
Return only valid JSON:
{
  "score": 0,
  "recommendation_type": "direct topical match",
  "reason": "brief reason"
}
Use this scale:
0 = irrelevant
1 = weak/general match: broad NLP/LLM similarity only
2 = relevant: concrete shared task, method, dataset type, evaluation problem, research question, or complementary expertise
3 = very relevant: strong concrete match or strong complementarity grounded in specific papers
Allowed recommendation_type values:
- direct topical match
- methodological match
- complementary expertise
- weak/general match
Do not use any hidden model score.
If the connection is only broad NLP/LLM similarity, score it 0 or 1 and set recommendation_type to "weak/general match".
Reject weak reasons such as "Both papers discuss NLP methodologies", "Both papers address challenges in LLMs", "Both works focus on language models", or "share strong interests in multilingual language models".
Prefer specific evidence: shared task, shared method, shared dataset type, shared evaluation problem, shared research question, or complementary expertise."""

EVIDENCE_SYSTEM_PROMPT = """You extract evidence for a potential scientific collaboration.
Return only valid JSON:
{
  "recommendation_type": "direct topical match",
  "links": [
    {"user_paper_idx": 1, "candidate_paper_idx": 1, "reason": "short reason"}
  ],
  "summary": "high-level explanation of shared interests or complementarity"
}
Allowed recommendation_type values:
- direct topical match
- methodological match
- complementary expertise
- weak/general match
Only use paper numbers that are present in the prompt.
Include at most 2 strongest article-to-article links.
Each reason must name a concrete shared task, method, dataset type, evaluation problem, research question, or complementary expertise.
If the connection is only broad NLP/LLM similarity, do not include it.
Do not use generic reasons like "both papers discuss NLP", "both address LLM challenges", "both focus on language models", or "shared interests in multilingual language models"."""

FINAL_SYSTEM_PROMPT = """You write concise scientific coauthor recommendations.
Return only valid JSON:
{
  "overview": "short overall summary",
  "recommendations": [
    {"candidate_id": "id", "recommendation_type": "direct topical match", "text": "connected explanation with markdown links"}
  ]
}
Allowed recommendation_type values:
- direct topical match
- methodological match
- complementary expertise
- weak/general match
Use the provided OpenAlex URLs as markdown links. Do not invent papers.
Only include the strongest concrete connections.
If a candidate has only broad NLP/LLM similarity, do not recommend them.
Avoid weak generic justifications such as "both papers discuss NLP methodologies", "both papers address challenges in LLMs", "both works focus on language models", or "share strong interests in multilingual language models"."""


def openalex_url(openalex_id: str) -> str:
    return f"https://openalex.org/{openalex_id}"


def normalize_score(value: Any) -> int:
    try:
        score = int(value)
    except Exception:
        score = 0
    return max(0, min(3, score))


def normalize_recommendation_type(value: Any) -> str:
    allowed = {
        "direct topical match",
        "methodological match",
        "complementary expertise",
        "weak/general match",
    }
    text = str(value or "").strip().lower()
    return text if text in allowed else "weak/general match"


def stable_shuffle(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    out = list(items)
    rng.shuffle(out)
    return out


@lru_cache(maxsize=1)
def resources() -> dict[str, Any]:
    papers = load_papers_in_embedding_order()
    author_ids = load_json(PREDICTIONS / "author_ids.json")
    history_index = load_json(PREDICTIONS / "author_history_index.json")
    pred = np.load(PREDICTIONS / "author_pred_emb.npy").astype(np.float32, copy=False)
    pred /= np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    author_pos = {aid: i for i, aid in enumerate(author_ids)}

    coauthors: dict[str, set[str]] = defaultdict(set)
    papers_by_author: dict[str, set[int]] = defaultdict(set)
    for idx, paper in enumerate(papers):
        aids = [a.get("author_id") for a in paper.get("authors", []) if a.get("author_id")]
        for aid in aids:
            papers_by_author[aid].add(idx)
            coauthors[aid].update(x for x in aids if x != aid)

    active_2026 = [
        aid
        for aid in author_ids
        if aid in history_index
        and any(int(papers[idx]["year"]) == 2026 for idx in papers_by_author.get(aid, set()))
    ]
    return {
        "papers": papers,
        "author_ids": author_ids,
        "history_index": history_index,
        "pred": pred,
        "author_pos": author_pos,
        "coauthors": coauthors,
        "papers_by_author": papers_by_author,
        "active_2026": active_2026,
    }


def author_name(author_id: str) -> str:
    return str(resources()["history_index"].get(author_id, {}).get("name", author_id))


def paper_records(paper_idxs: list[int], max_chars: int = 600) -> list[dict[str, Any]]:
    papers = resources()["papers"]
    records = []
    for n, idx in enumerate(paper_idxs[-5:], start=1):
        paper = papers[int(idx)]
        records.append(
            {
                "idx": n,
                "paper_idx": int(idx),
                "paper_id": paper["paper_id"],
                "title": paper.get("title", ""),
                "abstract": compact_abstract(paper.get("abstract", ""), max_chars),
                "url": openalex_url(paper["paper_id"]),
            }
        )
    return records


def paper_context(records: list[dict[str, Any]]) -> str:
    chunks = []
    for rec in records:
        chunks.append(
            f"{rec['idx']}. Paper id: {rec['paper_id']}\n"
            f"Title: {rec['title']}\n"
            f"Abstract: {rec['abstract']}\n"
            f"OpenAlex: {rec['url']}"
        )
    return "\n\n".join(chunks)


def author_recent_papers(author_id: str) -> list[dict[str, Any]]:
    hist = resources()["history_index"][author_id]
    return paper_records(hist["last5_paper_idxs"])


def stage1_dense_retrieval(user_id: str, top_n: int = 10) -> list[dict[str, Any]]:
    res = resources()
    if user_id not in res["author_pos"]:
        raise ValueError(f"unknown author_id: {user_id}")
    u_pos = res["author_pos"][user_id]
    all_author_set = set(res["author_ids"])
    excluded = {user_id} | (res["coauthors"].get(user_id, set()) & all_author_set)
    candidate_positions = [i for i, aid in enumerate(res["author_ids"]) if aid not in excluded]
    scores = res["pred"][candidate_positions] @ res["pred"][u_pos]
    top_local = np.argsort(-scores)[:top_n]

    candidates = []
    for rank, local in enumerate(top_local, start=1):
        pos = candidate_positions[int(local)]
        aid = res["author_ids"][pos]
        candidates.append(
            {
                "author_id": aid,
                "name": author_name(aid),
                "author_url": openalex_url(aid),
                "dense_rank": rank,
                "model_cosine": float(scores[int(local)]),
            }
        )
    return candidates


def build_relevance_prompt(user_id: str, candidate: dict[str, Any]) -> str:
    user_papers = author_recent_papers(user_id)
    cand_papers = author_recent_papers(candidate["author_id"])
    return f"""User author id: {user_id}
User name: {author_name(user_id)}

User recent papers:
{paper_context(user_papers)}

Candidate author id: {candidate["author_id"]}
Candidate name: {candidate["name"]}

Candidate recent papers:
{paper_context(cand_papers)}

Evaluate whether this candidate is a relevant potential collaborator for the user."""


def score_candidate(user_id: str, candidate: dict[str, Any], cache: LLMCache) -> dict[str, Any]:
    obj = cache.complete_json(RELEVANCE_SYSTEM_PROMPT, build_relevance_prompt(user_id, candidate))
    out = dict(candidate)
    out["llm_score"] = normalize_score(obj.get("score"))
    out["recommendation_type"] = normalize_recommendation_type(obj.get("recommendation_type"))
    out["llm_reason"] = str(obj.get("reason", "")).strip()
    return out


def stage2_relevance_scoring(
    user_id: str,
    candidates: list[dict[str, Any]],
    n_workers: int = 8,
    cache: LLMCache | None = None,
) -> list[dict[str, Any]]:
    cache = cache or LLMCache()
    scored = []
    shuffled = stable_shuffle(candidates, user_id)
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(score_candidate, user_id, cand, cache) for cand in shuffled]
        for future in as_completed(futures):
            scored.append(future.result())
    by_id = {row["author_id"]: row for row in scored}
    return [
        by_id.get(
            cand["author_id"],
            {**cand, "llm_score": 0, "recommendation_type": "weak/general match", "llm_reason": "missing rating"},
        )
        for cand in candidates
    ]


def stage3_select(scored: list[dict[str, Any]], top_k: int = 3) -> tuple[list[dict[str, Any]], bool]:
    selected = [
        row
        for row in scored
        if int(row.get("llm_score", 0)) == 3 and row.get("recommendation_type") != "weak/general match"
    ]
    if len(selected) < top_k:
        selected.extend(
            row
            for row in scored
            if int(row.get("llm_score", 0)) == 2 and row.get("recommendation_type") != "weak/general match"
        )
    selected = selected[:top_k]
    return selected, len(selected) >= top_k


def build_evidence_prompt(user_papers: list[dict[str, Any]], cand_papers: list[dict[str, Any]], candidate: dict[str, Any]) -> str:
    return f"""User recent papers:
{paper_context(user_papers)}

Candidate author id: {candidate["author_id"]}
Candidate name: {candidate["name"]}

Candidate recent papers:
{paper_context(cand_papers)}

Find only the strongest concrete article-to-article links that justify collaboration relevance.
If the evidence is only broad NLP/LLM similarity, return an empty links list and recommendation_type "weak/general match"."""


def stage4_evidence(
    user_papers: list[dict[str, Any]],
    cand_papers: list[dict[str, Any]],
    candidate: dict[str, Any],
    cache: LLMCache | None = None,
) -> dict[str, Any]:
    cache = cache or LLMCache()
    obj = cache.complete_json(EVIDENCE_SYSTEM_PROMPT, build_evidence_prompt(user_papers, cand_papers, candidate))
    links = []
    for item in obj.get("links", []):
        try:
            user_idx = int(item.get("user_paper_idx"))
            cand_idx = int(item.get("candidate_paper_idx"))
        except Exception:
            continue
        user_rec = next((p for p in user_papers if int(p["idx"]) == user_idx), None)
        cand_rec = next((p for p in cand_papers if int(p["idx"]) == cand_idx), None)
        if not user_rec or not cand_rec:
            continue
        links.append(
            {
                "user_paper_idx": user_idx,
                "candidate_paper_idx": cand_idx,
                "user_paper_id": user_rec["paper_id"],
                "candidate_paper_id": cand_rec["paper_id"],
                "user_paper_title": user_rec["title"],
                "candidate_paper_title": cand_rec["title"],
                "user_url": user_rec["url"],
                "candidate_url": cand_rec["url"],
                "reason": str(item.get("reason", "")).strip(),
            }
        )
        if len(links) >= 2:
            break
    return {
        "candidate_id": candidate["author_id"],
        "recommendation_type": normalize_recommendation_type(obj.get("recommendation_type")),
        "summary": str(obj.get("summary", "")).strip(),
        "links": links,
    }


def build_final_prompt(user_id: str, selected: list[dict[str, Any]], evidences: list[dict[str, Any]]) -> str:
    evidence_by_id = {ev["candidate_id"]: ev for ev in evidences}
    blocks = []
    for cand in selected:
        ev = evidence_by_id.get(cand["author_id"], {"summary": "", "links": []})
        link_lines = []
        for link in ev.get("links", []):
            link_lines.append(
                f"- User article: {link.get('user_paper_title', '')} ({link['user_url']}) | "
                f"Candidate article: {link.get('candidate_paper_title', '')} ({link['candidate_url']}) | "
                f"Reason: {link['reason']}"
            )
        blocks.append(
            f"Candidate id: {cand['author_id']}\n"
            f"Candidate name: {cand['name']}\n"
            f"Candidate profile: {cand['author_url']}\n"
            f"Recommendation type: {ev.get('recommendation_type') or cand.get('recommendation_type', '')}\n"
            f"Short reason: {cand.get('llm_reason', '')}\n"
            f"Evidence summary: {ev.get('summary', '')}\n"
            f"Article links:\n{chr(10).join(link_lines)}"
        )
    return f"""User author id: {user_id}
User name: {author_name(user_id)}

Selected recommendation evidence:
{chr(10).join(blocks)}

Write the final recommendation text. Include candidate profile links and cited OpenAlex article links."""


def stage5_final_answer(
    user_id: str,
    selected: list[dict[str, Any]],
    evidences: list[dict[str, Any]],
    cache: LLMCache | None = None,
) -> dict[str, Any]:
    if not selected:
        return {
            "overview": "Достаточно релевантные потенциальные соавторы не найдены.",
            "recommendations": [],
            "message": "Достаточно релевантные потенциальные соавторы не найдены.",
        }
    cache = cache or LLMCache()
    obj = cache.complete_json(FINAL_SYSTEM_PROMPT, build_final_prompt(user_id, selected, evidences))
    recommendations = obj.get("recommendations", [])
    overview = str(obj.get("overview", "")).strip()
    message = overview
    if recommendations:
        message = overview + "\n\n" + "\n\n".join(str(item.get("text", "")) for item in recommendations)
    return {"overview": overview, "recommendations": recommendations, "message": message.strip()}


def run_pipeline(user_id: str, top_n: int = 10, top_k: int = 3, n_workers: int = 8) -> dict[str, Any]:
    cache = LLMCache()
    candidates = stage1_dense_retrieval(user_id, top_n=top_n)
    scored = stage2_relevance_scoring(user_id, candidates, n_workers=n_workers, cache=cache)
    selected, has_enough = stage3_select(scored, top_k=top_k)
    if not selected:
        final = stage5_final_answer(user_id, selected, [], cache=cache)
        return {
            "user_id": user_id,
            "user_name": author_name(user_id),
            "candidates": scored,
            "selected": [],
            "has_enough": False,
            "evidence": [],
            **final,
        }

    user_papers = author_recent_papers(user_id)
    evidences = []
    with ThreadPoolExecutor(max_workers=min(n_workers, len(selected))) as executor:
        futures = [
            executor.submit(stage4_evidence, user_papers, author_recent_papers(cand["author_id"]), cand, cache)
            for cand in selected
        ]
        for future in as_completed(futures):
            evidences.append(future.result())
    evidence_by_id = {ev["candidate_id"]: ev for ev in evidences}
    selected = [
        cand
        for cand in selected
        if cand["author_id"] in evidence_by_id
        and evidence_by_id[cand["author_id"]].get("recommendation_type") != "weak/general match"
        and evidence_by_id[cand["author_id"]].get("links")
    ]
    has_enough = len(selected) >= top_k
    evidences = [evidence_by_id[cand["author_id"]] for cand in selected]
    final = stage5_final_answer(user_id, selected, evidences, cache=cache)
    return {
        "user_id": user_id,
        "user_name": author_name(user_id),
        "candidates": scored,
        "selected": selected,
        "has_enough": has_enough,
        "evidence": evidences,
        **final,
    }


def graded_ndcg(scores: list[int]) -> float:
    if not scores:
        return 0.0
    gains = np.array([2**int(s) - 1 for s in scores], dtype=np.float64)
    discounts = 1.0 / np.log2(np.arange(2, len(scores) + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.sort(gains)[::-1]
    idcg = float((ideal * discounts).sum())
    return dcg / idcg if idcg else 0.0


def strip_markdown(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
