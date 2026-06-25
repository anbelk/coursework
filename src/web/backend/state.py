from __future__ import annotations

import colorsys
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from common.compat import AUTHORS, DATA, load_json, load_paper_ids, load_papers_in_embedding_order, topic_dir, variant_dir


LAYOUT_DIR = DATA / "layout"
META_DIR = DATA / "meta"

CYR2LAT = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "i",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "iu",
    "я": "ia",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^0-9a-zA-Zа-яА-ЯёЁ]+", " ", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def cyr_to_lat(text: str) -> str:
    return "".join(CYR2LAT.get(ch, ch) for ch in text.lower())


def query_variants(text: str) -> set[str]:
    norm = normalize_text(text)
    variants = {norm}
    variants.add(cyr_to_lat(norm))
    return {v for v in variants if v}


def openalex_url(openalex_id: str) -> str:
    return f"https://openalex.org/{openalex_id}"


def short_title(title: str, max_chars: int = 78) -> str:
    title = re.sub(r"\s+", " ", (title or "").strip())
    if len(title) <= max_chars:
        return title
    return title[: max_chars - 1].rstrip() + "…"


def rgb_for_hsl(h: float, s: float = 0.68, l: float = 0.54) -> list[int]:
    r, g, b = colorsys.hls_to_rgb((h % 360.0) / 360.0, l, s)
    return [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]


class AppState:
    def __init__(self) -> None:
        self.paper_ids = load_paper_ids()
        self.papers = load_papers_in_embedding_order()
        self.paper_by_id = {paper["paper_id"]: paper for paper in self.papers}
        self.paper_pos = {pid: i for i, pid in enumerate(self.paper_ids)}
        self.xy = np.load(LAYOUT_DIR / "papers_xy.npy").astype(np.float32, copy=False)
        self.fine_labels = np.load(variant_dir("hdbscan_fine") / "labels.npy").astype(np.int32, copy=False)
        self.fine_topic_labels = load_json(topic_dir("hdbscan_fine") / "llm_label.json")
        self.fine_terms = load_json(topic_dir("hdbscan_fine") / "top_terms.json")
        self.fine_reps = load_json(topic_dir("hdbscan_fine") / "representative_papers.json")
        self.fine_sizes = load_json(variant_dir("hdbscan_fine") / "sizes.json")
        self.meta = load_json(META_DIR / "meta_assignments.json")
        self.meta_terms = load_json(topic_dir(self.meta["winner"]) / "top_terms.json")
        self.meta_reps = load_json(topic_dir(self.meta["winner"]) / "representative_papers.json")
        self.fine_to_meta = {int(k): int(v) for k, v in self.meta["fine_to_meta"].items()}
        self.meta_to_fine: dict[int, list[int]] = {}
        for fine_id, meta_id in self.fine_to_meta.items():
            self.meta_to_fine.setdefault(meta_id, []).append(fine_id)
        self.fine_to_papers = {
            cid: np.flatnonzero(self.fine_labels == cid).astype(np.int64)
            for cid in range(int(self.fine_labels.max()) + 1)
        }
        self.author_papers: dict[str, list[tuple[int, int]]] = {}
        self.colors = self.build_colors()
        self.authors = self.build_authors()
        self.author_ids = {row["author_id"] for row in self.authors}
        self.map_payload = self.build_map_payload()

    def build_colors(self) -> dict[int, list[int]]:
        colors: dict[int, list[int]] = {}
        meta_ids = sorted(self.meta_to_fine)
        meta_hue = {mid: (360.0 * i / max(1, len(meta_ids))) for i, mid in enumerate(meta_ids)}
        for meta_id in meta_ids:
            fine_ids = sorted(self.meta_to_fine[meta_id])
            center = (len(fine_ids) - 1) / 2.0
            for j, fine_id in enumerate(fine_ids):
                hue = meta_hue[meta_id] + (j - center) * 7.0
                colors[fine_id] = rgb_for_hsl(hue)
        return colors

    def fine_label(self, fine_id: int) -> str:
        item = self.fine_topic_labels.get(str(fine_id), {})
        return str(item.get("name", f"Fine topic {fine_id}"))

    def meta_label(self, meta_id: int) -> str:
        item = self.meta.get("meta_label", {}).get(str(meta_id), {})
        return str(item.get("name", f"Meta cluster {meta_id}"))

    def build_authors(self) -> list[dict[str, Any]]:
        history_index = load_json(DATA / "predictions" / "author_history_index.json")
        author_index = load_json(AUTHORS / "author_index.json")
        author_by_id = {row["author_id"]: row for row in author_index}
        out = []
        for author_id, hist in history_index.items():
            row = author_by_id.get(author_id, {})
            self.author_papers[author_id] = [
                (int(year), int(idx))
                for idx, year in zip(row.get("paper_idxs", hist.get("history_paper_idxs", [])), row.get("years", []), strict=False)
            ]
            name = str(hist.get("name") or row.get("name") or author_id)
            last_papers = [
                {
                    "paper_id": self.paper_ids[int(idx)],
                    "title": self.papers[int(idx)].get("title", ""),
                    "year": self.papers[int(idx)].get("year"),
                }
                for idx in hist.get("last5_paper_idxs", [])
            ]
            query = normalize_text(f"{name} {author_id}")
            query_field = f"{query} {cyr_to_lat(query)}"
            out.append(
                {
                    "author_id": author_id,
                    "display_name": name,
                    "n_papers": len(row.get("paper_idxs", hist.get("history_paper_idxs", []))),
                    "openalex_url": openalex_url(author_id),
                    "last_papers": last_papers,
                    "query": query_field,
                }
            )
        return out

    def fine_center(self, fine_id: int) -> tuple[float, float]:
        idxs = self.fine_to_papers.get(fine_id, np.array([], dtype=np.int64))
        if len(idxs) == 0:
            return 0.0, 0.0
        center = np.median(self.xy[idxs], axis=0)
        return float(center[0]), float(center[1])

    def meta_center(self, meta_id: int) -> tuple[float, float]:
        idxs = np.concatenate([self.fine_to_papers.get(fid, np.array([], dtype=np.int64)) for fid in self.meta_to_fine[meta_id]])
        if len(idxs) == 0:
            return 0.0, 0.0
        center = np.median(self.xy[idxs], axis=0)
        return float(center[0]), float(center[1])

    def build_map_payload(self) -> dict[str, Any]:
        papers = []
        for idx, paper_id in enumerate(self.paper_ids):
            fine_id = int(self.fine_labels[idx])
            if fine_id < 0:
                continue
            meta_id = self.fine_to_meta.get(fine_id, -1)
            papers.append(
                {
                    "paper_id": paper_id,
                    "title_short": short_title(self.papers[idx].get("title", "")),
                    "x": float(self.xy[idx, 0]),
                    "y": float(self.xy[idx, 1]),
                    "fine_id": fine_id,
                    "meta_id": meta_id,
                    "color": self.colors.get(fine_id, [120, 120, 120]),
                }
            )
        fine = []
        for fine_id in sorted(self.fine_to_meta):
            x, y = self.fine_center(fine_id)
            fine.append(
                {
                    "id": fine_id,
                    "label": self.fine_label(fine_id),
                    "x": x,
                    "y": y,
                    "paper_count": int(self.fine_sizes.get(str(fine_id), 0)),
                    "meta_id": self.fine_to_meta[fine_id],
                    "color": self.colors.get(fine_id, [120, 120, 120]),
                }
            )
        meta = []
        for meta_id in sorted(self.meta_to_fine):
            x, y = self.meta_center(meta_id)
            base_fine = self.meta_to_fine[meta_id][0]
            meta.append(
                {
                    "id": meta_id,
                    "label": self.meta_label(meta_id),
                    "x": x,
                    "y": y,
                    "paper_count": int(self.meta.get("meta_paper_count", {}).get(str(meta_id), 0)),
                    "color": self.colors.get(base_fine, [120, 120, 120]),
                }
            )
        return {"papers": papers, "fine": fine, "meta": meta}

    def paper_meta(self, paper_id: str) -> tuple[int, str, int, str]:
        idx = self.paper_pos[paper_id]
        fine_id = int(self.fine_labels[idx])
        meta_id = self.fine_to_meta.get(fine_id, -1)
        return fine_id, self.fine_label(fine_id), meta_id, self.meta_label(meta_id)


state = AppState()
