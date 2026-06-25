from __future__ import annotations

import re
import unicodedata

CYR2LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch", "ъ": "",
    "ы": "y", "ь": "", "э": "e", "ю": "iu", "я": "ia",
}


def compact_abstract(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


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
