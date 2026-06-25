from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from .paths import LLM_CACHE_PATH, ROOT

LLM_MODEL = "gpt-4.1-mini"
LLM_VERSION = "topic_eval_v1"


class LLMCache:
    def __init__(self, path: Path = LLM_CACHE_PATH, model: str = LLM_MODEL) -> None:
        load_dotenv(ROOT / ".env")
        self.path = path
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=60.0)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def key(self, system_prompt: str, user_prompt: str) -> str:
        payload = f"{LLM_VERSION}\n{self.model}\n{system_prompt}\n{user_prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def complete_json(self, system_prompt: str, user_prompt: str, max_retries: int = 6) -> dict[str, Any]:
        key = self.key(system_prompt, user_prompt)
        with self.lock:
            row = self.conn.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
        if row:
            return json.loads(row[0])

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0,
                )
                raw = response.choices[0].message.content or "{}"
                obj = json.loads(raw)
                with self.lock:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO cache(key, model, prompt, response, created_at) VALUES (?, ?, ?, ?, ?)",
                        (key, self.model, user_prompt, json.dumps(obj, ensure_ascii=False), time.strftime("%Y-%m-%dT%H:%M:%S")),
                    )
                    self.conn.commit()
                return obj
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                # TPM/RPM bursts under parallel workers: back off longer on 429.
                status = getattr(exc, "status_code", None)
                if status == 429:
                    time.sleep(min(60.0, 5.0 * (2**attempt)))
                else:
                    time.sleep(2**attempt)
        raise RuntimeError("unreachable")
