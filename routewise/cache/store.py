"""Semantic cache storage: FAISS for vector search, SQLite for entries.

SQLite is the single source of truth (it stores each entry's embedding too); the FAISS
index is an in-memory acceleration structure rebuilt from SQLite at startup. That avoids
the classic bug where an index file and a database drift out of sync after a crash.

Entries are partitioned by a "namespace" (hash of system prompt + model + params), with
one FAISS index per namespace, so a pirate-persona answer is never served to a request
with a lawyer-persona system prompt.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import faiss
import numpy as np


@dataclass
class CacheEntry:
    id: int
    namespace: str
    query: str
    answer: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    route: str | None
    created_at: float
    expires_at: float


class SemanticCacheStore(Protocol):
    def add(self, namespace: str, query: str, embedding: np.ndarray, answer: str, model: str, provider: str,
            prompt_tokens: int, completion_tokens: int, route: str | None, ttl_s: float) -> int: ...

    def search(self, namespace: str, embedding: np.ndarray, k: int) -> list[tuple[CacheEntry, float]]: ...

    def __len__(self) -> int: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS semantic_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    query TEXT NOT NULL,
    answer TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    route TEXT,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_semantic_ns ON semantic_entries(namespace);
"""

_COLS = "id, namespace, query, answer, model, provider, prompt_tokens, completion_tokens, route, created_at, expires_at"


class FaissSqliteStore:
    def __init__(self, db_path: str | Path, dim: int, clock=time.time) -> None:
        self.dim = dim
        self._clock = clock
        self._lock = threading.Lock()  # FAISS + sqlite3 are used from worker threads
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._indexes: dict[str, faiss.IndexIDMap2] = {}
        self._rebuild()

    def _new_index(self) -> faiss.IndexIDMap2:
        # Exact (brute-force) inner-product search. For a cache of <~1M entries this is
        # fast enough on CPU and has perfect recall; approximate indexes (IVF/HNSW) only
        # pay off at larger scale.
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))

    def _rebuild(self) -> None:
        now = self._clock()
        with self._lock:
            self._db.execute("DELETE FROM semantic_entries WHERE expires_at <= ?", (now,))
            self._db.commit()
            rows = self._db.execute("SELECT id, namespace, embedding FROM semantic_entries").fetchall()
            self._indexes.clear()
            for row_id, ns, blob in rows:
                vec = np.frombuffer(blob, dtype=np.float32).reshape(1, -1)
                self._indexes.setdefault(ns, self._new_index()).add_with_ids(vec, np.array([row_id], dtype=np.int64))

    def add(self, namespace: str, query: str, embedding: np.ndarray, answer: str, model: str, provider: str,
            prompt_tokens: int, completion_tokens: int, route: str | None, ttl_s: float) -> int:
        vec = np.ascontiguousarray(embedding, dtype=np.float32).reshape(1, self.dim)
        now = self._clock()
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO semantic_entries (namespace, query, answer, model, provider, prompt_tokens, "
                "completion_tokens, route, created_at, expires_at, embedding) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (namespace, query, answer, model, provider, prompt_tokens, completion_tokens, route,
                 now, now + ttl_s, vec.tobytes()),
            )
            self._db.commit()
            row_id = int(cur.lastrowid)
            self._indexes.setdefault(namespace, self._new_index()).add_with_ids(vec, np.array([row_id], dtype=np.int64))
        return row_id

    def search(self, namespace: str, embedding: np.ndarray, k: int) -> list[tuple[CacheEntry, float]]:
        vec = np.ascontiguousarray(embedding, dtype=np.float32).reshape(1, self.dim)
        now = self._clock()
        with self._lock:
            index = self._indexes.get(namespace)
            if index is None or index.ntotal == 0:
                return []
            sims, ids = index.search(vec, min(k, index.ntotal))
            found = [(int(i), float(s)) for i, s in zip(ids[0], sims[0]) if i != -1]
            if not found:
                return []
            placeholders = ",".join("?" * len(found))
            rows = self._db.execute(
                f"SELECT {_COLS} FROM semantic_entries WHERE id IN ({placeholders})", [i for i, _ in found]
            ).fetchall()
            entries = {r[0]: CacheEntry(*r) for r in rows}
            expired = [i for i, _ in found if i in entries and entries[i].expires_at <= now]
            if expired:
                # Lazy TTL expiry: drop stale entries when we stumble on them.
                index.remove_ids(np.array(expired, dtype=np.int64))
                self._db.execute(f"DELETE FROM semantic_entries WHERE id IN ({','.join('?' * len(expired))})", expired)
                self._db.commit()
        return [(entries[i], s) for i, s in found if i in entries and i not in expired]

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM semantic_entries")
            self._db.commit()
            self._indexes.clear()

    def __len__(self) -> int:
        with self._lock:
            return sum(ix.ntotal for ix in self._indexes.values())
