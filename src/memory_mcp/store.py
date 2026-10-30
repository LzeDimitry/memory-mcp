"""SQLite store: memories table + FTS5 + BLOB vectors with numpy cosine.

Vector search is a full scan over in-memory numpy arrays — fast for the
10-20k records a per-project database will realistically hold. If that
ever becomes the bottleneck, swapping in sqlite-vec is isolated here.
"""

import json
import os
import re
import sqlite3
import time

import numpy as np

from .config import TYPES

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL CHECK (type IN ('decision','fact','bug','style','invariant')),
    content TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    scope TEXT NOT NULL DEFAULT '',
    links TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active',          -- active | superseded
    superseded_by INTEGER,
    importance REAL NOT NULL DEFAULT 0.5,
    vectorized INTEGER NOT NULL DEFAULT 0,
    embedding BLOB,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_superseded_by ON memories(superseded_by);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags, scope,
    content='memories', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags, scope)
    VALUES (new.id, new.content, new.tags, new.scope);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, scope)
    VALUES ('delete', old.id, old.content, old.tags, old.scope);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, scope)
    VALUES ('delete', old.id, old.content, old.tags, old.scope);
    INSERT INTO memories_fts(rowid, content, tags, scope)
    VALUES (new.id, new.content, new.tags, new.scope);
END;
"""

_DECAYED_TYPES = ("fact", "bug")


def _row_to_dict(row: sqlite3.Row, include_meta: bool = False) -> dict:
    keys = set(row.keys())
    base = {
        "id": row["id"],
        "type": row["type"],
        "content": row["content"],
        "tags": json.loads(row["tags"]) if "tags" in keys else [],
        "scope": row["scope"] if "scope" in keys else "",
    }
    if include_meta:
        base.update(
            links=json.loads(row["links"]),
            status=row["status"],
            superseded_by=row["superseded_by"],
            importance=row["importance"],
            vectorized=bool(row["vectorized"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )
    return base


def rrf_merge(rank_lists: list[list[int]], k: int) -> dict[int, float]:
    """Reciprocal Rank Fusion: combines ranked id lists by position, so
    incomparable raw scores (BM25 vs cosine distance) never mix."""
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for pos, id_ in enumerate(ranks):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + pos + 1)
    return scores


class MemoryStore:
    def __init__(self, cfg, embedder) -> None:
        self.cfg = cfg
        self.embedder = embedder
        parent = os.path.dirname(os.path.abspath(cfg.db_path))
        os.makedirs(parent, exist_ok=True)
        self.db = sqlite3.connect(cfg.db_path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self._vec_cache: np.ndarray | None = None
        self._vec_ids: list[int] = []

    def close(self) -> None:
        self.db.close()

    # ---------------------------------------------------------------- save

    def save(
        self,
        content: str,
        type_: str,
        tags: list[str] | None = None,
        scope: str = "",
        links: list[int] | None = None,
        importance: float = 0.5,
        supersedes_id: int | None = None,
    ) -> dict:
        if type_ not in TYPES:
            return {"error": f"type must be one of {TYPES}"}
        now = time.time()
        vector = self.embedder.embed_documents([content])
        blob = (
            np.asarray(vector[0], dtype=np.float32).tobytes() if vector else None
        )

        cur = self.db.execute(
            "INSERT INTO memories (type, content, tags, scope, links, importance,"
            " vectorized, embedding, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                type_,
                content,
                json.dumps(tags or []),
                scope,
                json.dumps(links or []),
                importance,
                1 if vector else 0,
                blob,
                now,
                now,
            ),
        )
        new_id = cur.lastrowid

        superseded = None
        if supersedes_id is not None:
            upd = self.db.execute(
                "UPDATE memories SET status='superseded', superseded_by=?,"
                " updated_at=? WHERE id=? AND status='active'",
                (new_id, now, supersedes_id),
            )
            if upd.rowcount:
                row = self.db.execute(
                    "SELECT content FROM memories WHERE id=?", (supersedes_id,)
                ).fetchone()
                superseded = (
                    {"id": supersedes_id, "content": row["content"]} if row else None
                )

        self.db.commit()
        self._vec_cache = None

        result: dict = {
            "saved_id": new_id,
            "vectorized": bool(vector),
        }
        if superseded:
            result["superseded"] = superseded
        similar = self.find_similar(content, exclude_id=new_id)
        if similar:
            # never auto-merge on similarity alone: "we use lib X" and
            # "we no longer use lib X" are near-identical vectors. The
            # agent decides whether to keep both or supersede.
            result["similar"] = similar
            result["hint"] = (
                "A similar record exists. If the new record replaces it, "
                "re-save with supersedes_id=<id>; otherwise keep both."
            )
        return result

    def find_similar(self, content: str, exclude_id: int, limit: int = 3) -> list[dict]:
        # document-to-document comparison, NOT query-to-document: nomic
        # (and similar instruction-prefixed models) use separate vector
        # spaces for search_document vs search_query, so cross-space
        # similarity saturates far below the real duplicate threshold
        new_vec = None
        embedded = self.embedder.embed_documents([content])
        if embedded:
            new_vec = embedded[0]
        sims: dict[int, float] = {}
        if new_vec is not None:
            rows = self.db.execute(
                "SELECT id, type, content, embedding FROM memories"
                " WHERE status='active' AND id != ? AND embedding IS NOT NULL",
                (exclude_id,),
            ).fetchall()
            if rows:
                matrix = np.stack(
                    [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
                )
                q = np.asarray(new_vec, dtype=np.float32)
                scores = matrix @ q / (
                    np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-9
                )
                for row, score in zip(rows, scores):
                    if score >= self.cfg.similarity_warn_threshold:
                        sims[row["id"]] = float(score)
        # lexical overlap is an independent duplicate signal: negations like
        # "we use X" / "we no longer use X" are lexically near-identical even
        # when the embedder keeps their semantic distance low
        for id_ in self._fts_ids(content, limit=limit):
            if id_ != exclude_id:
                sims.setdefault(id_, 0.0)
        if not sims:
            return []
        ph = ",".join("?" * len(sims))
        rows = self.db.execute(
            f"SELECT id, type, content, embedding FROM memories WHERE id IN ({ph})",
            list(sims),
        ).fetchall()
        out = []
        for row in rows:
            sim = sims[row["id"]]
            if sim == 0.0 and row["embedding"] and new_vec is not None:
                v = np.frombuffer(row["embedding"], dtype=np.float32)
                q = np.asarray(new_vec, dtype=np.float32)
                sim = float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q) + 1e-9))
                sims[row["id"]] = sim
            out.append(
                {
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"],
                    "similarity": round(sim, 3),
                }
            )
        out.sort(key=lambda item: -item["similarity"])
        return out[:limit]

    # -------------------------------------------------------------- search

    def _vector_matrix(self) -> tuple[list[int], np.ndarray | None]:
        if self._vec_cache is None:
            rows = self.db.execute(
                "SELECT id, embedding FROM memories"
                " WHERE status='active' AND embedding IS NOT NULL"
            ).fetchall()
            self._vec_ids = [r["id"] for r in rows]
            self._vec_cache = (
                np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                if rows
                else np.zeros((0, self.cfg.embed_dim), dtype=np.float32)
            )
        return self._vec_ids, self._vec_cache

    def _fts_ids(self, fts_query: str, limit: int = 30) -> list[int]:
        # raw content breaks FTS5 syntax (punctuation, quotes) and search
        # operators (AND/OR/NOT) must stay literal: extract word tokens,
        # quote each, OR-join so any shared token counts as overlap
        tokens = re.findall(r"\w+", fts_query)
        if not tokens:
            return []
        match_expr = " OR ".join(f'"{t}"' for t in tokens[:32])
        try:
            rows = self.db.execute(
                "SELECT rowid FROM memories_fts WHERE memories_fts MATCH ?"
                " ORDER BY rank LIMIT ?",
                (match_expr, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r["rowid"] for r in rows]

    def _vector_ids(self, query_vec: list[float] | None, limit: int = 30) -> list[int]:
        if query_vec is None:
            return []
        ids, matrix = self._vector_matrix()
        if matrix.shape[0] == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        sims = matrix @ q / (
            np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-9
        )
        order = np.argsort(-sims)[:limit]
        return [ids[i] for i in order]

    def search(
        self,
        query: str,
        type_: str | None = None,
        tags: list[str] | None = None,
        scope: str | None = None,
    ) -> dict:
        now = time.time()
        query_vec = self.embedder.embed_query(query)
        fts_ids = self._fts_ids(query)
        vec_ids = self._vector_ids(query_vec)
        scores = rrf_merge([fts_ids, vec_ids], self.cfg.rrf_k)
        if not scores:
            return {"results": [], "vectorized": query_vec is not None}

        placeholders = ",".join("?" * len(scores))
        sql = f"SELECT * FROM memories WHERE id IN ({placeholders}) AND status='active'"
        params: list = list(scores)
        if type_:
            sql += " AND type=?"
            params.append(type_)
        if scope:
            sql += " AND scope LIKE ?"
            params.append(f"%{scope}%")
        rows = self.db.execute(sql, params).fetchall()

        results = []
        for row in rows:
            record = _row_to_dict(row)
            record["links"] = json.loads(row["links"])
            for tag in tags or []:
                if tag not in record["tags"]:
                    break
            else:
                score = scores[row["id"]]
                if row["type"] in _DECAYED_TYPES:
                    age_days = (now - row["created_at"]) / 86400
                    score *= 0.5 ** (age_days / self.cfg.decay_half_life_days)
                results.append((score, record))

        results.sort(key=lambda pair: -pair[0])
        # hard token budget: top-k AND a total character cap, so one search
        # can never flood the context
        out, used = [], 0
        for score, record in results[: self.cfg.search_top_k]:
            snippet = {"id": record["id"], "type": record["type"], "content": record["content"]}
            if record["scope"]:
                snippet["scope"] = record["scope"]
            if record["tags"]:
                snippet["tags"] = record["tags"]
            out.append(snippet)
            used += len(record["content"])
            if used >= self.cfg.search_char_budget:
                break
        return {"results": out, "vectorized": query_vec is not None}

    # ------------------------------------------------- get / list / delete

    def get(self, id_: int, include_superseded_chain: bool = True) -> dict | None:
        row = self.db.execute("SELECT * FROM memories WHERE id=?", (id_,)).fetchone()
        if row is None:
            return None
        record = _row_to_dict(row, include_meta=True)
        if include_superseded_chain and row["status"] == "superseded":
            pass  # superseded_by already in meta
        # linked records
        link_ids = record["links"]
        if link_ids:
            ph = ",".join("?" * len(link_ids))
            rows = self.db.execute(
                f"SELECT id, type, content FROM memories WHERE id IN ({ph})",
                link_ids,
            ).fetchall()
            record["linked_records"] = [_row_to_dict(r) for r in rows]
        # who replaced this one
        if record["status"] == "superseded" and record["superseded_by"]:
            r2 = self.db.execute(
                "SELECT id, type, content FROM memories WHERE id=?",
                (record["superseded_by"],),
            ).fetchone()
            if r2:
                record["replaced_by"] = _row_to_dict(r2)
        return record

    def list_records(
        self, type_: str | None = None, scope: str | None = None, limit: int = 50
    ) -> list[dict]:
        sql = "SELECT * FROM memories WHERE status='active'"
        params: list = []
        if type_:
            sql += " AND type=?"
            params.append(type_)
        if scope:
            sql += " AND scope LIKE ?"
            params.append(f"%{scope}%")
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        return [_row_to_dict(r) for r in self.db.execute(sql, params).fetchall()]

    def delete(self, id_: int) -> bool:
        cur = self.db.execute("DELETE FROM memories WHERE id=?", (id_,))
        self.db.commit()
        self._vec_cache = None
        return cur.rowcount > 0

    # -------------------------------------------------------------- export

    def export_markdown(self) -> str:
        rows = self.db.execute(
            "SELECT * FROM memories ORDER BY type, updated_at DESC"
        ).fetchall()
        lines = ["# Memory export\n"]
        current_type = None
        for r in rows:
            if r["type"] != current_type:
                current_type = r["type"]
                lines.append(f"\n## {current_type}\n")
            status = " (SUPERSEDED)" if r["status"] == "superseded" else ""
            lines.append(
                f"- [{r['id']}] {r['content']}{status}"
                f" `scope={r['scope'] or '-'}` `tags={r['tags']}`"
            )
        return "\n".join(lines) + "\n"
