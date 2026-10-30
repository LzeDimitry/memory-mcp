"""Tests run without Ollama: a deterministic hashing embedder provides
vectors that are identical for identical text and similar for related text.
"""

import numpy as np
import pytest

from memory_mcp.config import Config
from memory_mcp.embedder import Embedder
from memory_mcp.store import MemoryStore


class HashEmbedder(Embedder):
    """Deterministic bag-of-words embeddings: same words -> same vector."""

    def __init__(self, cfg, dim=64) -> None:
        super().__init__(cfg)
        self.dim = dim
        self.available = True

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            v[hash(word) % self.dim] += 1.0
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class OutageEmbedder(HashEmbedder):
    """Simulates an Ollama outage: the first save stores no embedding."""

    def __init__(self, cfg, dim=64) -> None:
        super().__init__(cfg, dim)
        self.outage = True

    def embed_documents(self, texts):
        if self.outage:
            return None
        return super().embed_documents(texts)


@pytest.fixture
def store(tmp_path):
    cfg = Config(db_path=str(tmp_path / "test.db"))
    return MemoryStore(cfg, HashEmbedder(cfg))


def test_save_survives_rows_saved_during_outage(tmp_path):
    """Rows saved while the embedder was down get NULL embeddings; the next
    vectorized save must skip them in find_similar instead of crashing on
    np.frombuffer(None)."""
    cfg = Config(db_path=str(tmp_path / "test.db"))
    embedder = OutageEmbedder(cfg)
    store = MemoryStore(cfg, embedder)
    first = store.save("Auth rate limit is 10 per minute", "fact")
    assert first["vectorized"] is False

    embedder.outage = False
    second = store.save("Auth rate limit is 20 per minute", "fact")
    assert second["saved_id"] == first["saved_id"] + 1
    assert second["vectorized"] is True


def test_save_and_search_finds_exact_words(store):
    r = store.save("Auth uses JWT tokens stored in httpOnly cookies", "decision", scope="auth")
    assert r["saved_id"] == 1 and r["vectorized"] is True
    out = store.search("JWT tokens auth")
    assert out["results"]
    assert "JWT" in out["results"][0]["content"]


def test_supersede_archives_old_record(store):
    r1 = store.save("Use library X for hashing", "decision")
    r2 = store.save("Do NOT use library X for hashing", "decision", supersedes_id=r1["saved_id"])
    out = store.search("library X hashing")
    ids = [res["id"] for res in out["results"]]
    assert r1["saved_id"] not in ids
    assert r2["saved_id"] in ids
    old = store.get(r1["saved_id"])
    assert old["status"] == "superseded"
    assert old["superseded_by"] == r2["saved_id"]
    assert old["replaced_by"]["id"] == r2["saved_id"]


def test_conflicting_duplicate_raises_similar_warning(store):
    """Gemini's counter-example: negation produces near-identical vectors,
    so save must WARN instead of silently merging."""
    store.save("We use library X", "decision")
    r2 = store.save("We no longer use library X", "decision")
    assert r2["similar"], "conflicting duplicate must surface a warning"
    assert r2["similar"][0]["id"] == 1


def test_search_excludes_superseded(store):
    r1 = store.save("Cache TTL is 60 seconds", "fact")
    store.save("Cache TTL is 300 seconds", "fact", supersedes_id=r1["saved_id"])
    out = store.search("cache TTL")
    contents = [res["content"] for res in out["results"]]
    assert contents == ["Cache TTL is 300 seconds"]


def test_char_budget_limits_results(store):
    for i in range(20):
        store.save(f"Fact about module alpha number {i} " + "detail " * 100, "fact", scope="alpha")
    out = store.search("module alpha detail")
    total = sum(len(r["content"]) for r in out["results"])
    assert total <= store.cfg.search_char_budget + 200
    assert len(out["results"]) <= store.cfg.search_top_k


def test_filters_type_tags_scope(store):
    store.save("Lexer handles unicode escapes", "fact", tags=["parser"], scope="lexer")
    store.save("Auth rate limit is 10/min", "invariant", tags=["auth"], scope="auth")
    out = store.search("limit unicode", type_="invariant")
    assert [r["content"] for r in out["results"]] == ["Auth rate limit is 10/min"]
    out = store.search("unicode", scope="lexer")
    assert len(out["results"]) == 1


def test_get_links(store):
    r1 = store.save("DB schema v2 dropped column 'flags'", "fact")
    r2 = store.save("Migration 004 depends on schema v2", "decision", links=[r1["saved_id"]])
    got = store.get(r2["saved_id"])
    assert got["linked_records"][0]["id"] == r1["saved_id"]


def test_delete_and_list(store):
    r = store.save("Temp fact", "fact")
    assert store.delete(r["saved_id"]) is True
    assert store.get(r["saved_id"]) is None
    store.save("Keep me", "fact")
    assert [rec["content"] for rec in store.list_records()] == ["Keep me"]


def test_export_markdown(store):
    store.save("Use ruff for linting", "style", scope="tooling")
    store.save("Use ruff for linting v2", "style", scope="tooling", supersedes_id=1)
    text = store.export_markdown()
    assert "## style" in text and "SUPERSEDED" in text
