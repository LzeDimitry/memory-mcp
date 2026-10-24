"""MCP server exposing the memory store as tools.

Tools:
- memory_save:    store an atomic record (dedup warning, supersedes_id)
- memory_search:  hybrid FTS+vector search, RRF-fused, token-budgeted
- memory_get:     full record + linked records + replacement chain
- memory_list:    browse active records
- memory_delete:  hard delete
- memory_export:  markdown export for sharing/backups
"""

import json
import os
import sys

from mcp.server.fastmcp import FastMCP

from .config import Config
from .embedder import Embedder
from .store import MemoryStore

mcp = FastMCP("memory")

_default_db = os.path.join(os.getcwd(), ".memory", "memories.db")
cfg = Config(db_path=_default_db)
embedder = Embedder(cfg)
store = MemoryStore(cfg, embedder)


def _j(data) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


@mcp.tool()
def memory_save(
    content: str,
    type: str,
    tags: list[str] | None = None,
    scope: str = "",
    links: list[int] | None = None,
    importance: float = 0.5,
    supersedes_id: int | None = None,
) -> str:
    """Save one atomic memory (one thought per record).

    type: decision | fact | bug | style | invariant.
    scope: module or area, e.g. "auth" or "parser/lexer".
    links: ids of related records.
    supersedes_id: id of an active record this one replaces (old one is
    archived, not deleted). If a similar record exists in the response,
    prefer superseding it over keeping duplicates.
    """
    result = store.save(
        content=content,
        type_=type,
        tags=tags,
        scope=scope,
        links=links,
        importance=importance,
        supersedes_id=supersedes_id,
    )
    return _j(result)


@mcp.tool()
def memory_search(
    query: str,
    type: str | None = None,
    tags: list[str] | None = None,
    scope: str | None = None,
) -> str:
    """Hybrid search (keywords + semantic). Returns compact top-k records
    within a character budget. Use filters to narrow: type, tags, scope."""
    return _j(store.search(query, type_=type, tags=tags, scope=scope))


@mcp.tool()
def memory_get(id: int) -> str:
    """Fetch a full record with metadata, linked records, and (if it was
    superseded) the record that replaced it."""
    record = store.get(id)
    return _j(record) if record else _j({"error": f"no record id={id}"})


@mcp.tool()
def memory_list(
    type: str | None = None, scope: str | None = None, limit: int = 50
) -> str:
    """Browse active records, newest first. For an overview, not for search."""
    return _j(store.list_records(type_=type, scope=scope, limit=limit))


@mcp.tool()
def memory_delete(id: int) -> str:
    """Permanently delete a record. Prefer memory_save(supersedes_id=...) for
    outdated facts — deletion loses history."""
    return _j({"deleted": store.delete(id), "id": id})


@mcp.tool()
def memory_export() -> str:
    """Export all records as markdown (human-readable, shareable)."""
    return store.export_markdown()


def main() -> None:
    os.makedirs(os.path.dirname(cfg.db_path), exist_ok=True)
    if not embedder.available:
        # try one warm-up call so the first real tool call doesn't stall
        embedder.embed_query("warmup")
        if not embedder.available:
            print(
                "memory-mcp: Ollama unreachable — running in FTS-only mode",
                file=sys.stderr,
            )
    mcp.run()


if __name__ == "__main__":
    main()
