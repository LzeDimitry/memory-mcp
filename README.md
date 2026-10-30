# memory-mcp

MCP server for long-term AI agent memory: atomic records, hybrid search
(SQLite FTS5 + vector embeddings via local Ollama), and a hard response
budget — so recalling one fact in a large project costs a few lines of
context instead of a full re-telling.

## Tools

| Tool | Purpose |
|---|---|
| `memory_save` | Store one atomic record (dedup warning, `supersedes_id`) |
| `memory_search` | Hybrid search (FTS + vectors, RRF fusion, decay for fact/bug, character budget) |
| `memory_get` | Full record + linked records + supersession chain |
| `memory_list` | Browse active records |
| `memory_delete` | Hard delete (prefer `supersedes_id` for outdated facts) |
| `memory_export` | Markdown export for sharing / backups |

Record types: `decision`, `fact`, `bug`, `style`, `invariant`.

## Installation

```bash
uv sync
# embedding model (optional — pulled automatically on first use):
ollama pull nomic-embed-text
```

Ollama is started automatically by the server when it is not running;
the embedding model is pulled automatically on first use.

## Connecting to an MCP client (ZCode etc.)

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["--directory", "/path/to/memory-mcp", "run", "memory-mcp"],
      "env": { "MEMORY_MCP_DB_PATH": "/path/to/project/.memory/memories.db" }
    }
  }
}
```

Omit `MEMORY_MCP_DB_PATH` to get a separate database per project
(`<cwd>/.memory/memories.db`).

### Codex CLI / IDE extension

Add to `~/.codex/config.toml` (shared by CLI and IDE extension):

```toml
[mcp_servers.memory]
command = "uv"
args = ["--directory", "/path/to/memory-mcp", "run", "memory-mcp"]
```

### Google Antigravity (IDE)

Agent panel → additional options (…) → **MCP Servers** → add, or edit
`~/.gemini/antigravity/mcp_config.json` directly. Known gotcha:
Antigravity rejects a `type` field — stdio is the default, so use the
snippet as-is, without `type`.

### Antigravity CLI

Edit `~/.gemini/antigravity-cli/mcp_config.json` with the same
`mcpServers` JSON block as above, or run `/mcp` inside the CLI and add
the server through the interactive MCP Manager. A project-local variant
lives at `.antigravitycli/mcp_config.json` inside the repository.

All clients share one database when pointed at the same
`MEMORY_MCP_DB_PATH`, or keep independent per-project databases when
it is omitted.

## Configuration (env)

| Variable | Default | Description |
|---|---|---|
| `MEMORY_MCP_DB_PATH` | `<cwd>/.memory/memories.db` | Per-project database file |
| `MEMORY_MCP_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama address |
| `MEMORY_MCP_EMBED_MODEL` | `nomic-embed-text` | Embedding model (`bge-m3` for higher quality) |
| `MEMORY_MCP_NO_EMBEDDINGS` | — | `1` = disable vectors, FTS only |

If Ollama cannot be started or reached, the server automatically falls
back to FTS-only mode.

## Making the agent use memory without being told every time

MCP tool descriptions are always visible to the agent, so it knows the
tools exist. To make it use them **proactively**, add this snippet to the
project's `AGENTS.md` (or system prompt) once — no repeated instructions
needed afterwards:

```markdown
# Project Memory

A persistent memory store is available via the `memory_*` MCP tools.

- Before starting non-trivial work, run `memory_search` with keywords for
  the task (e.g. `memory_search(query="auth token refresh")`). Do not ask
  the user about things that may already be recorded.
- When you learn a durable fact, make a decision, fix a bug, or establish
  a rule, save it immediately with `memory_save` — one atomic thought per
  record, type: decision | fact | bug | style | invariant.
- When a new record contradicts an old one, re-save with
  `supersedes_id=<old id>`. Never keep two conflicting active records.
- Use `memory_get(id)` for full context of a search hit; linked records
  and replacement history are included.
- Do not save transient details (file contents, conversation text) —
  only durable knowledge. Content is written in English.
```

## Design guarantees

- **No blind auto-merge:** `memory_save` returns a `similar` warning; the
  agent decides. Conflicts ("we use X" / "we no longer use X") are caught
  both by vector similarity and lexical overlap.
- **Supersede, not delete:** the old record stays in history
  (`status=superseded`, `superseded_by`) and disappears from search.
- **Response budget:** top-k results and a character cap — one search
  can never flood the context.

## Tests

```bash
uv run python -m pytest -q
```

Tests run without Ollama (deterministic fake embedder). Use
`python -m pytest` — the bare `pytest` script fails to import the
src-layout package under `uv run`.
