"""Configuration resolved from environment variables."""

import os
from dataclasses import dataclass, field


# nomic-embed-text v1.5 instruction prefixes. bge-m3 and most other models
# need none of these; disabled automatically for any model other than nomic.
NOMIC_PREFIXES = {"document": "search_document: ", "query": "search_query: "}

TYPES = ("decision", "fact", "bug", "style", "invariant")


@dataclass
class Config:
    db_path: str
    ollama_url: str = "http://127.0.0.1:11434"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 768
    # search tuning
    search_top_k: int = 8
    search_char_budget: int = 2000
    rrf_k: int = 60
    similarity_warn_threshold: float = 0.92
    # exponential decay half-life in days, applied to fact/bug only
    decay_half_life_days: float = 90.0
    _prefixes: dict = field(default_factory=lambda: dict(NOMIC_PREFIXES))

    def __post_init__(self) -> None:
        self.db_path = os.environ.get("MEMORY_MCP_DB_PATH", self.db_path)
        self.ollama_url = os.environ.get(
            "MEMORY_MCP_OLLAMA_URL", self.ollama_url
        ).rstrip("/")
        self.embed_model = os.environ.get(
            "MEMORY_MCP_EMBED_MODEL", self.embed_model
        )
        if os.environ.get("MEMORY_MCP_NO_EMBEDDINGS") == "1":
            self._prefixes = {}
        elif "nomic" not in self.embed_model:
            # prefixes are a nomic-specific training detail
            self._prefixes = {}

    @property
    def prefixes(self) -> dict:
        return self._prefixes
