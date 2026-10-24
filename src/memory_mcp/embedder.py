"""Embeddings via a local Ollama instance.

Any failure (Ollama down, model missing, timeout) degrades to
`available = False` and the store falls back to FTS-only search. If the
Ollama server is not running, it is started automatically (autostart),
and the embedding model is pulled on first use if missing.
"""

import shutil
import subprocess
import sys
import time

import httpx


class Embedder:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.available = False
        self._server_ensured = False

    # ------------------------------------------------------------- server

    def _server_up(self) -> bool:
        try:
            httpx.get(f"{self.cfg.ollama_url}/api/version", timeout=2.0)
            return True
        except httpx.HTTPError:
            return False

    def _ensure_server(self) -> None:
        """Start Ollama automatically if it is not running, and pull the
        embedding model on first use if it is missing."""
        if self._server_ensured:
            return
        self._server_ensured = True
        if self._server_up():
            return
        if not shutil.which("ollama"):
            print("memory-mcp: ollama binary not found", file=sys.stderr)
            return
        print("memory-mcp: starting ollama serve", file=sys.stderr)
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(40):  # up to ~20s for the server to come up
            if self._server_up():
                break
            time.sleep(0.5)
        else:
            print("memory-mcp: ollama did not start in time", file=sys.stderr)
            return
        try:
            tags = httpx.get(f"{self.cfg.ollama_url}/api/tags", timeout=5.0).json()
            models = [m["name"] for m in tags.get("models", [])]
            if not any(m.startswith(self.cfg.embed_model) for m in models):
                print(
                    f"memory-mcp: pulling {self.cfg.embed_model} (first use)",
                    file=sys.stderr,
                )
                subprocess.run(
                    ["ollama", "pull", self.cfg.embed_model],
                    timeout=600,
                    check=False,
                )
        except (httpx.HTTPError, ValueError) as exc:
            print(f"memory-mcp: model check failed: {exc}", file=sys.stderr)

    # ----------------------------------------------------------- embed

    def _embed(self, inputs: list[str]) -> list[list[float]] | None:
        self._ensure_server()
        try:
            resp = httpx.post(
                f"{self.cfg.ollama_url}/api/embed",
                json={
                    "model": self.cfg.embed_model,
                    "input": inputs,
                    # keep the model resident so MCP calls never stall on
                    # cold-start model loading
                    "keep_alive": -1,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            embeddings = resp.json()["embeddings"]
            self.available = True
            return embeddings
        except (httpx.HTTPError, KeyError, ValueError):
            self.available = False
            return None

    def embed_documents(self, texts: list[str]) -> list[list[float]] | None:
        prefix = self.cfg.prefixes.get("document", "")
        return self._embed([prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float] | None:
        prefix = self.cfg.prefixes.get("query", "")
        result = self._embed([prefix + text])
        return result[0] if result else None
