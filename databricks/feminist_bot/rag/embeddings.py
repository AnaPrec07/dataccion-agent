"""
embeddings.py — Adapted for Databricks
───────────────────────────────────────
Uses Databricks Foundation Model API for text embeddings via the
OpenAI-compatible endpoint.

Model: databricks-bge-large-en (768 dimensions)
  - Available on Databricks Free Edition via serving endpoints.
  - Supports English; for multilingual, use databricks-gte-large-en if available.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "databricks-bge-large-en"
_BATCH_SIZE = 100


class EmbeddingClient:
    """
    Generates dense vector embeddings via Databricks Foundation Model API.

    Args:
        model: Embedding model endpoint name.
        host: Databricks workspace host URL.
        token: Databricks personal access token.
    """

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str | None = None,
        token: str | None = None,
    ) -> None:
        self.model = model
        self._host = host or os.environ.get("DATABRICKS_HOST", "")
        self._token = token or os.environ.get("DATABRICKS_TOKEN", "")

        # Use OpenAI client for Databricks serving endpoint
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._token,
                base_url=f"{self._host.rstrip('/')}/serving-endpoints",
            )
        except ImportError:
            # Fallback: use requests directly
            self._client = None
            logger.warning("openai package not installed, using requests fallback")

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a list of texts. Returns vectors in same order."""
        if not texts:
            return []

        texts = [t if t.strip() else " " for t in texts]
        vectors: list[list[float]] = []

        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch = list(texts[batch_start: batch_start + _BATCH_SIZE])
            batch_vectors = self._embed_batch(batch)
            vectors.extend(batch_vectors)
            logger.debug(
                "Embedded batch %d-%d (%d texts)",
                batch_start,
                batch_start + len(batch) - 1,
                len(batch),
            )

        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._client is not None:
            response = self._client.embeddings.create(
                input=texts,
                model=self.model,
            )
            return [item.embedding for item in response.data]
        else:
            # Requests fallback
            import requests
            resp = requests.post(
                f"{self._host.rstrip('/')}/serving-endpoints/{self.model}/invocations",
                headers={"Authorization": f"Bearer {self._token}"},
                json={"input": texts},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
