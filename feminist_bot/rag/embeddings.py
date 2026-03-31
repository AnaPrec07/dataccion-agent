"""
embeddings.py
─────────────
Thin wrapper around the Vertex AI Text Embeddings API accessed through the
``google-genai`` SDK (already used by the rest of the app).

Model: ``text-multilingual-embedding-002``
  • Supports Spanish and English — appropriate for this project.
  • Output dimension: 768.
  • Max tokens per request: 2 048.

The class batches texts automatically so callers never need to worry about
API request limits.
"""

from __future__ import annotations

import logging
from typing import Sequence

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Vertex AI multilingual model; swap to "text-embedding-004" for English-only
# with higher quality on longer texts.
_DEFAULT_MODEL = "text-multilingual-embedding-002"

# Vertex AI allows up to 250 texts per embed_content call.
_BATCH_SIZE = 100


class EmbeddingClient:
    """
    Generates dense vector embeddings via Vertex AI.

    Args:
        project_id: GCP project that has Vertex AI enabled.
        location: Vertex AI region (e.g. ``"us-central1"``).
        model: Embedding model name.
        genai_client: Injected ``google.genai.Client``; created automatically
            when not supplied.
    """

    def __init__(
        self,
        project_id: str,
        location: str = "us-east4",
        model: str = _DEFAULT_MODEL,
        genai_client: genai.Client | None = None,
    ) -> None:
        self.model = model
        self._client = genai_client or genai.Client(
            vertexai=True, project=project_id, location=location
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Embed a list of texts.

        Returns a list of float vectors in the same order as *texts*.
        Empty or whitespace-only strings are replaced by zero vectors so
        the caller always receives a vector per input.
        """
        if not texts:
            return []

        texts = [t if t.strip() else " " for t in texts]  # guard empty strings
        vectors: list[list[float]] = []

        for batch_start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[batch_start : batch_start + _BATCH_SIZE]
            batch_vectors = self._embed_batch(batch)
            vectors.extend(batch_vectors)
            logger.debug(
                "Embedded batch %d–%d (%d texts)",
                batch_start,
                batch_start + len(batch) - 1,
                len(batch),
            )

        return vectors

    def embed_query(self, query: str) -> list[float]:
        """Convenience method — embed a single query string."""
        return self.embed_texts([query])[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        contents = [types.Content(parts=[types.Part(text=t)]) for t in texts]
        response = self._client.models.embed_content(
            model=self.model,
            contents=contents,
        )
        return [embedding.values for embedding in response.embeddings]
