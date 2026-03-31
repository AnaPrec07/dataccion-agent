"""
retriever.py
────────────
High-level orchestrator for the RAG pipeline.

Ingest path (run once, or when new PDFs arrive):
  GCS bucket → PDFLoader → Chunks → EmbeddingClient → VectorStore

Query path (called on every user message):
  user query → EmbeddingClient → VectorStore.similarity_search → context string

Typical usage in app.py
-----------------------
    from rag import Retriever

    retriever = Retriever(
        project_id=PROJECT_ID,
        bucket_name=os.getenv("BUCKET_NAME"),
    )
    # One-time ingestion (or call nightly / on-demand):
    retriever.ingest(prefix="forward_looking/")

    # At query time:
    context = retriever.retrieve_context("¿Cuál es la brecha salarial en México?")
    # Pass `context` as part of the system prompt or user turn.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from rag.embeddings import EmbeddingClient
from rag.pdf_loader import PDFLoader
from rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A single result returned by :meth:`Retriever.retrieve`."""

    text: str
    source: str
    page: int
    chunk_index: int
    distance: float


class Retriever:
    """
    End-to-end RAG retriever.

    Args:
        project_id: GCP project ID (must have Vertex AI and GCS enabled).
        bucket_name: GCS bucket containing the PDF reports.
        location: Vertex AI region for the embedding model.
        chunk_size: Maximum characters per text chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        top_k: Default number of chunks to return per query.
        persist_directory: Where ChromaDB stores its index on disk.
    """

    def __init__(
        self,
        project_id: str,
        bucket_name: str,
        location: str = "us-central1",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 5,
        persist_directory: str = "",
    ) -> None:
        self._top_k = top_k
        self._loader = PDFLoader(
            bucket_name=bucket_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._embedder = EmbeddingClient(
            project_id=project_id,
            location=location,
        )
        kwargs = (
            {"persist_directory": persist_directory}
            if persist_directory
            else {}
        )
        self._store = VectorStore(**kwargs)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, prefix: str = "", force: bool = False) -> int:
        """
        Load, embed, and store every PDF under *prefix* in the GCS bucket.

        Args:
            prefix: GCS object prefix to filter which PDFs to ingest
                (e.g. ``"forward_looking/"``).  Defaults to the entire bucket.
            force: When ``True``, wipe the vector store before ingesting so
                previously stored chunks are replaced.  Use this when PDFs
                have been updated.

        Returns:
            Number of new chunks added to the store.
        """
        if force:
            logger.warning(
                "force=True: clearing existing vector store before ingest"
            )
            self._store.clear()

        before = self._store.count
        chunks = self._loader.load_and_split(prefix=prefix)

        if not chunks:
            logger.warning("No chunks found under prefix '%s'", prefix)
            return 0

        logger.info("Embedding %d chunk(s)…", len(chunks))
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_texts(texts)

        self._store.add_chunks(chunks, embeddings)
        added = self._store.count - before
        logger.info("Ingest complete — %d new chunk(s) stored", added)
        return added

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Return the *k* most relevant chunks for *query*.

        Args:
            query: User's question (Spanish or English).
            k: Override the default ``top_k`` set at construction time.
            source_filter: Restrict results to a specific GCS blob name.

        Returns:
            Ordered list of :class:`RetrievedChunk` objects, most relevant first.
        """
        if self._store.count == 0:
            logger.warning(
                "Vector store is empty — call retriever.ingest() before querying"
            )
            return []

        query_vector = self._embedder.embed_query(query)
        where = {"source": {"$eq": source_filter}} if source_filter else None
        raw_results = self._store.similarity_search(
            query_embedding=query_vector,
            k=k or self._top_k,
            where=where,
        )
        return [RetrievedChunk(**r) for r in raw_results]

    def retrieve_context(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
        separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Convenience wrapper — returns retrieved chunks joined as a single
        string suitable for inclusion in a prompt.

        Each chunk is prefixed with its source and page number so the LLM
        can cite provenance.
        """
        chunks = self.retrieve(query, k=k, source_filter=source_filter)
        if not chunks:
            return ""

        parts = [
            f"[Source: {c.source}, page {c.page + 1}]\n{c.text}"
            for c in chunks
        ]
        return separator.join(parts)
