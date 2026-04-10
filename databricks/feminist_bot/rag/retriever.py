"""
retriever.py — Adapted for Databricks
──────────────────────────────────────
High-level orchestrator for the RAG pipeline.

Ingest path: UC Volume PDFs → PDFLoader → Chunks → EmbeddingClient → VectorStore
Query path:  user query → EmbeddingClient → VectorStore.similarity_search → context

Usage in a Databricks notebook:
    from feminist_bot.rag import Retriever

    retriever = Retriever(
        volume_path="/Volumes/workspace/dataccion/pdfs",
    )
    retriever.ingest(prefix="forward_looking/")
    context = retriever.retrieve_context("Cual es la brecha salarial en Mexico?")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from feminist_bot.rag.embeddings import EmbeddingClient
from feminist_bot.rag.pdf_loader import PDFLoader
from feminist_bot.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A single result returned by Retriever.retrieve."""
    text: str
    source: str
    page: int
    chunk_index: int
    distance: float


class Retriever:
    """
    End-to-end RAG retriever adapted for Databricks.

    Args:
        volume_path: UC Volume path containing PDF reports.
        embedding_model: Databricks Foundation Model API endpoint name.
        chunk_size: Maximum characters per text chunk.
        chunk_overlap: Character overlap between consecutive chunks.
        top_k: Default number of chunks to return per query.
        persist_directory: Where ChromaDB stores its index on disk.
    """

    def __init__(
        self,
        volume_path: str = "/Volumes/workspace/dataccion/pdfs",
        embedding_model: str = "databricks-bge-large-en",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 5,
        persist_directory: str = "/tmp/feminist_bot_chroma_db",
    ) -> None:
        self._top_k = top_k
        self._loader = PDFLoader(
            volume_path=volume_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._embedder = EmbeddingClient(model=embedding_model)
        kwargs = {"persist_directory": persist_directory} if persist_directory else {}
        self._store = VectorStore(**kwargs)

    def ingest(self, prefix: str = "", force: bool = False) -> int:
        """Load, embed, and store every PDF under *prefix*."""
        if force:
            logger.warning("force=True: clearing existing vector store before ingest")
            self._store.clear()

        before = self._store.count
        chunks = self._loader.load_and_split(prefix=prefix)

        if not chunks:
            logger.warning("No chunks found under prefix '%s'", prefix)
            return 0

        logger.info("Embedding %d chunk(s)...", len(chunks))
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_texts(texts)

        self._store.add_chunks(chunks, embeddings)
        added = self._store.count - before
        logger.info("Ingest complete — %d new chunk(s) stored", added)
        return added

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        source_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the k most relevant chunks for query."""
        if self._store.count == 0:
            logger.warning("Vector store is empty — call retriever.ingest() before querying")
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
        """Returns retrieved chunks joined as a single string for prompt inclusion."""
        chunks = self.retrieve(query, k=k, source_filter=source_filter)
        if not chunks:
            return ""

        parts = [
            f"[Source: {c.source}, page {c.page + 1}]\n{c.text}"
            for c in chunks
        ]
        return separator.join(parts)
