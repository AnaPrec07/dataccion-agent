"""
extractor.py
────────────
Pull all documents and their pre-computed embeddings out of a ChromaDB
collection so they can be fed directly into a topic model.

No new embeddings are generated here — the vectors that were produced by
``EmbeddingClient`` during ingestion are reused verbatim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

_DEFAULT_COLLECTION = "feminist_bot_rag"
_DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / "rag" / ".chroma_db"


@dataclass
class ChromaCorpus:
    """Container returned by :class:`ChromaExtractor.extract`."""

    texts: list[str]
    embeddings: np.ndarray          # shape (n_docs, embedding_dim)
    metadata: list[dict]
    ids: list[str]
    sources: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.sources = [m.get("source", "") for m in self.metadata]

    def __len__(self) -> int:
        return len(self.texts)


class ChromaExtractor:
    """
    Read every document and its embedding from a persistent ChromaDB
    collection.

    Args:
        persist_directory: Path to the ``.chroma_db`` directory used by
            :class:`~rag.vector_store.VectorStore`.
        collection_name: Name of the ChromaDB collection to read from.
    """

    def __init__(
        self,
        persist_directory: str | Path = _DEFAULT_PERSIST_DIR,
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> None:
        self._persist_directory = Path(persist_directory)
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(self._persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return self._collection.count()

    def extract(self, batch_size: int = 500) -> ChromaCorpus:
        """
        Retrieve all documents, embeddings, and metadata from ChromaDB.

        Args:
            batch_size: Number of records to fetch per ChromaDB ``get``
                call.  Reduce if you hit memory pressure on very large
                collections.

        Returns:
            A :class:`ChromaCorpus` ready to pass to
            :class:`~model.TopicModel`.

        Raises:
            ValueError: If the collection is empty.
        """
        total = self._collection.count()
        if total == 0:
            raise ValueError(
                f"Collection '{self._collection_name}' is empty. "
                "Run the RAG ingestion pipeline first."
            )

        logger.info(
            "Extracting %d documents from collection '%s'",
            total,
            self._collection_name,
        )

        all_texts: list[str] = []
        all_embeddings: list[list[float]] = []
        all_metadata: list[dict] = []
        all_ids: list[str] = []

        offset = 0
        while offset < total:
            batch = self._collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "embeddings", "metadatas"],
            )
            all_texts.extend(batch["documents"])
            all_embeddings.extend(batch["embeddings"])
            all_metadata.extend(batch["metadatas"])
            all_ids.extend(batch["ids"])
            offset += batch_size
            logger.debug("Fetched %d / %d", min(offset, total), total)

        corpus = ChromaCorpus(
            texts=all_texts,
            embeddings=np.array(all_embeddings, dtype=np.float32),
            metadata=all_metadata,
            ids=all_ids,
        )
        logger.info(
            "Corpus ready: %d documents, embedding dim=%d",
            len(corpus),
            corpus.embeddings.shape[1],
        )
        return corpus
