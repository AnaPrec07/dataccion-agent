"""
vector_store.py — ChromaDB vector store (same as original)
──────────────────────────────────────────────────────────
ChromaDB works on Databricks clusters. For production, replace with
Databricks Vector Search.

Persistence: Chunks and embeddings are stored in *persist_directory*.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import chromadb
from chromadb.config import Settings

from feminist_bot.rag.pdf_loader import Chunk

logger = logging.getLogger(__name__)

_DEFAULT_COLLECTION = "feminist_bot_rag"
_DEFAULT_PERSIST_DIR = "/tmp/feminist_bot_chroma_db"


class VectorStore:
    """
    Stores and retrieves Chunk objects by semantic similarity using ChromaDB.

    Args:
        persist_directory: Directory where ChromaDB writes its data.
        collection_name: ChromaDB collection to use.
    """

    def __init__(
        self,
        persist_directory: str | Path = _DEFAULT_PERSIST_DIR,
        collection_name: str = _DEFAULT_COLLECTION,
    ) -> None:
        self._persist_directory = Path(persist_directory)
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(self._persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore ready: collection='%s', path='%s' (%d documents stored)",
            collection_name,
            self._persist_directory,
            self._collection.count(),
        )

    @property
    def count(self) -> int:
        return self._collection.count()

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[list[float]],
    ) -> None:
        """Persist chunks with their pre-computed embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must have the same length"
            )

        ids = [_chunk_id(c) for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "page": c.page,
                "chunk_index": c.chunk_index,
                **{k: str(v) for k, v in c.metadata.items()},
            }
            for c in chunks
        ]

        existing_ids = set(self._collection.get(ids=ids, include=[])["ids"])
        new_mask = [_id not in existing_ids for _id in ids]
        if not any(new_mask):
            logger.info("All %d chunks already present — nothing to add", len(chunks))
            return

        new_ids = [i for i, keep in zip(ids, new_mask) if keep]
        new_docs = [d for d, keep in zip(documents, new_mask) if keep]
        new_metas = [m for m, keep in zip(metadatas, new_mask) if keep]
        new_embs = [e for e, keep in zip(embeddings, new_mask) if keep]

        self._collection.add(
            ids=new_ids,
            documents=new_docs,
            metadatas=new_metas,
            embeddings=new_embs,
        )
        logger.info(
            "Added %d new chunk(s) (skipped %d duplicates)",
            len(new_ids),
            len(chunks) - len(new_ids),
        )

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """Return the k most similar chunks to query_embedding."""
        query_kwargs: dict = dict(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            query_kwargs["where"] = where

        results = self._collection.query(**query_kwargs)

        output: list[dict] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append(
                {
                    "text": doc,
                    "source": meta.get("source", ""),
                    "page": int(meta.get("page", -1)),
                    "chunk_index": int(meta.get("chunk_index", -1)),
                    "distance": dist,
                }
            )
        return output

    def clear(self) -> None:
        """Delete all documents from the collection."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' cleared", self._collection_name)


def _chunk_id(chunk: Chunk) -> str:
    """Stable, deterministic ID for a chunk based on its provenance."""
    safe_source = chunk.source.replace("/", "__").replace(".", "_")
    return f"{safe_source}__p{chunk.page}__c{chunk.chunk_index}"
