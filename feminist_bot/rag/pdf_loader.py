"""
pdf_loader.py
─────────────
Handles everything between a GCS bucket and a list of text chunks ready
for embedding:
  1. List PDF blobs in a bucket/prefix.
  2. Stream each blob to memory (no temp files).
  3. Extract per-page text with pypdf.
  4. Split pages into overlapping chunks with a recursive character splitter.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from typing import Iterator

import pypdf
from google.cloud import storage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """Raw text of one PDF page plus its provenance."""

    source: str          # GCS blob name  (e.g. "forward_looking/report.pdf")
    page: int            # 0-indexed page number
    text: str


@dataclass
class Chunk:
    """A sub-page text slice ready to be embedded."""

    source: str
    page: int
    chunk_index: int     # position within the document
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Recursive character splitter (no external dependency)
# ---------------------------------------------------------------------------


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """
    Split *text* into chunks of at most *chunk_size* characters, with
    *chunk_overlap* characters of context carried into the next chunk.

    Tries separators in order (paragraph → sentence → word → character)
    so chunk boundaries align with natural language breaks when possible.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    sep = ""
    remaining = separators.copy()
    while remaining:
        candidate = remaining.pop(0)
        if candidate in text:
            sep = candidate
            break

    parts = text.split(sep) if sep else list(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for part in parts:
        part_len = len(part) + len(sep)
        if current_len + part_len > chunk_size and current:
            chunk_text = sep.join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)
            # Keep overlap: drop leading parts until we're within overlap budget
            while current and current_len > chunk_overlap:
                removed = current.pop(0)
                current_len -= len(removed) + len(sep)
        current.append(part)
        current_len += part_len

    if current:
        chunk_text = sep.join(current).strip()
        if chunk_text:
            chunks.append(chunk_text)

    return chunks


# ---------------------------------------------------------------------------
# PDFLoader
# ---------------------------------------------------------------------------


class PDFLoader:
    """
    Reads PDF files from a GCS bucket, extracts their text, and splits them
    into overlapping chunks.

    Args:
        bucket_name: GCS bucket that holds the PDFs.
        chunk_size: Maximum characters per chunk (default 800).
        chunk_overlap: Characters of context overlap between chunks (default 150).
        storage_client: Injected ``google.cloud.storage.Client``; created
            automatically when not supplied (uses ADC).
    """

    def __init__(
        self,
        bucket_name: str,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        storage_client: storage.Client | None = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._client = storage_client or storage.Client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_pdfs(self, prefix: str = "") -> list[str]:
        """Return the GCS blob names of every PDF under *prefix*."""
        bucket = self._client.bucket(self.bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        names = [b.name for b in blobs if b.name.lower().endswith(".pdf")]
        logger.info("Found %d PDF(s) under gs://%s/%s", len(names), self.bucket_name, prefix)
        return names

    def load_and_split(self, prefix: str = "") -> list[Chunk]:
        """
        Full pipeline for every PDF under *prefix*:
          GCS blob → raw bytes → pages → chunks.

        Returns a flat list of :class:`Chunk` objects.
        """
        all_chunks: list[Chunk] = []
        for blob_name in self.list_pdfs(prefix):
            try:
                docs = self._load_documents(blob_name)
                chunks = self._split_documents(docs)
                all_chunks.extend(chunks)
                logger.info("Loaded %d chunk(s) from '%s'", len(chunks), blob_name)
            except Exception:
                logger.exception("Failed to process '%s' — skipping", blob_name)
        return all_chunks

    def iter_chunks(self, prefix: str = "") -> Iterator[Chunk]:
        """Lazy version of :meth:`load_and_split` — yields one chunk at a time."""
        for blob_name in self.list_pdfs(prefix):
            try:
                docs = self._load_documents(blob_name)
                yield from self._split_documents(docs)
            except Exception:
                logger.exception("Failed to process '%s' — skipping", blob_name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_blob(self, blob_name: str) -> bytes:
        bucket = self._client.bucket(self.bucket_name)
        blob = bucket.blob(blob_name)
        data = blob.download_as_bytes()
        logger.debug("Downloaded %d bytes from '%s'", len(data), blob_name)
        return data

    def _load_documents(self, blob_name: str) -> list[Document]:
        """Download a blob and return one :class:`Document` per PDF page."""
        raw = self._download_blob(blob_name)
        reader = pypdf.PdfReader(io.BytesIO(raw))
        docs: list[Document] = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                docs.append(Document(source=blob_name, page=page_num, text=text))
        logger.debug("Extracted %d page(s) from '%s'", len(docs), blob_name)
        return docs

    def _split_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split each :class:`Document` into overlapping :class:`Chunk` objects."""
        chunks: list[Chunk] = []
        for doc in documents:
            pieces = _recursive_split(doc.text, self.chunk_size, self.chunk_overlap)
            for idx, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        source=doc.source,
                        page=doc.page,
                        chunk_index=idx,
                        text=piece,
                        metadata={"source": doc.source, "page": doc.page},
                    )
                )
        return chunks
