"""
pdf_loader.py — Adapted for Databricks
───────────────────────────────────────
Reads PDFs from Unity Catalog Volumes instead of GCS.
Extracts per-page text with pypdf and splits into overlapping chunks.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pypdf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """Raw text of one PDF page plus its provenance."""
    source: str
    page: int
    text: str


@dataclass
class Chunk:
    """A sub-page text slice ready to be embedded."""
    source: str
    page: int
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Recursive character splitter
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
# PDFLoader — reads from Unity Catalog Volumes
# ---------------------------------------------------------------------------


class PDFLoader:
    """
    Reads PDF files from a Unity Catalog Volume, extracts text, and splits
    into overlapping chunks.

    Args:
        volume_path: UC Volume path (e.g. "/Volumes/workspace/dataccion/pdfs").
        chunk_size: Maximum characters per chunk (default 800).
        chunk_overlap: Characters of context overlap between chunks (default 150).
    """

    def __init__(
        self,
        volume_path: str,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ) -> None:
        self.volume_path = Path(volume_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def list_pdfs(self, prefix: str = "") -> list[str]:
        """Return paths of every PDF under the volume/prefix."""
        search_dir = self.volume_path / prefix if prefix else self.volume_path
        if not search_dir.exists():
            logger.warning("Directory does not exist: %s", search_dir)
            return []
        names = [str(p) for p in search_dir.rglob("*.pdf")]
        logger.info("Found %d PDF(s) under %s", len(names), search_dir)
        return names

    def load_and_split(self, prefix: str = "") -> list[Chunk]:
        """Full pipeline: Volume PDFs → pages → chunks."""
        all_chunks: list[Chunk] = []
        for pdf_path in self.list_pdfs(prefix):
            try:
                docs = self._load_documents(pdf_path)
                chunks = self._split_documents(docs)
                all_chunks.extend(chunks)
                logger.info("Loaded %d chunk(s) from '%s'", len(chunks), pdf_path)
            except Exception:
                logger.exception("Failed to process '%s' — skipping", pdf_path)
        return all_chunks

    def iter_chunks(self, prefix: str = "") -> Iterator[Chunk]:
        """Lazy version of load_and_split."""
        for pdf_path in self.list_pdfs(prefix):
            try:
                docs = self._load_documents(pdf_path)
                yield from self._split_documents(docs)
            except Exception:
                logger.exception("Failed to process '%s' — skipping", pdf_path)

    def _load_documents(self, pdf_path: str) -> list[Document]:
        """Read a PDF file and return one Document per page."""
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            docs: list[Document] = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if text:
                    docs.append(Document(source=Path(pdf_path).name, page=page_num, text=text))
        logger.debug("Extracted %d page(s) from '%s'", len(docs), pdf_path)
        return docs

    def _split_documents(self, documents: list[Document]) -> list[Chunk]:
        """Split each Document into overlapping Chunks."""
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
