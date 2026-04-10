"""
RAG package — adapted for Databricks.

Uses Unity Catalog Volumes for PDF storage,
Databricks Foundation Model API for embeddings,
and ChromaDB for vector storage.
"""

from feminist_bot.rag.retriever import Retriever
from feminist_bot.rag.pdf_loader import PDFLoader
from feminist_bot.rag.embeddings import EmbeddingClient
from feminist_bot.rag.vector_store import VectorStore

__all__ = ["Retriever", "PDFLoader", "EmbeddingClient", "VectorStore"]
