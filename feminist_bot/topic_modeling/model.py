"""
model.py
────────
BERTopic wrapper that consumes pre-computed ChromaDB embeddings.

BERTopic pipeline used here:
  1. UMAP  — reduce 768-dim Vertex AI embeddings to a lower-dim space.
  2. HDBSCAN — cluster the reduced embeddings.
  3. c-TF-IDF — represent each cluster as a ranked list of keywords.

Because embeddings are reused from ChromaDB, no external model is loaded
at runtime.  The ``language="multilingual"`` CountVectorizer setting keeps
Spanish stop-words out of the topic keywords.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP

if TYPE_CHECKING:
    from topic_modeling.extractor import ChromaCorpus

logger = logging.getLogger(__name__)

_SPANISH_STOPWORDS = [
    "de",
    "la",
    "el",
    "en",
    "y",
    "a",
    "los",
    "las",
    "del",
    "se",
    "que",
    "un",
    "una",
    "es",
    "por",
    "con",
    "para",
    "su",
    "sus",
    "al",
    "lo",
    "como",
    "más",
    "o",
    "pero",
    "si",
    "porque",
    "cuando",
    "también",
    "le",
    "ya",
    "entre",
    "era",
    "son",
    "dos",
    "puede",
    "esto",
    "esta",
    "este",
    "así",
    "bien",
    "sin",
    "sobre",
    "ser",
    "hay",
    "fue",
    "han",
    "sido",
    "tiene",
    "tienen",
    "hacia",
    "cada",
    "muy",
    "nos",
    "me",
    "mi",
    "tu",
    "te",
    "nos",
    "les",
    "otro",
    "estos",
    "estas",
    "ese",
    "esa",
    "esos",
    "esas",
    # common filler words that survive tokenization
    "the",
    "this",
    "these",
    "those",
    "also",
    "both",
]

_STOPWORDS = list(ENGLISH_STOP_WORDS) + _SPANISH_STOPWORDS


class TopicModel:
    """
    Fit and query a BERTopic model on a :class:`~extractor.ChromaCorpus`.

    Args:
        n_components: UMAP target dimensionality before clustering.
        n_neighbors: UMAP neighbourhood size (controls local vs global
            structure).
        min_cluster_size: HDBSCAN minimum cluster size.  Smaller values
            produce more fine-grained topics; larger values produce fewer,
            more general ones.
        min_topic_size: BERTopic minimum number of documents per topic.
        top_n_words: Number of keywords kept per topic.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 5,
        n_neighbors: int = 15,
        min_cluster_size: int = 5,
        min_topic_size: int = 10,
        top_n_words: int = 10,
        nr_topics: int | str | None = None,
        seed: int = 42,
    ) -> None:
        self._n_components = n_components
        self._seed = seed

        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric="cosine",
            random_state=seed,
            low_memory=False,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        vectorizer = CountVectorizer(
            stop_words=_STOPWORDS,
            ngram_range=(1, 2),
            min_df=2,
            token_pattern=r"(?u)\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]{3,}\b",
        )

        self._bertopic = BERTopic(
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            top_n_words=top_n_words,
            min_topic_size=min_topic_size,
            nr_topics=nr_topics,
            calculate_probabilities=True,
            verbose=True,
        )

        self._topics: list[int] | None = None
        self._probs: np.ndarray | None = None
        self._corpus: ChromaCorpus | None = None

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------

    def fit(self, corpus: ChromaCorpus) -> TopicModel:
        """
        Fit the BERTopic model on *corpus*.

        Pre-computed embeddings are passed directly so BERT inference is
        skipped entirely.

        Args:
            corpus: Output of :class:`~extractor.ChromaExtractor.extract`.

        Returns:
            ``self`` for chaining.
        """
        logger.info(
            "Fitting BERTopic on %d documents (embedding dim=%d)",
            len(corpus),
            corpus.embeddings.shape[1],
        )
        self._corpus = corpus
        self._topics, self._probs = self._bertopic.fit_transform(
            documents=corpus.texts,
            embeddings=corpus.embeddings,
        )
        n_topics = len(self._bertopic.get_topic_info()) - 1  # exclude -1
        logger.info("BERTopic found %d topics (excluding outliers)", n_topics)
        return self

    def transform(
        self, texts: list[str], embeddings: np.ndarray
    ) -> tuple[list[int], np.ndarray]:
        """
        Assign topics to new *texts* using their *embeddings*.

        Args:
            texts: Raw document strings.
            embeddings: Pre-computed vectors, shape ``(n, embedding_dim)``.

        Returns:
            ``(topics, probabilities)`` — same convention as BERTopic.
        """
        self._assert_fitted()
        return self._bertopic.transform(documents=texts, embeddings=embeddings)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def get_topic_info(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per topic.

        Columns: ``Topic``, ``Count``, ``Name``, ``Representation``,
        ``Representative_Docs``.
        """
        self._assert_fitted()
        return self._bertopic.get_topic_info()

    def get_topic_keywords(self, topic_id: int) -> list[tuple[str, float]]:
        """Return ``[(word, score), ...]`` for *topic_id*."""
        self._assert_fitted()
        return self._bertopic.get_topic(topic_id)

    def get_document_topics(self) -> pd.DataFrame:
        """
        Return a DataFrame mapping each document to its assigned topic.

        Columns: ``id``, ``source``, ``page``, ``chunk_index``,
        ``topic``, ``probability``.
        """
        self._assert_fitted()
        rows = []
        probs = self._probs
        for i, (doc_id, meta, topic) in enumerate(
            zip(
                self._corpus.ids,
                self._corpus.metadata,
                self._topics,
            )
        ):
            prob = float(probs[i].max()) if probs is not None else float("nan")
            rows.append(
                {
                    "id": doc_id,
                    "source": meta.get("source", ""),
                    "page": meta.get("page", -1),
                    "chunk_index": meta.get("chunk_index", -1),
                    "topic": topic,
                    "probability": prob,
                }
            )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """
        Save the fitted BERTopic model to *directory*.

        Args:
            directory: Target directory (created if it does not exist).
        """
        self._assert_fitted()
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._bertopic.save(str(path / "bertopic_model"))
        logger.info("Model saved to '%s'", path)

    @classmethod
    def load(cls, directory: str | Path) -> TopicModel:
        """
        Load a previously saved BERTopic model.

        Args:
            directory: Directory passed to :meth:`save`.

        Returns:
            A new :class:`TopicModel` instance with ``_bertopic`` loaded
            but ``_corpus`` set to ``None`` (re-extract if needed).
        """
        path = Path(directory)
        instance = cls.__new__(cls)
        instance._bertopic = BERTopic.load(str(path / "bertopic_model"))
        instance._topics = None
        instance._probs = None
        instance._corpus = None
        logger.info("Model loaded from '%s'", path)
        return instance

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def bertopic(self) -> BERTopic:
        """Direct access to the underlying BERTopic instance."""
        return self._bertopic

    def _assert_fitted(self) -> None:
        if self._topics is None:
            raise RuntimeError(
                "Model has not been fitted yet. Call fit() first."
            )
