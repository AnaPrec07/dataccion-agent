"""
visualizer.py
─────────────
Plotly-based visualizations for a fitted :class:`~model.TopicModel`.

All methods return ``plotly.graph_objects.Figure`` objects so they can be
embedded in Streamlit with ``st.plotly_chart(fig)`` or saved as
self-contained HTML files with ``fig.write_html(path)``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import plotly.graph_objects as go

if TYPE_CHECKING:
    from topic_modeling.model import TopicModel

logger = logging.getLogger(__name__)


class TopicVisualizer:
    """
    Generate visualizations from a fitted :class:`~model.TopicModel`.

    Args:
        model: A model on which :meth:`~model.TopicModel.fit` has been
            called.
    """

    def __init__(self, model: TopicModel) -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Individual figures
    # ------------------------------------------------------------------

    def intertopic_distance_map(
        self, top_n_topics: int = 20
    ) -> go.Figure:
        """
        2-D scatter of topic centroids sized by document count.

        Args:
            top_n_topics: How many topics to include (by document count).
        """
        fig = self._model.bertopic.visualize_topics(
            top_n_topics=top_n_topics
        )
        fig.update_layout(title_text="Intertopic Distance Map")
        return fig

    def topic_barchart(self, top_n_topics: int = 15, n_words: int = 8) -> go.Figure:
        """
        Horizontal bar chart of top words for each topic.

        Args:
            top_n_topics: Number of topics to show.
            n_words: Keywords per topic bar.
        """
        fig = self._model.bertopic.visualize_barchart(
            top_n_topics=top_n_topics,
            n_words=n_words,
        )
        fig.update_layout(title_text="Topic Word Scores (c-TF-IDF)")
        return fig

    def documents_projection(
        self,
        reduced_embeddings: object | None = None,
        sample: float | None = None,
        hide_annotations: bool = False,
    ) -> go.Figure:
        """
        2-D UMAP projection of every document, coloured by topic.

        Args:
            reduced_embeddings: Optional pre-computed 2-D coordinates.
                If ``None``, BERTopic re-runs UMAP at 2 components
                internally.
            sample: Fraction of documents to plot (0, 1] — useful when
                the corpus is large.
            hide_annotations: If ``True``, topic labels are not drawn on
                the scatter plot.
        """
        corpus = self._model._corpus
        if corpus is None:
            raise RuntimeError(
                "documents_projection requires the corpus to be attached "
                "to the model (fit the model, not just load it)."
            )
        fig = self._model.bertopic.visualize_documents(
            docs=corpus.texts,
            embeddings=corpus.embeddings,
            reduced_embeddings=reduced_embeddings,
            sample=sample,
            hide_annotations=hide_annotations,
        )
        fig.update_layout(title_text="Documents Projection (UMAP)")
        return fig

    def topic_hierarchy(self) -> go.Figure:
        """Dendrogram of hierarchical topic structure."""
        fig = self._model.bertopic.visualize_hierarchy()
        fig.update_layout(title_text="Topic Hierarchy")
        return fig

    def topic_heatmap(self, top_n_topics: int = 20) -> go.Figure:
        """
        Similarity heatmap between the top *top_n_topics* topics.

        Args:
            top_n_topics: Number of topics to include.
        """
        fig = self._model.bertopic.visualize_heatmap(
            top_n_topics=top_n_topics
        )
        fig.update_layout(title_text="Topic Similarity Heatmap")
        return fig

    # ------------------------------------------------------------------
    # Batch export
    # ------------------------------------------------------------------

    def save_all(
        self,
        output_dir: str | Path,
        top_n_topics: int = 20,
        n_words: int = 8,
    ) -> dict[str, Path]:
        """
        Render and save every visualization as a self-contained HTML file.

        Args:
            output_dir: Directory to write HTML files into (created if
                absent).
            top_n_topics: Passed to relevant visualizations.
            n_words: Keywords per topic for the bar chart.

        Returns:
            Mapping of ``{visualization_name: file_path}``.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        figures: dict[str, go.Figure] = {
            "intertopic_distance_map": self.intertopic_distance_map(
                top_n_topics=top_n_topics
            ),
            "topic_barchart": self.topic_barchart(
                top_n_topics=top_n_topics, n_words=n_words
            ),
            "topic_hierarchy": self.topic_hierarchy(),
            "topic_heatmap": self.topic_heatmap(
                top_n_topics=top_n_topics
            ),
        }

        if self._model._corpus is not None:
            figures["documents_projection"] = self.documents_projection()

        paths: dict[str, Path] = {}
        for name, fig in figures.items():
            dest = out / f"{name}.html"
            fig.write_html(str(dest))
            paths[name] = dest
            logger.info("Saved '%s' → %s", name, dest)

        return paths
