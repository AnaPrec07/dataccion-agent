"""
pipeline.py
───────────
End-to-end topic modelling pipeline:

  ChromaDB  →  ChromaExtractor  →  TopicModel  →  TopicVisualizer

Typical usage
─────────────
    from topic_modeling.pipeline import TopicModelingPipeline

    pipeline = TopicModelingPipeline()
    results = pipeline.run(save_model=True, save_visualizations=True)
    print(results["topic_info"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from topic_modeling.extractor import ChromaCorpus, ChromaExtractor
from topic_modeling.model import TopicModel
from topic_modeling.visualizer import TopicVisualizer

logger = logging.getLogger(__name__)

_DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / "rag" / ".chroma_db"
_DEFAULT_COLLECTION = "feminist_bot_rag"
_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


@dataclass
class PipelineResult:
    """Holds every artefact produced by :meth:`TopicModelingPipeline.run`."""

    corpus: ChromaCorpus
    model: TopicModel
    topic_info: pd.DataFrame
    document_topics: pd.DataFrame
    visualization_paths: dict[str, Path] = field(default_factory=dict)
    model_path: Path | None = None


class TopicModelingPipeline:
    """
    Orchestrate extraction, modelling, and visualization in a single call.

    Args:
        persist_directory: Path to the ChromaDB ``.chroma_db`` directory.
        collection_name: ChromaDB collection to read.
        output_dir: Root directory for saved models and HTML reports.
        n_components: UMAP dimensionality.
        n_neighbors: UMAP neighbourhood size.
        min_cluster_size: HDBSCAN minimum cluster size.
        min_topic_size: BERTopic minimum documents per topic.
        top_n_words: Keywords per topic.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        persist_directory: str | Path = _DEFAULT_PERSIST_DIR,
        collection_name: str = _DEFAULT_COLLECTION,
        output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
        n_components: int = 5,
        n_neighbors: int = 15,
        min_cluster_size: int = 5,
        min_topic_size: int = 5,
        top_n_words: int = 10,
        nr_topics: int | str | None = None,
        seed: int = 42,
    ) -> None:
        self._extractor = ChromaExtractor(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
        self._model = TopicModel(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_cluster_size=min_cluster_size,
            min_topic_size=min_topic_size,
            top_n_words=top_n_words,
            nr_topics=nr_topics,
            seed=seed,
        )
        self._output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        save_model: bool = False,
        save_visualizations: bool = False,
        top_n_topics: int = 20,
        n_words: int = 8,
    ) -> PipelineResult:
        """
        Run the full pipeline.

        Args:
            save_model: If ``True``, persist the fitted BERTopic model
                under ``output_dir/model/``.
            save_visualizations: If ``True``, write HTML charts to
                ``output_dir/visualizations/``.
            top_n_topics: Passed to the visualizer.
            n_words: Keywords per topic bar in the bar chart.

        Returns:
            A :class:`PipelineResult` with the corpus, model, topic info
            DataFrame, per-document topic assignments, and (optionally)
            paths to saved artefacts.
        """
        logger.info("=== Topic Modelling Pipeline START ===")

        # 1. Extract
        logger.info(
            "Step 1/3 — Extracting corpus from ChromaDB (%d docs)",
            self._extractor.count,
        )
        corpus = self._extractor.extract()

        # 2. Fit
        logger.info("Step 2/3 — Fitting BERTopic")
        self._model.fit(corpus)

        topic_info = self._model.get_topic_info()
        document_topics = self._model.get_document_topics()
        result = PipelineResult(
            corpus=corpus,
            model=self._model,
            topic_info=topic_info,
            document_topics=document_topics,
        )

        # 3. Persist artefacts
        if save_model:
            model_dir = self._output_dir / "model"
            self._model.save(model_dir)
            result.model_path = model_dir

        if save_visualizations:
            logger.info("Step 3/3 — Generating visualizations")
            viz = TopicVisualizer(self._model)
            viz_dir = self._output_dir / "visualizations"
            result.visualization_paths = viz.save_all(
                output_dir=viz_dir,
                top_n_topics=top_n_topics,
                n_words=n_words,
            )
        else:
            logger.info("Step 3/3 — Skipping visualizations")

        logger.info("=== Topic Modelling Pipeline END ===")
        return result

    @classmethod
    def load_model(
        cls,
        model_dir: str | Path,
        persist_directory: str | Path = _DEFAULT_PERSIST_DIR,
        collection_name: str = _DEFAULT_COLLECTION,
        output_dir: str | Path = _DEFAULT_OUTPUT_DIR,
    ) -> tuple[TopicModelingPipeline, TopicModel]:
        """
        Restore a previously saved model without re-fitting.

        Returns:
            ``(pipeline_instance, loaded_model)`` — the pipeline can be
            used to re-generate visualizations; the model can be queried
            directly.
        """
        pipeline = cls(
            persist_directory=persist_directory,
            collection_name=collection_name,
            output_dir=output_dir,
        )
        loaded = TopicModel.load(model_dir)
        pipeline._model = loaded
        return pipeline, loaded
