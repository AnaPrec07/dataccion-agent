# Databricks notebook source

# COMMAND ----------

# MAGIC %md # Topic Modeling Pipeline — Dataccion
# MAGIC
# MAGIC BERTopic-based topic modeling on the RAG corpus.
# MAGIC Reuses pre-computed embeddings from ChromaDB (no additional model loading).
# MAGIC
# MAGIC **Pipeline:** ChromaDB → ChromaExtractor → BERTopic (UMAP + HDBSCAN) → Visualizations

# COMMAND ----------

# MAGIC %pip install chromadb bertopic hdbscan umap-learn plotly
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import os
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, "/Workspace/Users/kevinromerooviedo@gmail.com/dataccion")
logging.basicConfig(level=logging.INFO)

# COMMAND ----------

# MAGIC %md ## 1. Extract Corpus from ChromaDB

# COMMAND ----------

from dataclasses import dataclass, field
from pathlib import Path
import chromadb
from chromadb.config import Settings

CHROMA_PERSIST_DIR = "/tmp/feminist_bot_chroma_db"
COLLECTION_NAME = "feminist_bot_rag"
OUTPUT_DIR = "/Volumes/workspace/dataccion/raw_data/topic_modeling"


@dataclass
class ChromaCorpus:
    """Container for extracted corpus data."""
    texts: list[str]
    embeddings: np.ndarray
    metadata: list[dict]
    ids: list[str]
    sources: list[str] = field(init=False)

    def __post_init__(self):
        self.sources = [m.get("source", "") for m in self.metadata]

    def __len__(self):
        return len(self.texts)


def extract_corpus(persist_dir=CHROMA_PERSIST_DIR, collection_name=COLLECTION_NAME, batch_size=500):
    """Pull all documents and embeddings from ChromaDB."""
    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    total = collection.count()
    if total == 0:
        raise ValueError(
            f"Collection '{collection_name}' is empty. "
            "Run the RAG ingestion pipeline first (see chatbot_app notebook)."
        )

    print(f"Extracting {total} documents from '{collection_name}'...")

    all_texts, all_embeddings, all_metadata, all_ids = [], [], [], []
    offset = 0
    while offset < total:
        batch = collection.get(
            limit=batch_size, offset=offset,
            include=["documents", "embeddings", "metadatas"],
        )
        all_texts.extend(batch["documents"])
        all_embeddings.extend(batch["embeddings"])
        all_metadata.extend(batch["metadatas"])
        all_ids.extend(batch["ids"])
        offset += batch_size

    corpus = ChromaCorpus(
        texts=all_texts,
        embeddings=np.array(all_embeddings, dtype=np.float32),
        metadata=all_metadata,
        ids=all_ids,
    )
    print(f"Corpus ready: {len(corpus)} documents, embedding dim={corpus.embeddings.shape[1]}")
    return corpus

# COMMAND ----------

# Extract the corpus (requires prior ingestion via chatbot_app)
try:
    corpus = extract_corpus()
except ValueError as e:
    print(f"⚠ {e}")
    print("Skipping topic modeling — ingest PDFs first using the chatbot_app notebook.")
    dbutils.notebook.exit("No corpus available")

# COMMAND ----------

# MAGIC %md ## 2. Fit BERTopic Model

# COMMAND ----------

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP

# Spanish + English stopwords
SPANISH_STOPWORDS = [
    "de", "la", "el", "en", "y", "a", "los", "las", "del", "se", "que",
    "un", "una", "es", "por", "con", "para", "su", "sus", "al", "lo",
    "como", "más", "o", "pero", "si", "porque", "cuando", "también",
    "le", "ya", "entre", "era", "son", "dos", "puede", "esto", "esta",
    "este", "así", "bien", "sin", "sobre", "ser", "hay", "fue", "han",
    "sido", "tiene", "tienen", "hacia", "cada", "muy", "nos", "me",
    "mi", "tu", "te", "les", "otro", "estos", "estas", "ese", "esa",
    "esos", "esas", "the", "this", "these", "those", "also", "both",
]
ALL_STOPWORDS = list(ENGLISH_STOP_WORDS) + SPANISH_STOPWORDS

# Configure BERTopic components
umap_model = UMAP(n_neighbors=15, n_components=5, metric="cosine", random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=5, metric="euclidean", cluster_selection_method="eom", prediction_data=True)
vectorizer = CountVectorizer(
    stop_words=ALL_STOPWORDS,
    ngram_range=(1, 2),
    min_df=2,
    token_pattern=r"(?u)\b[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]{3,}\b",
)

topic_model = BERTopic(
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer,
    top_n_words=10,
    min_topic_size=5,
    calculate_probabilities=True,
    verbose=True,
)

# Fit using pre-computed embeddings
print(f"Fitting BERTopic on {len(corpus)} documents...")
topics, probs = topic_model.fit_transform(
    documents=corpus.texts,
    embeddings=corpus.embeddings,
)
n_topics = len(topic_model.get_topic_info()) - 1
print(f"Found {n_topics} topics (excluding outliers)")

# COMMAND ----------

# MAGIC %md ## 3. Topic Info & Document Assignments

# COMMAND ----------

topic_info = topic_model.get_topic_info()
display(topic_info)

# COMMAND ----------

# Document-topic assignments
doc_topics = pd.DataFrame({
    "id": corpus.ids,
    "source": [m.get("source", "") for m in corpus.metadata],
    "page": [m.get("page", -1) for m in corpus.metadata],
    "chunk_index": [m.get("chunk_index", -1) for m in corpus.metadata],
    "topic": topics,
    "probability": [float(p.max()) if p is not None else float("nan") for p in probs],
})
display(doc_topics.head(20))

# COMMAND ----------

# MAGIC %md ## 4. Visualizations

# COMMAND ----------

# Intertopic Distance Map
fig = topic_model.visualize_topics(top_n_topics=20)
fig.update_layout(title_text="Intertopic Distance Map")
fig.show()

# COMMAND ----------

# Topic Word Scores
fig = topic_model.visualize_barchart(top_n_topics=15, n_words=8)
fig.update_layout(title_text="Topic Word Scores (c-TF-IDF)")
fig.show()

# COMMAND ----------

# Documents Projection
fig = topic_model.visualize_documents(
    docs=corpus.texts,
    embeddings=corpus.embeddings,
)
fig.update_layout(title_text="Documents Projection (UMAP)")
fig.show()

# COMMAND ----------

# Topic Hierarchy
fig = topic_model.visualize_hierarchy()
fig.update_layout(title_text="Topic Hierarchy")
fig.show()

# COMMAND ----------

# Topic Similarity Heatmap
fig = topic_model.visualize_heatmap(top_n_topics=20)
fig.update_layout(title_text="Topic Similarity Heatmap")
fig.show()

# COMMAND ----------

# MAGIC %md ## 5. Save Results

# COMMAND ----------

# Save topic info and document assignments to the Volume
os.makedirs(OUTPUT_DIR, exist_ok=True)

topic_info.to_csv(f"{OUTPUT_DIR}/topic_info.csv", index=False)
doc_topics.to_csv(f"{OUTPUT_DIR}/document_topics.csv", index=False)

print(f"Results saved to {OUTPUT_DIR}")
print(f"  - topic_info.csv ({len(topic_info)} topics)")
print(f"  - document_topics.csv ({len(doc_topics)} documents)")
