# Dataccion Agent

Chatbot especializado desarrollado para **ONU Mujer (UN Women) Latinoamerica** que responde preguntas sobre las barreras estructurales, culturales, legales y economicas que enfrentan las mujeres en los mercados laborales de America Latina.

El agente combina **Retrieval-Augmented Generation (RAG)** con **Topic Modeling** sobre un corpus de informes PDF almacenados en Google Cloud Storage.

## Arquitectura

```
Usuario (Streamlit)
       |
       v
  Gemini 2.5 Flash (Vertex AI)
       |
       |-- Function Calling --> retrieve_context (RAG)
       |                             |
       |                             v
       |                        ChromaDB (busqueda por similitud coseno)
       |
       |-- Function Calling --> list_reports (GCS)
       |
       v
  Respuesta con citas de fuentes
```

### Componentes principales

| Componente | Descripcion |
|------------|-------------|
| **RAG Pipeline** | Ingesta PDFs desde GCS, los divide en chunks con overlap, genera embeddings multilingues (Vertex AI) y los almacena en ChromaDB para recuperacion semantica |
| **Topic Modeling** | Reutiliza los embeddings de ChromaDB para descubrir topicos con BERTopic (UMAP + HDBSCAN + c-TF-IDF) y generar visualizaciones interactivas |
| **Agente Conversacional** | Interfaz Streamlit con Gemini 2.5 Flash que decide autonomamente cuando consultar la base de conocimiento o listar reportes via function calling |
| **Analisis Exploratorio** | Notebooks con analisis de datos del BID y Banco Mundial sobre segregacion laboral, horas trabajadas y sobreeducacion |

### Estructura del proyecto

```
dataccion-agent/
├── feminist_bot/
│   ├── app.py                  # Aplicacion Streamlit principal
│   ├── instructions.py         # System prompt del agente
│   ├── tools.py
│   ├── rag/
│   │   ├── pdf_loader.py       # Carga PDFs de GCS y chunking
│   │   ├── embeddings.py       # Embeddings via Vertex AI (text-multilingual-embedding-002)
│   │   ├── vector_store.py     # ChromaDB como vector store
│   │   └── retriever.py        # Orquestador del pipeline RAG
│   └── topic_modeling/
│       ├── extractor.py        # Extrae corpus desde ChromaDB
│       ├── model.py            # BERTopic (UMAP + HDBSCAN + c-TF-IDF)
│       ├── pipeline.py         # Pipeline end-to-end de topic modeling
│       └── visualizer.py       # Visualizaciones Plotly (HTML)
├── extraction/                 # Notebooks de analisis exploratorio
├── enable_apis.sh              # Script para habilitar APIs de GCP
└── requirements.txt
```

## Requisitos previos

- Python 3.10+
- Proyecto de GCP con las siguientes APIs habilitadas:
  - Vertex AI (`aiplatform.googleapis.com`)
  - Cloud Storage
  - Cloud Resource Manager
- Autenticacion configurada con Google Cloud:
  ```bash
  gcloud auth application-default login
  gcloud auth application-default set-quota-project <TU_PROJECT_ID>
  ```
- Un bucket de GCS con los reportes PDF organizados por categoria (`current_situation/`, `forward_looking/`, `proposal/`)

## Instalacion

```bash
# Clonar el repositorio
git clone <url-del-repo>
cd dataccion-agent

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Variables de entorno

Crear un archivo `.env` en la raiz del proyecto:

```env
BUCKET_NAME=nombre-de-tu-bucket-gcs
```

## Como ejecutar

### Levantar el chatbot

```bash
cd feminist_bot
streamlit run app.py
```

Esto abre la aplicacion en `http://localhost:8501`.

### Ejecutar con puerto personalizado

```bash
cd feminist_bot
streamlit run app.py --server.port 8080
```

### Habilitar las APIs de GCP (primera vez)

```bash
bash enable_apis.sh
```

## Stack tecnologico

- **LLM**: Gemini 2.5 Flash (Vertex AI)
- **Embeddings**: text-multilingual-embedding-002 (768 dims, ES/EN)
- **Vector Store**: ChromaDB (persistencia local, metrica coseno)
- **Topic Modeling**: BERTopic (UMAP + HDBSCAN + c-TF-IDF)
- **Frontend**: Streamlit
- **Cloud**: Google Cloud Platform (Storage + Vertex AI)
