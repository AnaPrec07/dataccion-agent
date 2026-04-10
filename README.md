# Dataccion Agent

Herramientas de asistente de investigación para **ONU Mujer (UN Women) América Latina**: un chatbot en Streamlit basado en investigación sobre mercados laborales, más cuadernos para extraer y analizar datos regionales de género y empleo.

## Contenido del repositorio

| Área | Propósito |
|------|-----------|
| **`feminist_bot/`** | **Dataccion Chat Bot** — Gemini en Vertex AI con llamada a herramientas (`list_reports`, `retrieve_context`) sobre PDFs en Google Cloud Storage y un índice RAG en ChromaDB. |
| **`feminist_bot/rag/`** | Pipeline RAG: carga de PDFs, embeddings, almacén vectorial y recuperador. |
| **`feminist_bot/topic_modeling/`** | Utilidades de modelado de tópicos estilo BERTopic para análisis de texto. |
| **`extraction/`** | Cuadernos Jupyter y salidas CSV del Banco Mundial, BID e indicadores relacionados con trabajo y género. |

## Requisitos

- Python 3.10+ (recomendado)
- [Google Cloud SDK](https://cloud.google.com/sdk) (`gcloud`) con un proyecto donde estén habilitados Vertex AI y las APIs que uses
- Un bucket de GCS con informes en PDF organizados bajo prefijos como `current_situation/`, `forward_looking/` y `proposal/` (como usa la herramienta `list_reports` de la app)

## Configuración

1. **Clonar y crear un entorno virtual**

   ```bash
   cd dataccion-agent
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Autenticarse en Google Cloud**

   ```bash
   gcloud auth application-default login
   ```

   Opcional: ejecutar `./enable_apis.sh` (o habilitar las mismas APIs en la consola de Cloud) para Vertex AI y servicios relacionados con tu despliegue.

3. **Variables de entorno**

   Crea un archivo `.env` en `feminist_bot/` (o en el directorio desde el que ejecutes Streamlit) con al menos:

   ```env
   BUCKET_NAME=nombre-de-tu-bucket-gcs
   ```

   La aplicación requiere acceso a Vertex AI para el proyecto de GCP configurado (consulta `feminist_bot/app.py` para `PROJECT_ID` y la configuración del modelo).

4. **Índice RAG**

   En el primer uso puede ser necesario construir el índice vectorial a partir de los PDFs del bucket llamando a `retriever.ingest()` (ver comentarios en `feminist_bot/rag/retriever.py`). La app de Streamlit trae la ingesta comentada en `app.py`; actívala cuando quieras una re-ingesta completa.

## Ejecutar el chat

Desde el directorio `feminist_bot` (para que resuelvan los imports locales):

```bash
cd feminist_bot
streamlit run app.py
```

## Dependencias

Las bibliotecas principales están en `requirements.txt`, entre ellas `streamlit`, `google-genai`, `google-cloud-storage`, `chromadb`, `pypdf` y `bertopic`.

## Licencia

Añade un archivo de licencia si planeas distribuir este repositorio públicamente.
