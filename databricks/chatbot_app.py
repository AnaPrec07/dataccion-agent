# Databricks notebook source

# COMMAND ----------

# MAGIC %md # Dataccion Chat Bot — ONU Mujer Research Assistant
# MAGIC
# MAGIC This notebook runs a Streamlit chatbot powered by Databricks Foundation Model API
# MAGIC with RAG (Retrieval-Augmented Generation) for research on women's labor market
# MAGIC barriers in Latin America.
# MAGIC
# MAGIC **Adapted from the original GCP/Gemini implementation to Databricks.**
# MAGIC
# MAGIC ### Setup
# MAGIC 1. Upload PDF reports to `/Volumes/workspace/dataccion/pdfs/`
# MAGIC 2. Run the ingestion cell to build the vector store
# MAGIC 3. Use the chat interface to query the knowledge base

# COMMAND ----------

# MAGIC %pip install chromadb pypdf openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
import os
import logging

# Add the workspace files directory to Python path
sys.path.insert(0, "/Workspace/Users/kevinromerooviedo@gmail.com/dataccion")

from feminist_bot.instructions import SYSTEM_INSTRUCTIONS
from feminist_bot.rag import Retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

# Configuration
VOLUME_PATH = "/Volumes/workspace/dataccion/pdfs"  # Upload PDFs here
EMBEDDING_MODEL = "databricks-bge-large-en"
LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"  # or any available FM API model
CHROMA_PERSIST_DIR = "/tmp/feminist_bot_chroma_db"

# Databricks auth (auto-detected in notebook environment)
DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

print(f"Workspace: {DATABRICKS_HOST}")
print(f"PDF Volume: {VOLUME_PATH}")
print(f"Embedding model: {EMBEDDING_MODEL}")
print(f"LLM model: {LLM_MODEL}")

# COMMAND ----------

# MAGIC %md ## Initialize RAG Retriever

# COMMAND ----------

retriever = Retriever(
    volume_path=VOLUME_PATH,
    embedding_model=EMBEDDING_MODEL,
    chunk_size=800,
    chunk_overlap=150,
    top_k=5,
    persist_directory=CHROMA_PERSIST_DIR,
)

print(f"Vector store has {retriever._store.count} chunks")

# COMMAND ----------

# MAGIC %md ## Ingest PDFs (run once or when new PDFs are added)
# MAGIC
# MAGIC Upload your PDF reports to the Volume first:
# MAGIC ```
# MAGIC /Volumes/workspace/dataccion/pdfs/current_situation/
# MAGIC /Volumes/workspace/dataccion/pdfs/forward_looking/
# MAGIC /Volumes/workspace/dataccion/pdfs/proposal/
# MAGIC ```

# COMMAND ----------

# Uncomment to ingest PDFs:
# new_chunks = retriever.ingest(force=False)
# print(f"Ingested {new_chunks} new chunks. Total: {retriever._store.count}")

# COMMAND ----------

# MAGIC %md ## Tool Functions

# COMMAND ----------

def list_reports(report_category: str) -> list[dict]:
    """List PDF reports in a category folder within the Volume."""
    from pathlib import Path
    folder = Path(VOLUME_PATH) / report_category
    if not folder.exists():
        return [{"error": f"Category '{report_category}' not found"}]

    return [
        {
            "name": p.name,
            "size_bytes": p.stat().st_size,
            "path": str(p),
        }
        for p in folder.glob("*.pdf")
    ]


def retrieve_context(query: str, k: int = 5) -> str:
    """Search the RAG knowledge base for relevant excerpts."""
    return retriever.retrieve_context(query, k=k)

# COMMAND ----------

# MAGIC %md ## Chat with the Bot

# COMMAND ----------

from openai import OpenAI
import json

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{DATABRICKS_HOST}/serving-endpoints",
)

# Tool definitions for function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_context",
            "description": (
                "Searches the knowledge base of research reports and returns "
                "the most relevant excerpts to answer the user's question about "
                "women in the Latin American labor market."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or topic to search for.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of excerpts to retrieve (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_reports",
            "description": "Gets the list of PDF reports by category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_category": {
                        "type": "string",
                        "description": "Category: 'current_situation', 'forward_looking', or 'proposal'.",
                    },
                },
                "required": ["report_category"],
            },
        },
    },
]

# Available tool implementations
tool_functions = {
    "retrieve_context": retrieve_context,
    "list_reports": list_reports,
}


def chat(user_message: str, history: list[dict] | None = None) -> str:
    """
    Send a message to the LLM with tool-calling support.

    Args:
        user_message: The user's question.
        history: Optional conversation history (list of message dicts).

    Returns:
        The assistant's response text.
    """
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        choice = response.choices[0]

        if choice.finish_reason == "tool_calls":
            # Process tool calls
            assistant_msg = choice.message
            messages.append(assistant_msg.model_dump())

            for tool_call in assistant_msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)
                logger.info(f"Tool call: {fn_name}({fn_args})")

                result = tool_functions[fn_name](**fn_args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })
        else:
            return choice.message.content

# COMMAND ----------

# MAGIC %md ## Interactive Chat
# MAGIC
# MAGIC Run the cell below and enter your questions about women's labor market barriers in Latin America.

# COMMAND ----------

# Example queries:
print("=" * 60)
print("Dataccion Chat Bot — ONU Mujer Research Assistant")
print("=" * 60)
print()

# Single question example:
response = chat("¿Cuáles son las principales barreras que enfrentan las mujeres en el mercado laboral en América Latina?")
print(response)

# COMMAND ----------

# MAGIC %md ## Multi-turn Conversation Example

# COMMAND ----------

# Multi-turn conversation
history = []

questions = [
    "¿Qué es la brecha salarial de género en América Latina?",
    "¿Cómo afecta la maternidad al empleo de las mujeres?",
    "¿Qué políticas han sido exitosas para reducir estas barreras?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"USER: {q}")
    print(f"{'='*60}")

    response = chat(q, history=history)
    print(f"\nASSISTANT: {response}")

    # Update history
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": response})
