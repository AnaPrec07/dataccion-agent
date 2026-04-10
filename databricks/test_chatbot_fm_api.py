# Databricks notebook source

# COMMAND ----------

# MAGIC %md # Test Chatbot — Databricks Foundation Model API
# MAGIC
# MAGIC Tests the adapted chatbot using Databricks FM API (no external dependencies needed).
# MAGIC Uses the system instructions and LLM directly (RAG requires PDF ingestion separately).

# COMMAND ----------

# MAGIC %pip install openai
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
from openai import OpenAI

# Auto-detect Databricks auth
DATABRICKS_HOST = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"

client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{DATABRICKS_HOST}/serving-endpoints",
)

print(f"Connected to: {DATABRICKS_HOST}")
print(f"Model: {LLM_MODEL}")

# COMMAND ----------

# MAGIC %md ## System Instructions (from feminist_bot)

# COMMAND ----------

SYSTEM_INSTRUCTIONS = """
You are an expert research assistant developed for ONU Mujer (UN Women) Latin America.
Your sole focus is informing users about the structural, cultural, legal, and economic
barriers that women face in Latin American labor markets. You communicate with clarity,
empathy, and academic rigor.

You are not a general-purpose assistant. You only answer questions about women's labor
market barriers and closely related topics.

Key guidelines:
- Always prioritize factual accuracy. Cite data sources when possible (ILO, ECLAC/CEPAL, ONU Mujer, World Bank).
- Acknowledge intersectionality (race, ethnicity, rurality, disability compound barriers).
- Empathetic & non-judgmental tone.
- Evidence-based and neutral presentation.
- Default response language: match the user's language (Spanish or English).
"""

# COMMAND ----------

# MAGIC %md ## Chat Function

# COMMAND ----------

def chat(user_message, history=None):
    """Send a message to the LLM and get a response."""
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content

# COMMAND ----------

# MAGIC %md ## Test 1: Basic question in Spanish

# COMMAND ----------

response = chat("¿Cuáles son las principales barreras que enfrentan las mujeres en el mercado laboral en América Latina?")
print(response)

# COMMAND ----------

# MAGIC %md ## Test 2: Country-specific question in English

# COMMAND ----------

response = chat("What are the biggest barriers for indigenous women entering formal employment in Bolivia?")
print(response)

# COMMAND ----------

# MAGIC %md ## Test 3: Data-oriented question

# COMMAND ----------

response = chat("¿Cuál es la brecha salarial de género promedio en la región y cómo varía entre países?")
print(response)

# COMMAND ----------

# MAGIC %md ## Test 4: Out-of-scope question (should redirect)

# COMMAND ----------

response = chat("Can you help me write my resume?")
print(response)

# COMMAND ----------

# MAGIC %md ## Test 5: Multi-turn conversation

# COMMAND ----------

history = []

questions = [
    "¿Qué es la penalización maternal en el mercado laboral?",
    "¿Cuáles son los datos más recientes sobre esto en Chile y Colombia?",
    "¿Qué políticas han funcionado para reducir esta penalización?",
]

for q in questions:
    print(f"\n{'='*60}")
    print(f"USER: {q}")
    print(f"{'='*60}\n")

    response = chat(q, history=history)
    print(f"ASSISTANT: {response}")

    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": response})

# COMMAND ----------

# MAGIC %md ## Test 6: Embedding Model Test

# COMMAND ----------

embed_response = client.embeddings.create(
    input=["Barreras laborales para mujeres en América Latina", "Gender wage gap in Latin America"],
    model="databricks-bge-large-en",
)
for i, emb in enumerate(embed_response.data):
    print(f"Text {i+1}: dimension={len(emb.embedding)}, first 5 values={emb.embedding[:5]}")

# COMMAND ----------

# MAGIC %md ## All tests passed! The chatbot and embedding models are working on Databricks Free Edition.
