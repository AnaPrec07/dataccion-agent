import streamlit as st
from google import genai
from google.genai import types
import requests
import logging
import os
from google.cloud import storage
from instructions import SYSTEM_INSTRUCTIONS
from dotenv import load_dotenv
from rag import Retriever

# Load environment variables
load_dotenv()


# -- Defining variables and parameters --
REGION = "global"
PROJECT_ID = "porygon-dataccion"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
BUCKET_NAME = os.getenv("BUCKET_NAME", "")

temperature = 0.95
top_p = 0.95

# Initialize clients
storage_client = storage.Client()


# -- RAG Retriever --
@st.cache_resource
def get_retriever():
    retriever = Retriever(project_id=PROJECT_ID, bucket_name=BUCKET_NAME)
    # retriever.ingest()
    return retriever


retriever = get_retriever()

# --- Tooling ---
# Define retrieve context declaration
declaration_retrieve_context = types.FunctionDeclaration(
    name="retrieve_context",
    description=(
        "Searches the feminist bot's knowledge base of research reports "
        "and returns the most relevant excerpts to answer the user's question. "
        "Call this before answering any question about statistics, policies, or "
        "barriers related to women in the Latin American labor market."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user's question or topic to search for in the knowledge base.",
            },
            "k": {
                "type": "integer",
                "description": "Number of excerpts to retrieve (default 5).",
            },
        },
        "required": ["query"],
    },
)


# Define list country reports declaration
declaration_list_reports = types.FunctionDeclaration(
    name="list_reports",
    description="Gets the list of PDF reports in bucket.",
    parameters={
        "type": "object",
        "properties": {
            "report_category": {
                "type": "string",
                "description": (
                    "The category of the report. "
                    "Can only be 'current_situation', "
                    "'forward_looking' or 'proposal'"
                ),
            }
        },
        "required": ["report_category"],
    },
)


# Define list country reports function
def list_reports(report_category: str):
    """
    Lists country documents in project's GCS bucket.

    Args:
        country: Folder path prefix to filter results.

    Returns:
        List of dicts with name, size, content_type, and updated timestamp
    """
    bucket = storage_client.bucket(os.getenv("BUCKET_NAME"))
    blobs = bucket.list_blobs(prefix=report_category)

    return [
        {
            "name": blob.name,
            "size_bytes": blob.size,
            "content_type": blob.content_type,
            "updated": blob.updated.isoformat() if blob.updated else None,
        }
        for blob in blobs
    ]


# Define retrieve context function
def retrieve_context(query: str, k: int = 5) -> str:
    return retriever.retrieve_context(query, k=k)


# --- Initialie the Vertex AI Client ---
try:
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)
except Exception as e:
    st.error()
    st.stop()


# -- Get chat here --
def get_chat(model_name: str):
    if f"chat-{model_name}" not in st.session_state:

        # Define the tools configuration for the model
        tools = types.Tool(
            function_declarations=[
                declaration_list_reports,
                declaration_retrieve_context,
            ]
        )

        # Define the generate_content configuration, including tools
        generate_content_config = types.GenerateContentConfig(
            system_instruction=types.Part.from_text(text=SYSTEM_INSTRUCTIONS),
            tools=[tools],  # Pass the tool definition here
            temperature=temperature,
            top_p=top_p,
        )

        # Create a new chat session
        chat = client.chats.create(
            model=model_name,
            config=generate_content_config,
        )

        st.session_state[f"chat-{model_name}"] = chat
    return st.session_state[f"chat-{model_name}"]


# --- Call the model ---
def call_model(prompt: str, model_name: str) -> str:
    """
    This function interacts with a large language model (LLM) to generate text based on a given prompt.
    It maintains a chat session and handles function calls from the model to external tools.
    """
    try:
        message_content = prompt

        # Get the existing chat session or get a new one.
        chat = get_chat(model_name)

        # Start the tool-calling loop.
        while True:
            # Send the message to the model.
            response = chat.send_message(message_content)

            # Check if the model wants to call a tool.
            has_tool_calls = False

            for part in response.candidates[0].content.parts:
                if part.function_call:
                    has_tool_calls = True
                    function_call = part.function_call
                    logging.info(f"Function to call: {function_call.name}")
                    logging.info(f"Afguments: {function_call.args}")

                    # Call the appropriate function if the model requests it.
                    if function_call.name == "list_reports":
                        result = list_reports(**function_call.args)
                    elif function_call.name == "retrieve_context":
                        result = retrieve_context(**function_call.args)
                    function_response_part = types.Part.from_function_response(
                        name=function_call.name,
                        response={"result": result},
                    )
                    message_content = [function_response_part]

                elif part.text:
                    logging.info("No function call found in the response")
                    logging.info(response.text)

            # If no tool calls made, break the loop.
            if not has_tool_calls:
                break
        # Return the model's final response.
        return response.text

    except Exception as e:
        return f"error {e}"


# --- Presentation Tier (Streamlit) ---
# Set the title of the Streamlit application
st.title("Dataccion Chat Bot")

# initialize session state variables if htey don't exist
if "messages" not in st.session_state:
    # Initialize the chat history iwth a welcome message
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

# Display the chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Get user input
if prompt := st.chat_input():
    # Add the user's message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user's message
    st.chat_message("user").write(prompt)

    # Show a spinner while waiting for the model's response
    with st.spinner("Thinking..."):
        # Get the model's response using hte call_model function
        model_response = call_model(prompt, GEMINI_MODEL_NAME)
        # Add the model's repsonse to the chat history-
        st.session_state.messages.append(
            {"role": "assistant", "content": model_response}
        )
        # Display the model's response
        st.chat_message("assistant").write(model_response)
