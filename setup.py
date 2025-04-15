"""Set up dependencies and configuration for the MissionHelp Demo application.

This module initializes the LLM, embeddings, Qdrant vector store, and database,
ensuring all components are ready for the RAG pipeline.
"""

import os
import subprocess
import streamlit as st
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

import database
import rag


# Qdrant configuration
QDRANT_INFO = {
    "client": QdrantClient(**st.secrets.qdrant_client_credentials),
    "collection_name": "manual-text",
}


def install_playwright_browsers() -> None:
    """Install Playwright browsers for web scraping.

    Checks if browsers are already installed and installs them if needed.
    """
    playwright_dir = os.path.expanduser("~/.cache/ms-playwright")
    if not os.path.exists(playwright_dir) or not os.listdir(playwright_dir):
        st.session_state.status.write(":material/comedy_mask: Installing Playwright browsers...")
        try:
            subprocess.run(["playwright", "install"], check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to install Playwright browsers: {e}")


def get_llm() -> ChatOpenAI:
    """Initialize the Grok 3 language model.

    Returns:
        A ChatOpenAI instance configured with xAI API.
    """
    st.session_state.status.write(":material/emoji_objects: Setting up the Grok 3 LLM...")
    return ChatOpenAI(
        model="grok-3-beta",
        api_key=st.secrets.xai_api_key,
        base_url="https://api.x.ai/v1",
    )


def set_google_credentials() -> None:
    """Set Google Cloud credentials for database access.

    Writes credentials from secrets to a temporary file and sets the environment variable.
    """
    st.session_state.status.write(":material/key: Setting Google credentials...")
    credentials_json = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    temp_file_path = "google_credentials.json"
    with open(temp_file_path, "w") as f:
        f.write(credentials_json)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path


def get_embeddings() -> VertexAIEmbeddings:
    """Initialize the Gemini text embeddings model.

    Returns:
        A VertexAIEmbeddings instance for text-embedding-004.
    """
    st.session_state.status.write(
        ":material/token: Setting up the Gemini text-embedding-004 model..."
    )
    return VertexAIEmbeddings(
        model="text-embedding-004",
        project=st.secrets.google_project_id,
    )


def delete_collection() -> None:
    """Delete the existing Qdrant collection."""
    st.session_state.status.write(":material/delete: Deleting existing Qdrant collection...")
    client = QDRANT_INFO["client"]
    client.delete_collection(collection_name=QDRANT_INFO["collection_name"])


def create_collection() -> None:
    """Create a new Qdrant collection for vector storage."""
    st.session_state.status.write(":material/category: Creating new Qdrant collection...")
    client = QDRANT_INFO["client"]
    client.create_collection(
        collection_name=QDRANT_INFO["collection_name"],
        vectors_config=models.VectorParams(
            size=768,  # Matches text-embedding-004
            distance=models.Distance.COSINE,
        ),
    )


def collection_exists() -> bool:
    """Check if the Qdrant collection exists.

    Returns:
        True if the collection exists, False otherwise.
    """
    client = QDRANT_INFO["client"]
    existing_collections = client.get_collections()
    return any(
        col.name == QDRANT_INFO["collection_name"]
        for col in existing_collections.collections
    )


def points_exist() -> bool:
    """Check if the Qdrant collection contains points.

    Returns:
        True if points exist, False otherwise.
    """
    client = QDRANT_INFO["client"]
    collection_info = client.get_collection(QDRANT_INFO["collection_name"])
    return collection_info.points_count is not None and collection_info.points_count > 0


def get_vector_store(embeddings: VertexAIEmbeddings) -> QdrantVectorStore:
    """Initialize the Qdrant vector store.

    Args:
        embeddings: The embeddings model for vectorization.

    Returns:
        A QdrantVectorStore instance.
    """
    st.session_state.status.write(":material/database: Setting up Qdrant vector store...")
    return QdrantVectorStore(
        client=QDRANT_INFO["client"],
        collection_name=QDRANT_INFO["collection_name"],
        embedding=embeddings,
    )


def rebuild_database() -> None:
    """Rebuild the database and vector store from scratch.

    Deletes and recreates the Qdrant collection, scrapes webpages, and indexes chunks.
    """
    delete_collection()
    create_collection()
    database.create_images_table()

    docs = database.web_scrape()
    all_splits = database.chunk_text(docs=docs)
    database.index_chunks(
        all_splits=all_splits,
        vector_store=st.session_state.vector_store,
    )


def display_setup() -> None:
    """Display admin setup controls in the Streamlit UI.

    Provides options to re-scrape the manual or reconfigure the RAG pipeline,
    protected by a password.
    """
    with st.expander(label="Set-up", expanded=False, icon=":material/lock_open:"):
        password = st.text_input(
            label="Password",
            type="password",
            placeholder="Enter admin password",
        )

        if password == st.secrets.admin_password:
            st.divider()
            with st.container():
                if st.button(
                    label=":material/database: Re-scrape manual",
                    type="secondary",
                ):
                    rebuild_database()

                if st.button(
                    label=":material/settings: Re-configure RAG",
                    type="secondary",
                ):
                    st.session_state.graph = rag.build_graph(
                        llm=st.session_state.llm,
                        vector_store=st.session_state.vector_store,
                    )