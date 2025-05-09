"""Manage session state for the MissionHelp Demo application.

This module initializes and maintains Streamlit session state variables used
across the app for storing configuration, conversation history, and multimedia.
"""

import streamlit as st


def setup_session() -> None:
    """Initialize Streamlit session state variables.

    Sets default values for LLM, embeddings, vector store, graph, and other
    session-specific data to ensure consistent app behavior.
    """
    # Initialize status widget
    st.session_state.status = st.status(
        label="Setting things up...",
        expanded=True,
    )

    # Initialize core components
    if "llm" not in st.session_state:
        st.session_state.llm = False

    if "embeddings" not in st.session_state:
        st.session_state.embeddings = False

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = False

    if "graph" not in st.session_state:
        st.session_state.graph = False

    # Initialize conversation and multimedia
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = 0

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "webpage_urls" not in st.session_state:
        st.session_state.webpage_urls = []

    if "images" not in st.session_state:
        st.session_state.images = []

    if "videos" not in st.session_state:
        st.session_state.videos = []

    # Initialize retrieval parameters
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1000

    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 200
        
    if "retrieval_k" not in st.session_state:
        st.session_state.retrieval_k = 4

    if "test_csv" not in st.session_state:
        st.session_state.test_csv = False

    if "results_csv" not in st.session_state:
        st.session_state.results_csv = False