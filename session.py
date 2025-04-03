import streamlit as st


def setup_session():
    st.session_state.status = st.status(
        label='Setting things up...',
        expanded=True
    )

    if 'llm' not in st.session_state:
        st.session_state.llm = False

    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = False

    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = False

    if 'database_created' not in st.session_state:
        st.session_state.database_created = False

    if 'graph' not in st.session_state:
        st.session_state.graph = False

    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = 0
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "webpage_urls" not in st.session_state:
        st.session_state.webpage_urls = []