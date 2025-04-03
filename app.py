import setup
import session
import rag
import ui
import streamlit as st

st.title('MissionHelp Demo')

session.setup_session() # Set-up session_state variables

setup.install_playwright_browsers()

if not st.session_state.graph:
    setup.set_google_credentials()
    st.session_state.llm = setup.get_llm()
    st.session_state.embeddings = setup.get_embeddings()

    if not setup.collection_exists():
        setup.create_collection() # Set-up Qdrant collection

    if not setup.points_exist():
        setup.rebuild_database()

    if not st.session_state.vector_store:
        st.session_state.vector_store = setup.get_vector_store(embeddings=st.session_state.embeddings)
        
    st.session_state.graph = rag.build_graph(
        llm=st.session_state.llm,
        vector_store=st.session_state.vector_store
    )

setup.display_setup()

if st.session_state.graph:
    st.session_state.status.update(
        label='Set-up complete!',
        state='complete',
        expanded=False
    )
    ui.render_chatbot()