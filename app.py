"""Main entry point for the MissionHelp Demo application.

This Streamlit app initializes the environment, sets up dependencies (LLM, vector
store, LangGraph), and renders the chatbot interface for MissionOS queries.
"""

import streamlit as st

import database
import rag
import session
import setup
import ui


def main() -> None:
    """Initialize and run the MissionHelp Demo application."""
    st.set_page_config(
        page_title="MissionHelp Demo",
        page_icon=":material/support_agent:",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    
    # Initialize session state parameters
    session.setup_session()

    # Ensure Playwright browsers
    setup.install_playwright_browsers()

    # Configure core components
    if not st.session_state.graph:
        setup.set_google_credentials()
        st.session_state.llm = setup.get_llm()
        st.session_state.embeddings = setup.get_embeddings()

        if not setup.collection_exists() or not setup.points_exist():
            database.create_images_table()
            setup.create_collection()
            setup.rebuild_database()

        if not st.session_state.vector_store:
            st.session_state.vector_store = setup.get_vector_store(
                embeddings=st.session_state.embeddings
            )

        st.session_state.graph = rag.build_graph(
            llm=st.session_state.llm,
            vector_store=st.session_state.vector_store,
        )

    # Display UI components
    st.title("MissionHelp Demo")
    setup.display_setup()
    if st.session_state.graph:
        st.session_state.status.update(
            label="Set-up complete!",
            state="complete",
            expanded=False,
        )
        ui.render_chatbot()


if __name__ == "__main__":
    main()