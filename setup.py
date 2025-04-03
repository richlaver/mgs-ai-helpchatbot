import os
import subprocess
import streamlit as st
import database
import rag
from langchain_openai import ChatOpenAI
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models

web_paths = [
    'https://www.maxwellgeosystems.com/manuals/demo-manual/manual-web-content-highlight.php?manual_id=76',
    'https://www.maxwellgeosystems.com/manuals/demo-manual/manual-web-content-highlight.php?manual_id=117',
    'https://www.maxwellgeosystems.com/manuals/demo-manual/manual-web-content-highlight.php?manual_id=112',
    'https://www.maxwellgeosystems.com/manuals/demo-manual/manual-web-content-highlight.php?manual_id=72'
]

qdrant_info = {
    'client': QdrantClient(**st.secrets.qdrant_client_credentials),
    'collection_name': 'manual-text'
}


# Function to ensure Playwright browsers are installed
def install_playwright_browsers():
    # Check if browsers are already installed
    playwright_dir = os.path.expanduser("~/.cache/ms-playwright")
    if not os.path.exists(playwright_dir) or not os.listdir(playwright_dir):
        st.session_state.status.write(":material/comedy_mask: Installing Playwright browsers...")
        try:
            # Run the install command
            subprocess.run(["playwright", "install"], check=True)
        except subprocess.CalledProcessError as e:
            pass
            # st.error(f"Failed to install Playwright browsers: {e}")
    else:
        pass
        # st.write("Playwright browsers already installed.")


def get_llm():
    st.session_state.status.write(':material/emoji_objects: Setting-up the _Grok 2_ LLM...')
    # Initialize Grok via xAI API
    return ChatOpenAI(
        model="grok-2-latest",
        api_key=st.secrets.xai_api_key,
        base_url="https://api.x.ai/v1"
    )


def set_google_credentials():
    st.session_state.status.write(':material/key: Setting Google credentials...')

    # Get JSON string from secrets
    credentials_json = st.secrets["GOOGLE_CREDENTIALS_JSON"]
    
    # Write to a temporary file
    temp_file_path = "google_credentials.json"
    with open(temp_file_path, "w") as f:
        f.write(credentials_json)
    
    # Set environment variable to file path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = temp_file_path


def get_embeddings():
    st.session_state.status.write(':material/token: Setting-up the _Gemini text-embedding-004_ embeddings model...')
    return VertexAIEmbeddings(
        model="text-embedding-004",
        project=st.secrets.google_project_id
    )


def delete_collection():
    st.session_state.status.write(':material/delete: Deleting existing _Qdrant_ collection...')
    client = qdrant_info['client']
    client.delete_collection(collection_name=qdrant_info['collection_name'])


def create_collection():
    st.session_state.status.write(':material/category: Creating new _Qdrant_ collection...')
    client = qdrant_info['client']

    # Recreate the collection with the same configuration
    client.create_collection(
        collection_name=qdrant_info['collection_name'],
        vectors_config=models.VectorParams(
            size=768,  # Matches text-embedding-004
            distance=models.Distance.COSINE  # Your similarity metric
        )
    )


def collection_exists():
    client = qdrant_info['client']

    existing_collections = client.get_collections()
    return any(col.name == qdrant_info['collection_name'] for col in existing_collections.collections)


def points_exist():
    client = qdrant_info['client']
    collection_info = client.get_collection(qdrant_info['collection_name'])
    point_count = collection_info.vectors_count

    st.write('collection_info: ')
    st.write(collection_info)

    st.write('vector_count:')
    st.write(point_count)

    return point_count is not None and point_count > 0


def get_vector_store(embeddings):
    st.session_state.status.write(':material/database: Setting-up a _Qdrant_ vector store...')
    client = qdrant_info['client']

    return QdrantVectorStore(
        client=client,
        collection_name=qdrant_info['collection_name'],
        embedding=embeddings,
    )


def rebuild_database():
    delete_collection()
    create_collection()

    st.session_state.vector_store = get_vector_store(embeddings=st.session_state.embeddings)
    docs = database.web_scrape(web_paths=web_paths)
    all_splits = database.chunk_text(docs=docs)
    database.index_chunks(all_splits=all_splits, vector_store=st.session_state.vector_store)

    st.session_state.database_created = True


def display_setup():
    with st.expander(
        label='Set-up',
        expanded=False,
        icon=':material/lock_open:'
    ):    
        password = st.text_input(
            label='Password',
            type='password',
            placeholder='Enter admin password'
        )

        if password == st.secrets.admin_password:
            st.divider()
            with st.container():
                if st.button(
                    label=':material/database: Re-scrape manual',
                    type='secondary'
                ):
                    rebuild_database()

                if st.button(
                    label=':material/settings: Re-configure RAG',
                    type='secondary'
                ):
                    st.session_state.graph = rag.build_graph(
                        llm=st.session_state.llm,
                        vector_store=st.session_state.vector_store
                    )