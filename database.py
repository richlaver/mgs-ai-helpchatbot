from bs4 import BeautifulSoup
import streamlit as st
import pandas as pd
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from langchain_community.document_loaders import AsyncChromiumLoader


def generate_webpaths():
    id_filename = 'WUM articles.csv'
    base_url = 'https://www.maxwellgeosystems.com/manuals/demo-manual/manual-web-content-highlight.php?manual_id='

    ids = pd.read_csv(
        filepath_or_buffer=id_filename,
        usecols=[0],
        skip_blank_lines=True
    ).dropna().iloc[:, 0].astype(int).to_list()

    webpaths = [base_url + str(id) for id in ids]
    return webpaths


def web_scrape():
    webpaths = generate_webpaths()

    st.session_state.status.write(':material/hourglass_top: Loading webpages...')
    loader = AsyncChromiumLoader(urls=webpaths)
    docs = loader.load()

    st.session_state.status.write(':material/web: Scraping webpages...')
    img_id = 0
    with st.session_state.status:
        web_scrape_progress = st.progress(value=0)
    doc_num = 0
    for doc in docs:
        soup = BeautifulSoup(doc.page_content, "html.parser")
        div_print = soup.find("div", id="div_print")

        # Find and replace specific <p> tags
        for p_tag in soup.find_all("p", class_="headingp page-header"):
            # Create a new <h1> tag with the same content
            new_tag = soup.new_tag("h1")
            new_tag.string = p_tag.get_text()  # Copy the text content
            p_tag.replace_with(new_tag)

        if div_print:
            for img in div_print.find_all("img"):
                src = img.get("src", "")
                if src.startswith("data:image/png;base64,"):
                    # base64_string = src.split(",")[1]
                    # img_data = base64.b64decode(base64_string)
                    # cursor.execute("INSERT INTO images (data) VALUES (?)", (img_data,))
                    # img_id = cursor.lastrowid
                    img["src"] = f"db://images/{img_id}"
                    img_id += 1

            # Get only the contents of the div_print as a string
            doc.page_content = str(div_print.decode_contents())  # Inner HTML without <div> tags
        else:
            # print("No <div id='div_print'> found")
            doc.page_content = ""

        doc_num += 1
        web_scrape_progress.progress(value=doc_num/len(docs))

    return docs


def chunk_text(docs):
    st.session_state.status.write(':material/content_cut: Chunking text semantically...')
    headers_to_split_on = [
        # ('p class="headingp page-header"', 'Heading 1'),
        ('h1', 'Heading 1'),
        ('h2', 'Heading 2'),
        ('h3', 'Heading 3'),
        ('h4', 'Heading 4')
    ] # From inspecting the web user manual.

    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=headers_to_split_on,
        max_chunk_size=1000, # The text-embedding-004 model accepts a maximum of 2,408 tokens.
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "! ", "? "],
        preserve_links=True, # We want to link to links, images and videos.
        preserve_images=True,
        preserve_videos=True,
        preserve_audio=True,
        stopword_removal=False, # Preserve meaning.
        normalize_text=False, # Capitalisation in the web user manuals carries meaning.
        elements_to_preserve=["table", "ul", "ol"],
        denylist_tags=["script", "style", "head"]
    )

    return splitter.transform_documents(docs) #Injects metadata like source URL.


def index_chunks(all_splits, vector_store):
    st.session_state.status.write(':material/123: Indexing chunks...')
    _ = vector_store.add_documents(documents=all_splits)