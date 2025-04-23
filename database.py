"""Manage database operations and web scraping for the MissionHelp Demo application.

This module handles PostgreSQL connections, image storage, and scraping of MissionOS
manual webpages to extract text, images, and videos.
"""

import base64
import json
import logging
import os
from typing import List
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from google.cloud.sql.connector import Connector, IPTypes
from google.oauth2 import service_account
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_core.documents import Document
from langchain_text_splitters import HTMLSemanticPreservingSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Configure logging for database and scraping events
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("scrape_debug.log")
    ]
)
logger = logging.getLogger(__name__)


def getconn():
    """Establish a connection to the PostgreSQL database.

    Uses Google Cloud SQL Connector with IAM authentication.

    Returns:
        A database connection object.

    Raises:
        Exception: If connection fails.
    """
    credentials_file = "google_credentials.json"
    credentials = service_account.Credentials.from_service_account_file(credentials_file)
    connector = Connector(credentials=credentials)

    conn = connector.connect(
        instance_connection_string="intense-age-455102-i9:asia-east2:mgs-web-user-manual",
        driver="pg8000",
        user="langchain-tutorial-rag-service@intense-age-455102-i9.iam",
        enable_iam_auth=True,
        db="postgres",
        ip_type=IPTypes.PUBLIC,
    )
    return conn


def query_db() -> str:
    """Query the database to retrieve its version.

    Returns:
        The PostgreSQL version string.

    Raises:
        Exception: If the query fails.
    """
    conn = None
    try:
        conn = getconn()
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        result = cursor.fetchone()
        version_string = result[0]
        return version_string
    finally:
        if conn is not None:
            cursor.close()
            conn.close()


def create_images_table() -> None:
    """Create or recreate the images table in the database.

    Drops the existing table if present and creates a new one to store image data.
    """
    conn = None
    try:
        conn = getconn()
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS images;")
        cursor.execute(
            """
            CREATE TABLE images (
                id SERIAL PRIMARY KEY,
                url VARCHAR(255) NOT NULL,
                image_binary BYTEA NOT NULL,
                caption TEXT
            );
            """
        )
        cursor.execute("GRANT ALL PRIVILEGES ON images TO postgres;")
        conn.commit()
        st.session_state.status.write(":material/database: Images table recreated successfully.")
    except Exception as e:
        st.error(f"Error creating images table: {str(e)}")
        logger.error(f"Error creating images table: {str(e)}")
    finally:
        if conn is not None:
            cursor.close()
            conn.close()


def generate_webpaths() -> List[str]:
    """Generate URLs for MissionOS manual webpages.

    Reads webpage IDs from a CSV file and constructs URLs.

    Returns:
        A list of webpage URLs.
    """
    id_filename = "WUM articles.csv"
    base_url = (
        "https://www.maxwellgeosystems.com/manuals/demo-manual/"
        "manual-web-content-highlight.php?manual_id="
    )
    ids = (
        pd.read_csv(filepath_or_buffer=id_filename, usecols=[0], skip_blank_lines=True)
        .dropna()
        .iloc[:, 0]
        .astype(int)
        .to_list()
    )
    return [base_url + str(id) for id in ids]


def load_cached_docs(cache_dir: str = "scrape_cache") -> List[Document]:
    """Load cached raw documents from disk.

    Args:
        cache_dir: Directory where cached JSON files are stored.

    Returns:
        List of Document objects from cache, or empty list if cache is invalid.
    """
    if not os.path.exists(cache_dir):
        return []

    docs = []
    for filename in os.listdir(cache_dir):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(cache_dir, filename), "r") as f:
                    data = json.load(f)
                doc = Document(page_content=data["page_content"], metadata=data["metadata"])
                docs.append(doc)
            except Exception as e:
                logger.error(f"Error loading cached file {filename}: {str(e)}")
    return docs


def save_cached_docs(docs: List[Document], cache_dir: str = "scrape_cache") -> None:
    """Save raw documents to disk as JSON files.

    Args:
        docs: List of Document objects to cache.
        cache_dir: Directory to store JSON files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    for i, doc in enumerate(docs):
        try:
            cache_data = {"page_content": doc.page_content, "metadata": doc.metadata}
            filename = os.path.join(cache_dir, f"doc_{i}.json")
            with open(filename, "w") as f:
                json.dump(cache_data, f)
        except Exception as e:
            logger.error(f"Error caching document {doc.metadata.get('source', 'unknown')}: {str(e)}")


def web_scrape(use_cache: bool = True, cache_dir: str = "scrape_cache") -> List[Document]:
    """Scrape MissionOS manual webpages for text, images, and videos.

    Loads raw documents from cache if available, otherwise uses AsyncChromiumLoader.
    Processes documents to extract content, images, and videos, storing images in the database.

    Args:
        use_cache: If True, attempt to load from cache before scraping.
        cache_dir: Directory for cached JSON files.

    Returns:
        List of Document objects with processed content and metadata.

    Raises:
        Exception: If scraping or database operations fail.
    """
    if use_cache:
        cached_docs = load_cached_docs(cache_dir)
        if cached_docs:
            docs = cached_docs
        else:
            webpaths = generate_webpaths()
            st.session_state.status.write(":material/hourglass_top: Loading webpages...")
            loader = AsyncChromiumLoader(urls=webpaths)
            docs = loader.load()
            save_cached_docs(docs, cache_dir)
    else:
        webpaths = generate_webpaths()
        st.session_state.status.write(":material/hourglass_top: Loading webpages...")
        loader = AsyncChromiumLoader(urls=webpaths)
        docs = loader.load()
        save_cached_docs(docs, cache_dir)

    st.session_state.status.write(":material/web: Processing webpages...")
    conn = None
    try:
        conn = getconn()
        cursor = conn.cursor()
        with st.session_state.status:
            web_scrape_progress = st.progress(value=0)
        doc_num = 0

        for doc in docs:
            base_url = doc.metadata["source"]
            soup = BeautifulSoup(doc.page_content, "html.parser")
            div_print = soup.find("div", id="div_print")
            doc.metadata["videos"] = []

            if div_print:
                # Convert relative URLs to absolute
                for a_tag in div_print.find_all("a"):
                    href = a_tag.get("href")
                    if href and not href.startswith(("#", "mailto:", "javascript:", "tel:")):
                        a_tag["href"] = urljoin(base_url, href)

                # Extract YouTube videos from iframes
                for iframe in div_print.find_all("iframe"):
                    iframe_src = iframe.get("src", "")
                    if "youtube.com" in iframe_src or "youtu.be" in iframe_src:
                        try:
                            parsed_url = urlparse(iframe_src)
                            video_id = None
                            if "youtube.com" in parsed_url.netloc:
                                if "/embed/" in parsed_url.path:
                                    video_id = parsed_url.path.split("/embed/")[-1].split("?")[0]
                                else:
                                    video_id = parse_qs(parsed_url.query).get("v", [None])[0]
                            elif "youtu.be" in parsed_url.netloc:
                                video_id = parsed_url.path.strip("/")

                            if not video_id:
                                raise ValueError(f"Could not extract video ID from {iframe_src}")

                            watch_url = f"https://www.youtube.com/watch?v={video_id}"
                            headers = {
                                "User-Agent": (
                                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                                    "Chrome/91.0.4472.124 Safari/537.36"
                                )
                            }
                            response = requests.get(watch_url, headers=headers, timeout=10)
                            response.raise_for_status()

                            iframe_soup = BeautifulSoup(response.text, "html.parser")
                            title = None
                            meta_title = iframe_soup.find("meta", attrs={"name": "title"})
                            if meta_title and meta_title.get("content"):
                                title = meta_title.get("content").strip()
                            else:
                                og_title = iframe_soup.find(
                                    "meta", attrs={"property": "og:title"}
                                )
                                if og_title and og_title.get("content"):
                                    title = og_title.get("content").strip()
                                else:
                                    title_tag = iframe_soup.find("title")
                                    if (
                                        title_tag
                                        and title_tag.get_text(strip=True)
                                        and title_tag.get_text(strip=True) != "YouTube"
                                    ):
                                        title = title_tag.get_text(strip=True)

                            if title:
                                title = title.replace(" - YouTube", "").strip()
                                if not title or title == "-":
                                    title = None
                            if not title:
                                title = f"Untitled Video {video_id}"

                            url_tag = iframe_soup.find("link", rel="canonical")
                            video_url = url_tag.get("href", watch_url) if url_tag else watch_url
                            doc.metadata["videos"].append({"url": video_url, "title": title})
                        except Exception as e:
                            logger.warning(f"Error processing iframe {iframe_src}: {str(e)}")
                            video_id = (
                                parse_qs(parsed_url.query).get("v", [None])[0]
                                or parsed_url.path.strip("/")
                            )
                            fallback_title = (
                                f"Untitled Video {video_id}" if video_id else "Untitled Video"
                            )
                            doc.metadata["videos"].append(
                                {"url": iframe_src, "title": fallback_title}
                            )

                # Convert specific <p> tags to <h1>
                for p_tag in div_print.find_all("p", class_="headingp page-header"):
                    new_tag = soup.new_tag("h1")
                    new_tag.string = p_tag.get_text()
                    p_tag.replace_with(new_tag)

                # Process and store images
                img_count = 0
                imgs = div_print.find_all("img")
                for img in imgs:
                    img_count += 1
                    src = img.get("src", "")
                    if src.startswith("data:image/png;base64,"):
                        try:
                            base64_string = src.split(",")[1]
                            image_binary = base64.b64decode(base64_string)
                            figure = img.find_parent("figure")
                            caption = ""
                            if figure:
                                figcaption = figure.find("figcaption")
                                if figcaption:
                                    caption = figcaption.get_text(strip=True)

                            cursor.execute(
                                """
                                INSERT INTO images (url, image_binary, caption)
                                VALUES (%s, %s, %s)
                                RETURNING id
                                """,
                                (base_url, image_binary, caption),
                            )
                            img_id = cursor.fetchone()[0]
                            img["src"] = f"db://images/{img_id}"
                        except Exception as e:
                            logger.error(f"Error storing image {img_count} in {base_url}: {str(e)}")
                    else:
                        logger.warning(f"Image {img_count} in {base_url} has invalid src: {src}")

                doc.page_content = str(div_print.decode_contents())
            else:
                doc.page_content = ""

            doc_num += 1
            web_scrape_progress.progress(value=doc_num / len(docs))

        conn.commit()
    except Exception as e:
        st.session_state.error(f"Error during scraping: {str(e)}")
        logger.error(f"Scraping error: {str(e)}")
        raise
    finally:
        if conn is not None:
            cursor.close()
            conn.close()

    return docs


def chunk_text(docs: List[Document]) -> List[Document]:
    """Split documents into semantic chunks for vector storage.

    Uses HTML-aware splitting to preserve structure and parent metadata, with a
    fallback to recursive text splitting if needed.

    Args:
        docs: List of Document objects to chunk.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    st.session_state.status.write(":material/content_cut: Chunking text semantically...")
    logger.info(f"Processing {len(docs)} documents")

    # Configure HTML splitter
    headers_to_split_on = [
        ("h1", "Heading 1"),
        ("h2", "Heading 2"),
        ("h3", "Heading 3"),
        ("h4", "Heading 4"),
    ]
    splitter = HTMLSemanticPreservingSplitter(
        headers_to_split_on=headers_to_split_on,
        max_chunk_size=st.session_state.get("chunk_size", st.session_state.chunk_size),
        chunk_overlap=st.session_state.get("chunk_overlap", st.session_state.chunk_overlap),
        separators=["\n\n", "\n", ". ", "! ", "? "],
        preserve_links=True,
        preserve_images=True,
        preserve_videos=True,
        preserve_audio=True,
        stopword_removal=False,
        normalize_text=False,
        elements_to_preserve=["table", "ul", "ol"],
        denylist_tags=["script", "style", "head"],
        preserve_parent_metadata=True,
    )

    all_splits = splitter.transform_documents(documents=docs)

    # Fallback to recursive splitting if no chunks produced
    if not all_splits:
        logger.warning("No chunks from HTML splitter, using recursive text splitter")
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=st.session_state.get("chunk_size", st.session_state.chunk_size),
            chunk_overlap=st.session_state.get("chunk_overlap", st.session_state.chunk_overlap),
            separators=["\n\n", "\n", ". ", "! ", "? "],
        )
        all_splits = fallback_splitter.split_documents(docs)

    # Final fallback: use original documents
    if not all_splits:
        logger.warning("No chunks created, using original documents")
        all_splits = docs[:]

    st.session_state.status.write(":material/done: Chunking complete.")
    logger.info(f"Created {len(all_splits)} chunks/documents")
    return all_splits


def index_chunks(all_splits: List[Document], vector_store) -> None:
    """Index document chunks in the vector store.

    Args:
        all_splits: List of chunked Document objects.
        vector_store: Qdrant vector store instance.
    """
    st.session_state.status.write(":material/123: Indexing chunks...")
    try:
        vector_store.add_documents(documents=all_splits)
    except Exception as e:
        logger.error(f"Error indexing chunks: {str(e)}")
        st.error(f"Error indexing chunks: {str(e)}")
    st.session_state.status.write(":material/done: Indexing complete.")