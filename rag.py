"""Define the Retrieval-Augmented Generation (RAG) pipeline using LangGraph.

This module builds a workflow to handle MissionOS queries by retrieving context,
executing tools, and generating responses with text, images, and videos.
"""

import base64
import logging
import re
from typing import List, Tuple

import streamlit as st
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

import database
from classes import MessagesState


# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def build_graph(llm, vector_store):
    """Build the LangGraph workflow for query processing.

    Args:
        llm: The language model instance for generating responses.
        vector_store: The Qdrant vector store for document retrieval.

    Returns:
        A compiled LangGraph instance ready to process queries.
    """
    st.session_state.status.write(":material/lan: Building LangGraph workflow...")

    @tool(response_format="content_and_artifact")
    def retrieve(query: str) -> Tuple[str, dict]:
        """Retrieve MissionOS information including text, images, and videos.

        Searches the vector store for relevant documents, extracts image references,
        and fetches corresponding images from the database. Deduplicates videos by URL.

        Args:
            query: The user's query string.

        Returns:
            A tuple of serialized document content and an artifact dictionary
            containing documents, images, and videos.
        """
        # Perform similarity search
        retrieved_docs = vector_store.similarity_search(query, k=4)
        serialized_docs = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )

        # Collect image IDs and videos
        image_ids: List[int] = []
        videos_set = set()
        for doc in retrieved_docs:
            logger.info(
                f"Retrieved document: source={doc.metadata.get('source', 'unknown')}, "
                f"videos={doc.metadata.get('videos', [])}"
            )
            # Extract videos
            for video in doc.metadata.get("videos", []):
                videos_set.add((video["url"], video["title"]))
            # Extract image IDs from content
            chunk = doc.page_content
            ids = re.findall(r"db://images/(\d+)", chunk)
            image_ids.extend(int(id) for id in ids)

        # Convert videos set to list
        videos = [{"url": url, "title": title} for url, title in videos_set]

        # Fetch images from database
        images = []
        conn = None
        try:
            conn = database.getconn()
            cursor = conn.cursor()
            if image_ids:
                cursor.execute(
                    "SELECT id, image_binary, caption FROM images WHERE id = ANY(%s)",
                    (image_ids,),
                )
                db_images = cursor.fetchall()
                image_map = {
                    f"db://images/{img[0]}": {
                        "base64": base64.b64encode(img[1]).decode("utf-8"),
                        "caption": img[2],
                    }
                    for img in db_images
                }
                # Associate images with documents
                for doc in retrieved_docs:
                    chunk = doc.page_content
                    for img_ref, img_data in image_map.items():
                        if img_ref in chunk or (
                            img_data["caption"] and img_data["caption"] in chunk
                        ):
                            images.append(img_data)
        except Exception as e:
            logger.error(f"Error retrieving images: {str(e)}")
        finally:
            if conn is not None:
                cursor.close()
                conn.close()

        logger.info(f"retrieve: Returning videos: {videos}")
        return serialized_docs, {"docs": retrieved_docs, "images": images, "videos": videos}

    def query_or_respond(state: MessagesState) -> dict:
        """Decide whether to query tools or respond directly.

        Processes the latest user query and determines if tool usage is needed.

        Args:
            state: The current conversation state with messages.

        Returns:
            A dictionary with the LLM's response message.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an assistant for MissionOS. Use the 'retrieve' tool for "
                "any info-seeking queries about MissionOS."
            )),
            ("human", "{query}"),
        ])
        llm_with_tools = llm.bind_tools([retrieve])
        chain = prompt | llm_with_tools
        response = chain.invoke({"query": state["messages"][-1].content})
        return {"messages": [response]}

    def tools_condition(state: MessagesState) -> str:
        """Route based on whether the last message contains tool calls.

        Args:
            state: The current conversation state with messages.

        Returns:
            The next node ("tools" or END).
        """
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    def generate(state: MessagesState) -> dict:
        """Generate a response using retrieved context and multimedia.

        Combines tool outputs (text, images, videos) into a coherent response,
        referencing images appropriately. Logs details of retrieved chunks, including
        text, URL, ID, videos, and images (excluding base64 data).

        Args:
            state: The current conversation state with messages and artifacts.

        Returns:
            A dictionary with the response message, images, and videos.
        """
        # Extract tool messages (most recent first)
        tool_messages = [
            msg for msg in reversed(state["messages"]) if msg.type == "tool"
        ][::-1]

        # Log retrieved chunk details from single tool message
        for msg in tool_messages:
            if not hasattr(msg, "artifact") or not msg.artifact:
                logger.warning("Tool message has no artifact, skipping logging")
                continue
            artifact = msg.artifact
            # Log artifact for debugging
            logger.info(f"Tool message artifact: {artifact}")
            videos = artifact.get("videos", [])
            # Parse chunks from msg.content
            chunks = re.findall(
                r"Source: (https?://[^\n]+)\nContent: (.*?)(?=(Source:|$))",
                msg.content,
                re.DOTALL
            )
            if not chunks:
                logger.warning("No chunks parsed from tool message content")
                chunks = [(artifact.get("source", "unknown"), msg.content, "")]

            # Log each chunk
            chunk_id = msg.id if hasattr(msg, "id") else "unknown"
            for i, (source, content, _) in enumerate(chunks, 1):
                logger.info(
                    f"Retrieved chunk {i}: "
                    f"ID={chunk_id}-{i}, "
                    f"URL={source}, "
                    f"Videos={videos}, "
                    f"Text preview={content[:500].strip() + '...' if content else 'empty'}"
                )

        # Combine tool content
        retrieved_content = "\n\n".join(
            msg.content for msg in tool_messages if msg.content
        )
        images = []
        videos = []
        for msg in tool_messages:
            if hasattr(msg, "artifact") and msg.artifact:
                images.extend(msg.artifact.get("images", []))
                videos.extend(msg.artifact.get("videos", []))

        # Build system prompt with context
        system_message_content = (
            "You are a polite and helpful assistant providing information on MissionOS "
            "to users of MissionOS. Treat the user's input as a request for information "
            "and that the question has already been provided. Use the following pieces "
            "of retrieved context, images, and videos to provide information that is "
            "directly relevant to the user's request. Respond using simple language "
            "that is easy to understand. The image captions provide clues how you can "
            "reference the images in your response. The video titles provide clues how "
            "you can reference the videos in your response. Treat the user as if all "
            "he or she knows about MissionOS is that it is a construction and "
            "instrumentation data platform. Provide options for further requests for "
            "information. Start each response with an overview of the topic raised in "
            "the question. The overview should introduce the topic and its context. "
            "Order your response in a logical way and use bullet points or numbered "
            "lists where appropriate. If the user asks a question that is definitely "
            "not related to MissionOS, provide a polite response indicating that you "
            "cannot assist. If an image is relevant, reference it using [Image N] "
            "(e.g., [Image 1], [Image 2]) at the end of a sentence or logical break, "
            "ensuring the reference enhances the explanation without disrupting "
            "sentence flow. Do not place [Image N] mid-sentence unless absolutely "
            "necessary, and avoid trailing punctuation (e.g., '.', ',') after "
            "[Image N]. Number images sequentially based on their order (1 for first, "
            "2 for second, etc.). If you don't know the answer, say so clearly.\n\n"
            f"Context:\n{retrieved_content}\n\n"
            f"Available images: {len(images)} image(s)\n"
            f"Available videos: {len(videos)} video(s)"
        )

        # Filter conversation messages
        conversation_messages = [
            message for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(content=system_message_content)] + conversation_messages

        # Generate response
        response = llm.invoke(prompt)
        return {"messages": [response], "images": images, "videos": videos}

    # Initialize the graph
    graph_builder = StateGraph(MessagesState)

    # Add nodes
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", ToolNode([retrieve]))
    graph_builder.add_node("generate", generate)

    # Define edges
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        source="query_or_respond",
        path=tools_condition,
        path_map={END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    # Compile with memory
    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)