import base64
import csv
import logging
import re
from typing import List, Tuple

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
import time

import database
from classes import State

# Configure logging for debugging and monitoring
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def build_graph(llm, vector_store, k) -> StateGraph:
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
        """Retrieve MissionOS information including text, images, and videos."""
        logger.info(f"retrieve: Processing query: {query}")
        start_search = time.time()

        try:
            retrieved_docs = vector_store.similarity_search(query, k)
            search_time = time.time() - start_search

            serialized_docs = "\n\n".join(
                f"Source: {doc.metadata.get('source', 'unknown')}\nContent: {doc.page_content}"
                for doc in retrieved_docs
            )

            image_ids: List[int] = []
            videos_set = set()
            for doc in retrieved_docs:
                logger.info(
                    f"Retrieved document: source={doc.metadata.get('source', 'unknown')}, "
                    f"videos={doc.metadata.get('videos', [])}"
                )
                for video in doc.metadata.get("videos", []):
                    videos_set.add((video["url"], video["title"]))
                chunk = doc.page_content
                ids = re.findall(r"db://images/(\d+)", chunk)
                image_ids.extend(int(id) for id in ids)

            videos = [{"url": url, "title": title} for url, title in videos_set]
            start_image_fetch = time.time()
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
                    for doc in retrieved_docs:
                        chunk = doc.page_content
                        for img_ref, img_data in image_map.items():
                            if img_ref in chunk or (
                                img_data["caption"] and img_data["caption"] in chunk
                            ):
                                images.append(img_data)
            except Exception as e:
                logger.error(f"retrieve: Error retrieving images: {str(e)}")
            finally:
                if conn is not None:
                    cursor.close()
                    conn.close()
            image_fetch_time = time.time() - start_image_fetch

            timings = {
                "search_time": search_time,
                "image_fetch_time": image_fetch_time
            }
            logger.info(f"retrieve: Completed with videos: {len(videos)}, images: {len(images)}")
            return serialized_docs, {"docs": retrieved_docs, "images": images, "videos": videos, "timings": timings}
        except Exception as e:
            logger.error(f"retrieve: Failed with error: {str(e)}")
            raise

    def query_or_respond(state: State) -> dict:
        """Decide whether to query tools or respond directly."""
        logger.info(f"query_or_respond: Entering with state keys: {list(state.keys())}")
        logger.info(f"query_or_respond: Messages types: {[type(m).__name__ for m in state['messages']]}")
        start_time = time.time()

        try:
            if "messages" not in state or not state["messages"]:
                logger.error("query_or_respond: 'messages' key missing or empty in state")
                raise KeyError("'messages' key missing or empty in state")

            prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are an assistant for MissionOS. Use the 'retrieve' tool for "
                    "any info-seeking queries about MissionOS."
                )),
                ("human", "{query}"),
            ])
            llm_with_tools = llm.bind_tools([retrieve])
            chain = prompt | llm_with_tools
            response = chain.invoke({"query": state["messages"][-1].content}, config={"configurable": {"thread_id": f"{st.session_state.thread_id}"}})

            response_time = time.time() - start_time
            new_state = state.copy()
            new_state["messages"] = state["messages"] + [response]
            new_state["timings"] = state.get("timings", []) + [
                {"node": "query_or_respond", "time": response_time, "component": "LLM decision"}
            ]

            logger.info(f"query_or_respond: Exiting with state keys: {list(new_state.keys())}")
            logger.info(f"query_or_respond: Messages types after: {[type(m).__name__ for m in new_state['messages']]}")
            return new_state
        except Exception as e:
            logger.error(f"query_or_respond: Failed with error: {str(e)}")
            raise

    def tools_node(state: State) -> dict:
        """Custom node to execute tools and measure execution time."""
        logger.info(f"tools_node: Entering with state keys: {list(state.keys())}")
        logger.info(f"tools_node: Messages types before: {[type(m).__name__ for m in state['messages']]}")
        start_time = time.time()

        try:
            if "messages" not in state or not state["messages"]:
                logger.error("tools_node: 'messages' key missing or empty in state")
                raise KeyError("'messages' key missing or empty in state")

            # Invoke ToolNode and get updated messages
            tool_result = ToolNode([retrieve]).invoke(state)
            logger.info(f"tools_node: After ToolNode, messages types: {[type(m).__name__ for m in tool_result['messages']]}")

            # Explicitly merge with original state to preserve all keys and messages
            updated_state = state.copy()
            updated_state["messages"] = updated_state["messages"] + tool_result["messages"]
            tool_time = time.time() - start_time
            updated_state["timings"] = state.get("timings", []) + [
                {"node": "tools", "time": tool_time, "component": "Tool execution"}
            ]

            logger.info(f"tools_node: Exiting with state keys: {list(updated_state.keys())}")
            logger.info(f"tools_node: Messages types after: {[type(m).__name__ for m in updated_state['messages']]}")
            return updated_state
        except Exception as e:
            logger.error(f"tools_node: Failed with error: {str(e)}")
            raise

    def tools_condition(state: State) -> str:
        """Route based on whether the last message contains tool calls."""
        logger.info(f"tools_condition: Checking state keys: {list(state.keys())}")
        try:
            if "messages" not in state or not state["messages"]:
                logger.error("tools_condition: 'messages' key missing or empty in state")
                raise KeyError("'messages' key missing or empty in state")

            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                logger.info("tools_condition: Routing to 'tools'")
                return "tools"
            logger.info("tools_condition: Routing to END")
            return END
        except Exception as e:
            logger.error(f"tools_condition: Failed with error: {str(e)}")
            raise

    def generate(state: State) -> dict:
        """Generate a response using retrieved context and multimedia."""
        logger.info(f"generate: Entering with state keys: {list(state.keys())}")
        logger.info(f"generate: Messages types: {[type(m).__name__ for m in state['messages']]}")
        start_time = time.time()

        try:
            if "messages" not in state or not state["messages"]:
                logger.error("generate: 'messages' key missing or empty in state")
                raise KeyError("'messages' key missing or empty in state")

            query = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage) and hasattr(msg, "content"):
                    query = msg.content
                    break
            if query is None:
                logger.warning("generate: No HumanMessage found in state.messages")
                query = "unknown"
            logger.info(f"generate: Extracted query: {query}")

            tool_messages = [msg for msg in reversed(state["messages"]) if msg.type == "tool"][::-1]
            csv_file = "retrieved_chunks.csv"
            csv_headers = ["Query", "Qdrant Point ID", "Page Content", "URL"]
            csv_rows = []

            for msg in tool_messages:
                if not hasattr(msg, "artifact") or not msg.artifact:
                    logger.warning("generate: Tool message has no artifact")
                    continue
                artifact = msg.artifact
                retrieved_docs = artifact.get("docs", [])
                videos = artifact.get("videos", [])
                logger.info(f"generate: Retrieved docs structure: {[type(doc) for doc in retrieved_docs]}")
                logger.info(f"generate: Retrieved docs sample: {retrieved_docs[:1] if retrieved_docs else 'Empty'}")

                chunks = re.findall(
                    r"Source: (https?://[^\n]+)\nContent: (.*?)(?=(Source:|$))",
                    msg.content,
                    re.DOTALL
                )
                if not chunks:
                    logger.warning("generate: No chunks parsed from tool message content")
                    chunks = [(artifact.get("source", "unknown"), msg.content, "")]

                for i, (source, content, _) in enumerate(chunks, 1):
                    point_id = "unknown"
                    for doc in retrieved_docs:
                        if isinstance(doc, Document):
                            doc_content = doc.page_content
                            doc_metadata = doc.metadata
                        else:
                            doc_content = doc.get("page_content", "")
                            doc_metadata = doc.get("metadata", {})
                        if doc_content.strip() == content.strip():
                            point_id = doc_metadata.get("_id", "unknown")
                            break
                    if point_id == "unknown":
                        logger.warning(f"generate: No matching document or _id for chunk {i}")
                    logger.info(
                        f"generate: Chunk {i}: Point ID={point_id}, URL={source}, Videos={len(videos)}"
                    )
                    csv_rows.append({
                        "Query": query,
                        "Qdrant Point ID": point_id,
                        "Page Content": content.strip(),
                        "URL": source
                    })

            try:
                with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_headers)
                    if f.tell() == 0:
                        writer.writeheader()
                    for row in csv_rows:
                        writer.writerow(row)
                logger.info(f"generate: Wrote {len(csv_rows)} chunks to {csv_file}")
            except Exception as e:
                logger.error(f"generate: Error writing to CSV: {str(e)}")

            retrieved_content = "\n\n".join(msg.content for msg in tool_messages if msg.content)
            images = []
            videos = []
            for msg in tool_messages:
                if hasattr(msg, "artifact") and msg.artifact:
                    images.extend(msg.artifact.get("images", []))
                    videos.extend(msg.artifact.get("videos", []))

            # Build system prompt with context
            system_message_content = (
                "You are a polite and helpful assistant providing information on MissionOS "
                "to users of MissionOS. The user's query is provided "
                "in the messages that follow this instruction. Use the following pieces "
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

            ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
            system_messages = [msg for msg in state["messages"] if isinstance(msg, SystemMessage)]
            logger.info(f"generate: ai messages: {ai_messages}")
            logger.info(f"generate: system messages: {system_messages}")

            conversation_messages = [
                message for message in state["messages"]
                if message.type in ("human", "system")
                or (message.type == "ai" and not message.tool_calls)
            ]

            logger.info(f"generate: conversation_messages: {conversation_messages}")
        
            # Log HumanMessages in conversation_messages
            human_messages = [msg for msg in conversation_messages if isinstance(msg, HumanMessage)]
            if human_messages:
                logger.info(f"generate: Found {len(human_messages)} HumanMessage(s) in conversation_messages:")
                for i, msg in enumerate(human_messages, 1):
                    logger.info(f"generate: HumanMessage {i}: {msg.content}")
            else:
                logger.warning("generate: No HumanMessages found in conversation_messages")

            prompt = [SystemMessage(content=system_message_content)] + conversation_messages

            response = llm.invoke(prompt, config={"configurable": {"thread_id": f"{st.session_state.thread_id}"}})
            generate_time = time.time() - start_time

            new_state = state.copy()
            new_state["messages"] = state["messages"] + [response]
            new_state["images"] = images
            new_state["videos"] = videos
            new_state["timings"] = state.get("timings", []) + [
                {"node": "generate", "time": generate_time, "component": "LLM generation"}
            ]
            if tool_messages and tool_messages[-1].artifact and "timings" in tool_messages[-1].artifact:
                tool_timings = tool_messages[-1].artifact["timings"]
                for component, time_val in tool_timings.items():
                    new_state["timings"].append({"node": "retrieve", "time": time_val, "component": component})

            logger.info(f"generate: Exiting with state keys: {list(new_state.keys())}")
            logger.info(f"generate: Messages types after: {[type(m).__name__ for m in new_state['messages']]}")
            return new_state
        except Exception as e:
            logger.error(f"generate: Failed with error: {str(e)}")
            raise

    # Initialize the graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("query_or_respond", query_or_respond)
    graph_builder.add_node("tools", tools_node)
    graph_builder.add_node("generate", generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        source="query_or_respond",
        path=tools_condition,
        path_map={END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)