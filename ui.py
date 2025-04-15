"""Render the chatbot user interface for the MissionHelp Demo application.

This module defines the Streamlit-based UI, handling chat history display,
user input, and multimedia rendering (images, videos) for MissionOS queries.
"""

import base64
import logging
import re

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# Configure logging for UI events and errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def render_chatbot() -> None:
    """Render the chatbot interface and handle user interactions.

    Displays chat history, processes user queries, and renders responses with
    images and videos using the LangGraph workflow from rag.py.
    """
    # Apply custom CSS and JavaScript for styling and auto-scrolling
    st.markdown(
        """
        <style>
        .chat-messages {
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 60px;
        }
        .stChatInput {
            position: fixed;
            bottom: 10px;
            width: 100%;
            max-width: 720px;
            left: 50%;
            transform: translateX(-50%);
            background-color: white;
            z-index: 1000;
        }
        .inline-image {
            margin: 10px 0;
            max-width: 100%;
        }
        </style>
        <script>
        function scrollToBottom() {
            const chatMessages = document.querySelector('.chat-messages');
            if (chatMessages) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
        document.addEventListener('DOMContentLoaded', scrollToBottom);
        document.addEventListener('streamlit:render', scrollToBottom);
        </script>
        """,
        unsafe_allow_html=True,
    )

    # Render chat history
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.markdown(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                if (
                    "tool_calls" in msg.additional_kwargs
                    and msg.additional_kwargs["tool_calls"]
                ):
                    st.markdown(f"{msg.content} (Calling tool...)")
                else:
                    # Process message content
                    content = msg.content
                    images = getattr(msg, "images", st.session_state.images)
                    videos = getattr(msg, "videos", st.session_state.videos)

                    # Split content into numbered list items or sentences
                    pattern = r"((?:[0-9]+\.\s+[^\n]*?(?=(?:[0-9]+\.\s+|\Z)))|[^\n]+)"
                    segments = re.findall(pattern, content, re.DOTALL)
                    for segment in segments:
                        segment = segment.strip()
                        if not segment:
                            continue

                        # Extract and remove image references
                        image_refs = re.findall(r"\[Image (\d+)\]", segment)
                        cleaned_segment = re.sub(r"\[Image (\d+)\]\s*", "", segment)
                        # Normalize spacing before punctuation
                        cleaned_segment = re.sub(r"\s+([.!?])", r"\1", cleaned_segment)
                        if cleaned_segment.strip():
                            st.markdown(cleaned_segment)

                        # Render images
                        for ref in image_refs:
                            idx = int(ref) - 1
                            if 0 <= idx < len(images):
                                try:
                                    caption = images[idx].get("caption", "")
                                    cleaned_caption = re.sub(
                                        r"^Figure \d+:\s*", "", caption
                                    )
                                    st.image(
                                        base64.b64decode(images[idx]["base64"]),
                                        caption=(
                                            cleaned_caption
                                            if cleaned_caption.strip()
                                            else None
                                        ),
                                        use_container_width=True,
                                        output_format="auto",
                                        clamp=True,
                                        channels="RGB",
                                    )
                                except Exception as e:
                                    logger.error(f"Image render error: {str(e)}")

                    # Render videos
                    if videos:
                        logger.info(f"Rendering videos: {videos}")
                        for video in videos:
                            try:
                                st.markdown(
                                    f"**Video**: [{video['title']}]({video['url']})"
                                )
                                st.video(video["url"])
                            except Exception as e:
                                logger.error(f"Video render error: {str(e)}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Handle user input
    question = st.chat_input(placeholder="Ask a question about MissionOS:")
    if question:
        st.session_state.images = []
        st.session_state.videos = []

        # Add user message to history
        user_message = HumanMessage(content=question)
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(question)

        # Prepare graph execution
        st.session_state.thread_id += 1
        config = {"configurable": {"thread_id": f"{st.session_state.thread_id}"}}
        system_prompt = SystemMessage(
            content=(
                "You are an assistant for MissionOS, a platform for managing instruments and data. "
                "Use the 'retrieve' tool for any ambiguous, general, or info-seeking queries about MissionOS. "
                "Only respond directly without tools for simple greetings or clear non-info requests."
            )
        )
        initial_state = {
            "messages": [system_prompt, user_message],
            "images": [],
            "videos": [],
        }

        # Process query with LangGraph
        with st.spinner("Generating..."):
            try:
                for step in st.session_state.graph.stream(
                    initial_state,
                    stream_mode="values",
                    config=config,
                ):
                    new_messages = [
                        msg for msg in step["messages"]
                        if msg not in st.session_state.messages
                    ]
                    if "videos" in step:
                        st.session_state.videos = step["videos"]
                    if "images" in step:
                        st.session_state.images = step["images"]
                    for msg in new_messages:
                        st.session_state.messages.append(msg)
                        if (
                            isinstance(msg, AIMessage)
                            and not (
                                "tool_calls" in msg.additional_kwargs
                                and msg.additional_kwargs["tool_calls"]
                            )
                        ):
                            with st.chat_message("assistant"):
                                # Process response content
                                content = msg.content
                                images = getattr(msg, "images", st.session_state.images)
                                videos = getattr(msg, "videos", st.session_state.videos)

                                # Split content into segments
                                pattern = r"((?:[0-9]+\.\s+[^\n]*?(?=(?:[0-9]+\.\s+|\Z)))|[^\n]+)"
                                segments = re.findall(pattern, content, re.DOTALL)
                                for segment in segments:
                                    segment = segment.strip()
                                    if not segment:
                                        continue

                                    # Extract and remove image references
                                    image_refs = re.findall(r"\[Image (\d+)\]", segment)
                                    cleaned_segment = re.sub(
                                        r"\[Image (\d+)\]\s*", "", segment
                                    )
                                    cleaned_segment = re.sub(
                                        r"\s+([.!?])", r"\1", cleaned_segment
                                    )
                                    if cleaned_segment.strip():
                                        st.markdown(cleaned_segment)

                                    # Render images
                                    for ref in image_refs:
                                        idx = int(ref) - 1
                                        if 0 <= idx < len(images):
                                            try:
                                                caption = images[idx].get("caption", "")
                                                cleaned_caption = re.sub(
                                                    r"^Figure \d+:\s*", "", caption
                                                )
                                                st.image(
                                                    base64.b64decode(images[idx]["base64"]),
                                                    caption=(
                                                        cleaned_caption
                                                        if cleaned_caption.strip()
                                                        else None
                                                    ),
                                                    use_container_width=True,
                                                    output_format="auto",
                                                    clamp=True,
                                                    channels="RGB",
                                                )
                                            except Exception as e:
                                                logger.error(f"Image render error: {str(e)}")

                                # Render videos
                                if videos:
                                    logger.info(f"Rendering videos: {videos}")
                                    for video in videos:
                                        try:
                                            st.markdown(
                                                f"**Video**: [{video['title']}]({video['url']})"
                                            )
                                            st.video(video["url"])
                                        except Exception as e:
                                            logger.error(f"Video render error: {str(e)}")
            except Exception as e:
                logger.error(f"Graph streaming error: {str(e)}")
                st.error(f"Failed to generate response: {str(e)}")

        # Auto-scroll to latest message
        st.markdown('<script>scrollToBottom();</script>', unsafe_allow_html=True)