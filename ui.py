"""Render the chatbot user interface for the MissionHelp Demo application.

This module defines the Streamlit-based UI, handling chat history display,
user input, and multimedia rendering (images, videos) for MissionOS queries.
"""

import base64
import logging
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage


# Configure logging for UI events and errors
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def render_chatbot() -> None:
    """Renders the chatbot interface and processes user interactions."""
    # Apply custom CSS and JavaScript
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
                if "tool_calls" in msg.additional_kwargs and msg.additional_kwargs["tool_calls"]:
                    st.markdown(f"{msg.content} (Calling tool...)")
                else:
                    content = msg.content
                    images = msg.additional_kwargs.get("images", [])
                    videos = msg.additional_kwargs.get("videos", [])

                    segments = re.findall(r"((?:[0-9]+\.\s+[^\n]*?(?=(?:[0-9]+\.\s+|\Z)))|[^\n]+)", content, re.DOTALL)
                    for segment in segments:
                        segment = segment.strip()
                        if not segment:
                            continue

                        image_refs = re.findall(r"\[Image (\d+)\]", segment)
                        cleaned_segment = re.sub(r"\[Image (\d+)\]\s*", "", segment)
                        cleaned_segment = re.sub(r"\s+([.!?])", r"\1", cleaned_segment)
                        if cleaned_segment.strip():
                            st.markdown(cleaned_segment)

                        for ref in image_refs:
                            idx = int(ref) - 1
                            if 0 <= idx < len(images):
                                caption = images[idx].get("caption", "")
                                cleaned_caption = re.sub(r"^Figure \d+:\s*", "", caption)
                                st.image(
                                    base64.b64decode(images[idx]["base64"]),
                                    caption=cleaned_caption if cleaned_caption.strip() else None,
                                    use_container_width=True,
                                    output_format="auto",
                                    clamp=True,
                                    channels="RGB",
                                )

                    for video in videos:
                        st.markdown(f"**Video**: [{video['title']}]({video['url']})")
                        st.video(video["url"])

    st.markdown("</div>", unsafe_allow_html=True)

    # Handle user input
    if question := st.chat_input(placeholder="Ask a question about MissionOS:"):
        st.session_state.images = []
        st.session_state.videos = []
        user_message = HumanMessage(content=question)
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(question)

        config = {"configurable": {"thread_id": f"{st.session_state.thread_id}"}}
        initial_state = {
            "messages": [user_message],
            "images": [],
            "videos": [],
            "timings": [],
        }

        with st.spinner("Generating..."):
            for step in st.session_state.graph.stream(initial_state, stream_mode="values", config=config):
                # Render new AI messages incrementally
                new_messages = [msg for msg in step["messages"] if msg not in st.session_state.messages]
                for msg in new_messages:
                    st.session_state.messages.append(msg)
                    if isinstance(msg, AIMessage):
                        with st.chat_message("assistant"):
                            if "tool_calls" in msg.additional_kwargs and msg.additional_kwargs["tool_calls"]:
                                st.markdown(f"{msg.content} (Calling tool...)")
                            else:
                                content = msg.content
                                images = msg.additional_kwargs.get("images", [])
                                videos = msg.additional_kwargs.get("videos", [])

                                segments = re.findall(
                                    r"((?:[0-9]+\.\s+[^\n]*?(?=(?:[0-9]+\.\s+|\Z)))|[^\n]+)", content, re.DOTALL
                                )
                                for segment in segments:
                                    segment = segment.strip()
                                    if not segment:
                                        continue

                                    image_refs = re.findall(r"\[Image (\d+)\]", segment)
                                    cleaned_segment = re.sub(r"\[Image (\d+)\]\s*", "", segment)
                                    cleaned_segment = re.sub(r"\s+([.!?])", r"\1", cleaned_segment)
                                    if cleaned_segment.strip():
                                        st.markdown(cleaned_segment)

                                    for ref in image_refs:
                                        idx = int(ref) - 1
                                        if 0 <= idx < len(images):
                                            caption = images[idx].get("caption", "")
                                            cleaned_caption = re.sub(r"^Figure \d+:\s*", "", caption)
                                            st.image(
                                                base64.b64decode(images[idx]["base64"]),
                                                caption=cleaned_caption if cleaned_caption.strip() else None,
                                                use_container_width=True,
                                                output_format="auto",
                                                clamp=True,
                                                channels="RGB",
                                            )

                                for video in videos:
                                    st.markdown(f"**Video**: [{video['title']}]({video['url']})")
                                    st.video(video["url"])

                final_state = step

            if final_state:
                with st.expander("Latency Details", expanded=True):
                    timings = final_state.get("timings", [])
                    total_latency = sum(timing["time"] for timing in timings)
                    
                    latency_data = [
                        {
                            "Component": timing["component"],
                            "Latency (s)": f"{timing['time']:.2f}",
                            "Percentage (%)": f"{(timing['time'] / total_latency * 100):.0f}" if total_latency > 0 else "0"
                        }
                        for timing in timings
                    ]
                    
                    df = pd.DataFrame(latency_data)
                    
                    st.markdown(f"**Total Latency: {total_latency:.2f} seconds**")
                    
                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                    if timings and total_latency > 0:
                        fig, ax = plt.subplots()
                        components = [timing["component"] for timing in timings]
                        percentages = [(timing["time"] / total_latency * 100) for timing in timings]
                        pastel_colors = cm.Pastel1(range(len(components)))
                        ax.pie(percentages, labels=components, autopct='%1.0f%%', startangle=90, colors=pastel_colors)
                        ax.axis('equal')
                        st.pyplot(fig)
                        plt.close(fig)

        st.markdown('<script>scrollToBottom();</script>', unsafe_allow_html=True)