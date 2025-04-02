from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st

def render_chatbot():
    # Custom CSS for scrolling and fixed input
    st.markdown("""
        <style>
        .chat-messages {
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 60px; /* Space for fixed input */
        }
        .stChatInput {
            position: fixed;
            bottom: 10px;
            width: 95%;
            background-color: white;
            z-index: 1000;
        }
        </style>
    """, unsafe_allow_html=True)

    # JavaScript for auto-scrolling
    st.markdown("""
        <script>
        function scrollToBottom() {
            const chatMessages = document.querySelector('.chat-messages');
            if (chatMessages) {
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }
        }
        const observer = new MutationObserver(scrollToBottom);
        const chatElement = document.querySelector('.chat-messages');
        if (chatElement) {
            observer.observe(chatElement, { childList: true, subtree: true });
        }
        </script>
    """, unsafe_allow_html=True)

    # Chat messages area
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    
    # Render existing messages only once
    for msg in st.session_state.messages:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            if "tool_calls" in msg.additional_kwargs and msg.additional_kwargs["tool_calls"]:
                with st.chat_message("assistant"):
                    st.write(f"{msg.content} (Calling tool...)")
            elif "\n\n" in msg.content and "Source:" in msg.content and "Content:" in msg.content:
                with st.chat_message("system"):
                    st.write(f"**Retrieved Info:**\n{msg.content}")
            else:
                with st.chat_message("assistant"):
                    st.write(msg.content)
    
    # Placeholder for streaming new messages
    message_placeholder = st.empty()
    
    # Fixed chat input
    question = st.chat_input(placeholder="Ask a question about MissionOS:")
    if question:
        # New thread for each query
        st.session_state.thread_id += 1
        config = {"configurable": {"thread_id": f"{st.session_state.thread_id}"}}

        # Initial state
        system_prompt = SystemMessage(
            content="You are an assistant for MissionOS, a platform for managing instruments and data. "
                    "Use the 'retrieve' tool for any ambiguous, general, or info-seeking queries about MissionOS. "
                    "Only respond directly without tools for simple greetings or explicit non-info requests."
        )
        user_message = HumanMessage(content=question)
        initial_state = {"messages": [system_prompt, user_message]}

        # Add user message to history and display immediately
        if not any(m.content == question and isinstance(m, HumanMessage) for m in st.session_state.messages):
            st.session_state.messages.append(user_message)
            with st.chat_message("user"):
                st.write(question)

        # Stream new messages into the placeholder
        with message_placeholder.container():
            for step in st.session_state.graph.stream(
                initial_state,
                stream_mode="values",
                config=config,
            ):
                new_messages = [msg for msg in step["messages"][1:] if msg not in st.session_state.messages]  # Skip system_prompt and duplicates
                for msg in new_messages:
                    if isinstance(msg, AIMessage):
                        if "tool_calls" in msg.additional_kwargs and msg.additional_kwargs["tool_calls"]:
                            with st.chat_message("assistant"):
                                st.write(f"{msg.content} (Calling tool...)")
                        elif "\n\n" in msg.content and "Source:" in msg.content and "Content:" in msg.content:
                            with st.chat_message("system"):
                                st.write(f"**Retrieved Info:**\n{msg.content}")
                        else:
                            with st.chat_message("assistant"):
                                st.write(msg.content)
                    st.session_state.messages.append(msg)
    
    st.markdown('</div>', unsafe_allow_html=True)