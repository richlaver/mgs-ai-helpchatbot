from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

def build_graph(llm, vector_store):
    st.session_state.status.write(':material/lan: Building LangGraph graph...')
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        """Retrieve detailed information about MissionOS for any query that seeks knowledge, context, or clarification.
        Use this tool whenever a user asks about MissionOS, needs more details, or provides an ambiguous or general request
        (e.g., 'Tell me about it', 'What is this?', 'How does it work?').
        Returns up to 4 relevant documents from the MissionOS vector store."""
        retrieved_docs = vector_store.similarity_search(query, k=4)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs
    

    # System message to guide LLM behavior
    system_prompt = SystemMessage(
        content="You are an assistant for MissionOS, a platform for managing instruments and data. "
                "Use the 'retrieve' tool to fetch additional context for any query that is ambiguous, "
                "general, or requires specific MissionOS information (e.g., setup, usage, troubleshooting). "
                "Only respond directly without tools for simple greetings or clear non-info requests."
    )
    

    # Step 1: Generate an AIMessage that may include a tool-call to be sent.
    def query_or_respond(state: MessagesState):
        """Generate tool call for retrieval or respond."""
        llm_with_tools = llm.bind_tools([retrieve])

        # Prepend system prompt if not already in state
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [system_prompt] + messages

        response = llm_with_tools.invoke(messages)
        # MessagesState appends messages to state instead of overwriting
        return {"messages": [response]}


    # Step 2: Execute the retrieval.
    tools = ToolNode([retrieve])


    # Step 3: Generate a response using the retrieved content.
    def generate(state: MessagesState):
        """Generate answer."""
        # Get generated ToolMessages
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        # Format into prompt
        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks about MissionOS. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know."
            "\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system")
            or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        # Run
        response = llm.invoke(prompt)
        return {"messages": [response]}
    

    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    memory = MemorySaver()
    return graph_builder.compile(checkpointer=memory)