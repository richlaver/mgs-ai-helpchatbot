"""Define state classes for the MissionHelp Demo application.

This module provides typed dictionaries to structure the application's state, used
in the LangGraph workflow for managing queries, responses, and multimedia content.
"""

from typing import List, Optional, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

# from langchain_core.documents import Document
# from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


# class State(TypedDict):
#     """State for query processing in the RAG pipeline.

#     Attributes:
#         question: The user's query string.
#         context: List of retrieved documents providing context.
#         answer: The generated answer based on context.
#     """
#     question: str
#     context: List[Document]
#     answer: str


class State(TypedDict):
    """State for managing conversation and multimedia in LangGraph.

    Attributes:
        messages: List of conversation messages (human, AI, or tool).
        images: Optional list of image dictionaries with base64 data and captions.
        videos: Optional list of video dictionaries with URLs and titles.
        timings: List of timing dictionaries with node, time, and component details.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    images: Optional[List[dict]]
    videos: Optional[List[dict]]
    timings: List[dict]