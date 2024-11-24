import json
from typing import List, Literal, Optional

import tiktoken
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

import uuid

recall_vector_store = InMemoryVectorStore(OpenAIEmbeddings())

def get_user_id(config: RunnableConfig) -> str:
    user_id = config["configurable"].get("user_id")
    if user_id is None:
        raise ValueError("User ID needs to be provided to save a memory.")

    return user_id


@tool
def save_recall_memory(memory: str, config: RunnableConfig) -> str:
    """Save memory to vectorstore for later semantic retrieval."""
    user_id = get_user_id(config)
    document = Document(
        page_content=memory, id=str(uuid.uuid4()), metadata={"user_id": user_id}
    )
    recall_vector_store.add_documents([document])
    return memory


@tool
def search_recall_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search for relevant memories."""
    user_id = get_user_id(config)

    def _filter_function(doc: Document) -> bool:
        return doc.metadata.get("user_id") == user_id

    documents = recall_vector_store.similarity_search(
        query, k=3, filter=_filter_function
    )
    return [document.page_content for document in documents]

tools = [save_recall_memory, search_recall_memories]


class State(MessagesState):
    # add memories that will be retrieved based on the conversation context
    recall_memories: List[str]
    
# TODO: refactor to use langgraph

# def agent(state: State) -> State:
#     """Process the current state and generate a response using the LLM.

#     Args:
#         state (schemas.State): The current state of the conversation.

#     Returns:
#         schemas.State: The updated state with the agent's response.
#     """
#     bound = prompt | model_with_tools
#     recall_str = (
#         "<recall_memory>\n" + "\n".join(state["recall_memories"]) + "\n</recall_memory>"
#     )
#     prediction = bound.invoke(
#         {
#             "messages": state["messages"],
#             "recall_memories": recall_str,
#         }
#     )
#     return {
#         "messages": [prediction],
#     }


# def load_memories(state: State, config: RunnableConfig) -> State:
#     """Load memories for the current conversation.

#     Args:
#         state (schemas.State): The current state of the conversation.
#         config (RunnableConfig): The runtime configuration for the agent.

#     Returns:
#         State: The updated state with loaded memories.
#     """
#     convo_str = get_buffer_string(state["messages"])
#     convo_str = tokenizer.decode(tokenizer.encode(convo_str)[:2048])
#     recall_memories = search_recall_memories.invoke(convo_str, config)
#     return {
#         "recall_memories": recall_memories,
#     }


# def route_tools(state: State):
#     """Determine whether to use tools or end the conversation based on the last message.

#     Args:
#         state (schemas.State): The current state of the conversation.

#     Returns:
#         Literal["tools", "__end__"]: The next step in the graph.
#     """
#     msg = state["messages"][-1]
#     if msg.tool_calls:
#         return "tools"

#     return END