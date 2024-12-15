import chainlit as cl
from langchain_core.messages import get_buffer_string
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langserve import RemoteRunnable

from core.chat_handler.base import ChatHandler
from core.chat_handler.langchain.memory import State, search_recall_memories, tools


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class LangchainHandler(ChatHandler):
    def __init__(self):

        self.llm = RemoteRunnable("http://localhost:4000/api/v1/chat")

        self.sync_with_current_setting(remove_memory=True)

    def sync_with_current_setting(self, remove_memory=False):
        settings = cl.user_session.get("settings")
        system_prompt = f"あなたは「assistant」として、以下のキャラクター設定と世界観の情報に基づいて「user」のメッセージに自然な返事をしてください。\n\nassistantのキャラクター設定：{settings['character_setting']}。\n世界観の情報：{settings['world_view']}"
        messages = [
            (
                "system",
                "会話の記録:\n{recall_memories}\n\n" f"{system_prompt}",
            ),
            ("placeholder", "{messages}"),
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        # FIXME: tools
        model_with_tools = self.llm

        def agent(state: State) -> State:
            """Process the current state and generate a response using the LLM.

            Args:
                state (schemas.State): The current state of the conversation.

            Returns:
                schemas.State: The updated state with the agent's response.
            """
            bound = prompt | model_with_tools
            recall_str = (
                "<recall_memory>\n"
                + "\n".join(state["recall_memories"])
                + "\n</recall_memory>"
            )
            prediction = bound.invoke(
                {
                    "messages": state["messages"],
                    "recall_memories": recall_str,
                }
            )
            return {
                "messages": [prediction],
            }

        def load_memories(state: State, config: RunnableConfig) -> State:
            """Load memories for the current conversation.

            Args:
                state (schemas.State): The current state of the conversation.
                config (RunnableConfig): The runtime configuration for the agent.

            Returns:
                State: The updated state with loaded memories.
            """
            convo_str = get_buffer_string(state["messages"])
            # FIXME
            # convo_str = self.llm.tokenizer.decode(
            #     self.llm.tokenizer.encode(convo_str)[:2048]
            # )
            recall_memories = search_recall_memories.invoke(convo_str, config)
            return {
                "recall_memories": recall_memories,
            }

        def route_tools(state: State):
            """Determine whether to use tools or end the conversation based on the last message.

            Args:
                state (schemas.State): The current state of the conversation.

            Returns:
                Literal["tools", "__end__"]: The next step in the graph.
            """
            msg = state["messages"][-1]
            if msg.tool_calls:
                return "tools"

            return END

        builder = StateGraph(State)
        builder.add_node(load_memories)
        builder.add_node(agent)
        builder.add_node("tools", ToolNode(tools))

        # Add edges to the graph
        builder.add_edge(START, "load_memories")
        builder.add_edge("load_memories", "agent")
        builder.add_conditional_edges("agent", route_tools, ["tools", END])
        builder.add_edge("tools", "agent")

        # Compile the graph
        memory = MemorySaver()
        self.runnable = builder.compile(checkpointer=memory)

    async def process_question(self, question: str) -> str:
        def pretty_print_stream_chunk(chunk):
            for node, updates in chunk.items():
                print(f"Update from node: {node}")
                if "messages" in updates:
                    print(updates["messages"])
                    updates["messages"][-1].pretty_print()
                else:
                    print(updates)

                print("\n")

        messages = []
        config = {"configurable": {"user_id": "1", "thread_id": "1"}}
        for chunk in self.runnable.stream(
            {"messages": [("user", question)]}, config=config
        ):
            if "messages" in chunk:
                messages.extend(chunk["messages"])
            pretty_print_stream_chunk(chunk)

        # FIXME: session_id
        response = await cl.make_async(self.runnable.invoke)(
            {"messages": [("user", question)]},
            config={"configurable": {"user_id": "1", "thread_id": "1"}},
        )
        return response["messages"][-1].content
