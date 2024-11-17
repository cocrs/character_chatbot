import chainlit as cl
from core.agent import Agent
from core.chat_handler.langchain import LangchainHandler
# from core.chat_handler.llama_index import LlamaIndexHandler
from config import config


def initialize() -> None:
    if config.mode == "langchain":
        handler = langchain_initialize()
    # elif config.mode == "llama_index":
    #     handler = llama_index_initialize()
    agent = Agent(handler)
    cl.user_session.set("agent", agent)


# def llama_index_initialize():
#     return LlamaIndexHandler()


def langchain_initialize():
    return LangchainHandler()
