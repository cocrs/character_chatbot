import chainlit as cl
from core.langchain import LangchainHandler
from core.llama_index import LlamaIndexHandler


def llama_index_initialize():
    cl.user_session.set("documents", [])
    return LlamaIndexHandler()


def langchain_initialize():
    return LangchainHandler()
