import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

from config import config
from core.chat_handler.base import ChatHandler


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class LangchainHandler(ChatHandler):
    def __init__(self):
        # load character dialogue
        loader = JSONLoader(
            file_path=config.document_path, jq_schema=".dialogue[].content"
        )
        docs = loader.load()

        llm = ChatOllama(
            model="llama3.1",
            top_p=0.2,
            repeat_penalty=1.5,
            # other params...
        )
        system_prompt = (
            f"以下の内容はあなたの記憶です。"
            "この記憶を基に、あなたらしい話し方で相手と会話をしてください。"
            "\n\n"
            f"{format_docs(docs)}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{input}"),
            ]
        )
        runnable = prompt | llm | StrOutputParser()
        self.runnable = runnable

        if config.use_chat_history:
            # delete memory.db if it exists
            if os.path.exists("memory.db"):
                os.remove("memory.db")

            self.runnable = RunnableWithMessageHistory(
                # The underlying runnable
                runnable,
                # A function that takes in a session id and returns a memory object
                self.get_session_history,
                input_messages_key="input",
                history_messages_key="history",
            )

    def get_session_history(self, session_id: str):
        return SQLChatMessageHistory(session_id, "sqlite:///memory.db")

    def process_question(self, question: str) -> str:
        # FIXME: session_id
        response = self.runnable.invoke(
            {"input": question}, config={"configurable": {"session_id": "1"}}
        )
        return response
