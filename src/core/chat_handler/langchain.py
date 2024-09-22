import os
from langchain_huggingface import ChatHuggingFace
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory

from config import config
from core.chat_handler.base import ChatHandler
import chainlit as cl

from langchain_huggingface.llms import HuggingFacePipeline
from unsloth import FastLanguageModel
from transformers import pipeline


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class LangchainHandler(ChatHandler):
    def __init__(self):
        # llm = ChatOllama(
        #     model="llama3.1-RP",
        #     top_p=0.2,
        #     repeat_penalty=1.5,
        #     # other params...
        # )
        model, tokenizer = FastLanguageModel.from_pretrained(
            "cyberagent/Mistral-Nemo-Japanese-Instruct-2408",
            max_seq_length=2048,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)
        llm = ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe), tokenizer=tokenizer)

        system_prompt = f"あなたは「assistant」として、以下のキャラクター設定と世界観の情報に基づいて「user」のメッセージに自然な返事をしてください。\n\nassistantのキャラクター設定：カフェの常連の男性。名前は拓也。趣味は写真で、フレンドリーだが少しシャイ。\n世界観の情報：都会の喧騒の中、小さなカフェが人々の憩いの場となっている。\n\nuserのことは、会話の中で理解できます。"
        messages = [
            ("system", system_prompt),
        ]
        if config.use_chat_history:
            messages.append(MessagesPlaceholder(variable_name="history"))
        messages.append(("user", "{input}"))
        prompt = ChatPromptTemplate.from_messages(messages)

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

    async def process_question(self, question: str) -> str:
        # FIXME: session_id
        response = await cl.make_async(self.runnable.invoke)(
            {"input": question}, config={"configurable": {"session_id": "1"}}
        )
        return response
