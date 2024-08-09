import chainlit as cl
from langchain.schema.runnable import Runnable
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from config import *

from core.text2speech import text2speech


class LangchainHandler:
    def __init__(self):
        llm = ChatOllama(
            model="microai/suzume-llama3",
            temperature=0,
            # other params...
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that answers questions about the topic.",
                ),
                ("human", "{input}"),
            ]
        )
        runnable = prompt | llm
        cl.user_session.set("runnable", runnable)

    async def process_question(self, question):
        runnable = cl.user_session.get("runnable")  # type: Runnable

        response = await runnable.ainvoke(question)
        text2speech(response.content, AUDIO_OUTPUT_PATH)

        response_message = cl.Message(
            content="",
            elements=[
                cl.Audio(
                    name="audio", path=AUDIO_OUTPUT_PATH, display="inline", auto_play=True
                ),
            ],
        )

        for token in response.content:
            await response_message.stream_token(token=token)

        await response_message.send()
