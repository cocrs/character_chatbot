import chainlit as cl
from langchain.schema.runnable import Runnable
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from config import config
from core.text2speech import text2speech


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


class LangchainHandler:
    def __init__(self):
        loader = JSONLoader(
            file_path=config.document_path, jq_schema=".dialogue[].content"
        )
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        docs = loader.load()
        all_splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=OllamaEmbeddings(model="nomic-embed-text"),
        )
        self.retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        )

        llm = ChatOllama(
            model="microai/suzume-llama3",
            top_p=0.5,
            repeat_penalty=1.2,
            # other params...
        )
        system_prompt = (
            "以下の内容はあなたが話した言葉をあつめたもの。"
            "この内容をもとに，自分の個性、気持ち、話し方、話の長さを考えて，返事をしてください。"
            "答えが分からない場合は、無理に返事しなくでもいい。"
            "\n\n"
            f"{format_docs(docs)}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        runnable = prompt | llm | StrOutputParser()
        self.runnable = runnable

    async def process_question(self, question):
        response = self.runnable.invoke(question)

        elements = []
        if config.tts:
            text2speech(response.content, config.audio_output_path)
            elements.append(
                cl.Audio(
                    name="audio",
                    path=config.audio_output_path,
                    display="inline",
                    auto_play=True,
                ),
            )
        response_message = cl.Message(content="", elements=elements)

        for token in response:
            await response_message.stream_token(token=token)

        await response_message.send()
