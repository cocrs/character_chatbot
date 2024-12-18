# import chainlit as cl
# from llama_index.core import VectorStoreIndex
# from llama_index.core.chat_engine import (
#     CondensePlusContextChatEngine,
# )
# from llama_index.core.callbacks.base import CallbackManager
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.ollama import Ollama
# from llama_index.core import Settings
# from llama_index.core.storage.chat_store import SimpleChatStore
# from llama_index.core.memory import ChatMemoryBuffer

# from core.chat_handler.base import ChatHandler

# # ******* Set up context prompt and condense prompt *******
# JA_CONTEXT_PROMPT_TEMPLATE = """
#     以下は、ユーザーとクラスメート(レイナ)の間の会話です。
#     レイナは日本の女の子でっす、日本語以外の言語はしゃべれません。
#     以下に文脈に関連するドキュメントがあります：
    
#     {context_str}
#     指示：上記のドキュメントに基づいて、以下のユーザーの言葉に対しレイナの返事をしてください。
#     """

# JA_CONDENSE_PROMPT_TEMPLATE = """
#     次のユーザーとクラスメート(レイナ)の会話、およびユーザーからのフォローアップ質問を考慮し、
#     フォローアップ質問を独立した日本語の質問に言い換えてください。

#     チャット履歴:
#     {chat_history}
#     フォローアップ入力: {question}
#     独立した質問:"""


# class LlamaIndexHandler(ChatHandler):
#     def __init__(self):
#         self.embed_model = HuggingFaceEmbedding(
#             model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#         )
#         self.llm = Ollama(model="microai/suzume-llama3", request_timeout=360.0)

#         Settings.embed_model = self.embed_model
#         Settings.llm = self.llm

#         # ******* Set up chat memory buffer and chat store *******
#         chat_store = SimpleChatStore()
#         self.chat_memory = ChatMemoryBuffer.from_defaults(
#             token_limit=3000, chat_store=chat_store
#         )

#         # Not using any documents yet
#         self.documents = []

#     def process_question(self, question: str) -> str:
#         # Retrieve existing documents
#         index = VectorStoreIndex.from_documents(self.documents)

#         chat_engine = CondensePlusContextChatEngine.from_defaults(
#             retriever=index.as_retriever(),
#             embed_model=self.embed_model,
#             llm=self.llm,
#             callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
#             memory=self.chat_memory,
#             context_prompt=JA_CONTEXT_PROMPT_TEMPLATE,
#             # condense_prompt=JA_CONDENSE_PROMPT_TEMPLATE,
#             verbose=True,
#             skip_condense=True,
#         )

#         response = cl.make_async(chat_engine.chat)(question).response
#         return response
