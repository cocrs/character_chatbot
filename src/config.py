import os
from dotenv import load_dotenv
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import Settings
import torch

# ******* Load environment variables *******
load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
ENGINE_ID = os.environ.get("ENGINE_ID")

# ******* Set up embedding model and LLM *******
device = "cuda" if torch.cuda.is_available() else "cpu"

embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
llm = Ollama(model="microai/suzume-llama3", request_timeout=360.0)

Settings.embed_model = embed_model
Settings.llm = llm

# ******* Set up chat memory buffer and chat store *******
chat_store = SimpleChatStore()
chat_memory = ChatMemoryBuffer.from_defaults(token_limit=3000, chat_store=chat_store)


# ******* Set up audio output path *******
AUDIO_OUTPUT_PATH = "output.mp3"


# ******* Set up context prompt and condense prompt *******
JA_CONTEXT_PROMPT_TEMPLATE = """
    以下は、ユーザーとクラスメートAの間の会話です。
    Aは日本の女の子でっす、日本語以外の言語はしゃべれません。

    Here are the relevant documents for the context:

    {context_str}

    Instruction: Based on the above documents, provide a detailed answer for the user question below.
    """

JA_CONDENSE_PROMPT_TEMPLATE = """
    次のユーザーとクラスメート(レイナ)の会話、およびユーザーからのフォローアップ質問を考慮し、
    フォローアップ質問を独立した日本語の質問に言い換えてください。

    チャット履歴:
    {chat_history}
    フォローアップ入力: {question}
    独立した質問:"""
