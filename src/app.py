from io import BytesIO
import chainlit as cl
from chainlit.element import ElementBased
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import (
    CondensePlusContextChatEngine,
)
from llama_index.core.callbacks.base import CallbackManager
from utils.speech_recognition import transcribe_audio
from utils.text2speech import text2speech
from config import *
# from utils.voice_cloning.bark_voice_cloner import BarkVoiceCloner


@cl.on_chat_start
async def factory():
    await cl.Message(content="Welcome! Please ask your question.").send()
    cl.user_session.set("documents", [])
    cl.user_session.set("audio_buffer", None)
    cl.user_session.set("audio_mime_type", None)


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)


@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    # Get the audio buffer from the session
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0)  # Move the file pointer to the beginning
    # audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")

    # Transcribe the audio
    question = transcribe_audio(audio_buffer.read(), audio_mime_type)

    await cl.Message(content=f"Question: {question}").send()

    # Continue with the Chainlit processing
    await process_question(question)


@cl.on_message
async def on_message(message: cl.Message):
    await process_question(message.content)


async def process_question(question):
    # Retrieve existing documents
    documents = cl.user_session.get("documents")

    # TODO: logic to determine if we need to search
    # google_tool = GoogleSearchToolSpec(key=GOOGLE_API_KEY, engine=ENGINE_ID, num=10)
    # new_documents = google_tool.google_search(question)
    # documents.extend(new_documents)  # Append new documents to existing ones
    cl.user_session.set("documents", documents)

    index = VectorStoreIndex.from_documents(documents)

    chat_engine = CondensePlusContextChatEngine.from_defaults(
        retriever=index.as_retriever(),
        embed_model=embed_model,
        llm=llm,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
        memory=chat_memory,
        context_prompt=JA_CONTEXT_PROMPT_TEMPLATE,
        # condense_prompt=JA_CONDENSE_PROMPT_TEMPLATE,
        verbose=True,
        skip_condense=True,
    )

    # chat_engine = index.as_chat_engine(chat_mode="react", llm=llm)
    cl.user_session.set("chat_engine", chat_engine)

    response = await cl.make_async(chat_engine.chat)(question)

    text2speech(response.response, AUDIO_OUTPUT_PATH)
    # TODO: bark voice cloning, too slow now
    # cloner = BarkVoiceCloner()
    # cloner.generate(response.response)
    
    response_message = cl.Message(
        content="",
        elements=[
            cl.Audio(
                name="audio", path=AUDIO_OUTPUT_PATH, display="inline", auto_play=True
            ),
        ],
    )

    for token in response.response:
        await response_message.stream_token(token=token)

    await response_message.send()
