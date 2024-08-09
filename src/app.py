from io import BytesIO
import chainlit as cl
from chainlit.element import ElementBased

from core.speech_recognition import transcribe_audio
from config import mode
from core.initialize import langchain_initialize, llama_index_initialize

@cl.on_chat_start
async def factory():
    await cl.Message(content="Welcome! Please ask your question.").send()
    if mode == "langchain":
        handler = langchain_initialize()
    elif mode == "llama_index":
        handler = llama_index_initialize()
    cl.user_session.set("handler", handler)
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
    handler = cl.user_session.get("handler")
    await handler.process_question(question)


@cl.on_message
async def on_message(message: cl.Message):
    handler = cl.user_session.get("handler")
    await handler.process_question(message.content)