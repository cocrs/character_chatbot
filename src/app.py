from io import BytesIO
import chainlit as cl
from chainlit.element import ElementBased

from core.agent import Agent
from core.speech_recognition import transcribe_audio
from core.initialize import initialize


@cl.on_chat_start
async def start():
    await cl.Message(content="Welcome! Please ask your question.").send()
    await initialize()
    cl.user_session.set("audio_buffer", None)
    cl.user_session.set("audio_mime_type", None)


@cl.on_settings_update
async def setup_agent(settings):
    print("on_settings_update", settings)
    # TODO: should not be able to change setting after chat has started
    if "character_setting" in settings or "world_view" in settings:
        agent: Agent = cl.user_session.get("agent")
        agent.chat_handler.sync_with_current_setting()


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
    agent: Agent = cl.user_session.get("agent")
    await agent.process(question)


@cl.on_message
async def on_message(message: cl.Message):
    agent: Agent = cl.user_session.get("agent")
    await agent.process(message.content)
