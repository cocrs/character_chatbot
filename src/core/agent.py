import chainlit as cl
from omegaconf import OmegaConf
from config import config
from core.chat_handler.base import ChatHandler
from core.tts.bark import text2speech
from core.tts.fish.generate import LlamaGenerator
from core.tts.fish.inference import VQGANInference


class Agent:
    def __init__(self, chat_handler: ChatHandler):
        self.chat_handler = chat_handler
        if config.tts == "fish":
            # register eval resolver
            if not OmegaConf.has_resolver("eval"):
                OmegaConf.register_new_resolver("eval", eval)
            conf = OmegaConf.load("./configs/fish.yaml")
            conf_dict = OmegaConf.to_container(conf)
            self.fish_generator = LlamaGenerator(
                **conf_dict["generate"], device=config.device
            )
            self.fish_inference = VQGANInference(
                **conf_dict["inference"],
                output_path=config.audio_output_path,
                device=config.device,
            )

    async def process(self, question: str) -> None:
        response = await self.chat_handler.process_question(question)

        elements = []
        if config.tts is not None:
            if config.tts == "bark":
                text2speech(response, config.audio_output_path)
            elif config.tts == "fish":
                await cl.make_async(self.fish_generator.run)(response)
                await cl.make_async(self.fish_inference.run)()
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
