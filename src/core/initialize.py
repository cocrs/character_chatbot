import chainlit as cl
from chainlit.input_widget import TextInput
from core.agent import Agent
from core.chat_handler.langchain import LangchainHandler

# from core.chat_handler.llama_index import LlamaIndexHandler
from config import config


async def createDefaultSettings() -> None:
    settings = await cl.ChatSettings(
        [
            TextInput(
                id="character_setting",
                label="Character Setting",
                placeholder="キャラクター設定を入力してください",
                initial="カフェの常連の女性。名前はアンナ。趣味は読書、フレンドリーだが少しシャイ",
            ),
            TextInput(
                id="world_view",
                label="World View",
                placeholder="世界観を入力してください",
                initial="都会の喧騒の中、小さなカフェが人々の憩いの場となっている",
            ),
        ]
    ).send()
    cl.user_session.set("settings", settings)


async def initialize() -> None:
    print("Initializing settings...")
    await createDefaultSettings()
    print("Initializing agent...")
    if config.mode == "langchain":
        handler = langchain_initialize()
    # elif config.mode == "llama_index":
    #     handler = llama_index_initialize()
    agent = Agent(handler)
    cl.user_session.set("agent", agent)


# def llama_index_initialize():
#     return LlamaIndexHandler()


def langchain_initialize():
    return LangchainHandler()
