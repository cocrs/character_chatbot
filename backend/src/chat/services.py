from logger import logger

from core.chat.model_loader import ChatModelLoader
from langchain_core.language_models.base import LanguageModelInput

loader = ChatModelLoader()


def invoke(input: LanguageModelInput):
    logger.info(f"Input: {input}")
    prediction = loader.runnable.invoke(input)
    logger.info(f"Output: {prediction}")
    return prediction
