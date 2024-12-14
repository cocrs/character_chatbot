from core.chat.model_loader import ChatModelLoader
from langchain_core.language_models.base import LanguageModelInput

loader = ChatModelLoader()


def invoke(input: LanguageModelInput):
    prediction = loader.runnable.invoke(input)
    return {
        "message": prediction,
    }
