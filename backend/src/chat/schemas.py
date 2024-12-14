from src.schemas import CustomModel

from langchain_core.language_models.base import LanguageModelInput


class Request(CustomModel):
    input: LanguageModelInput
