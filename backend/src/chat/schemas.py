from src.schemas import CustomModel

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages import BaseMessage


class Input(CustomModel):
    input: LanguageModelInput


class Prediction(CustomModel):
    message: BaseMessage
