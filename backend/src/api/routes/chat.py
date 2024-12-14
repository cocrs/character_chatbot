from time import clock_getres
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

import chat.services as services_chat
import chat.schemas as schemas_chat

from langchain_core.language_models.base import LanguageModelInput

router = APIRouter()


@router.post("/invoke")
def invoke(
    input: LanguageModelInput,
):
    return services_chat.invoke(input)