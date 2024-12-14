from fastapi import APIRouter

import chat.services as services_chat
import chat.schemas as schemas_chat

from langchain_core.messages import BaseMessage

router = APIRouter()


@router.post("/invoke", response_model=BaseMessage)
def invoke(
    request: schemas_chat.Request,
):
    pred = services_chat.invoke(request.input)
    return pred
