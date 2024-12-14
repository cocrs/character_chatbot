from fastapi import APIRouter

from api.routes import chat

api_router = APIRouter()
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])