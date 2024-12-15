from fastapi import APIRouter

from src.core.chat.model_loader import load_model
from langserve import add_routes

api_router = APIRouter()
add_routes(api_router, load_model(), path="/chat")
