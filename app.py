from fastapi import FastAPI
from src.routes import base_router, chat_router

app = FastAPI()

app.include_router(base_router)
app.include_router(chat_router)
