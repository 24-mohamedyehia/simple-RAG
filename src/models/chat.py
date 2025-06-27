from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    question: str = Field(..., description="The question to ask")