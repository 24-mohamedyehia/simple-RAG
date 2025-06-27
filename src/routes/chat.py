from fastapi import APIRouter
from ..models import ChatRequest
from ..controllers import ChatController
from fastapi.responses import JSONResponse
from fastapi import status

chat_router = APIRouter(
    prefix="/v1/chat",
    tags=["chat"]
)

@chat_router.post("/")
async def create_chat(request: ChatRequest): 

    data_controller = ChatController()

    is_valid, answer = data_controller.get_user_question(request.question)
    
    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "answer": answer
            }
        )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "answer": answer
        }
    )

