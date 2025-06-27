from .BaseController import BaseController
from ..models import ResponseSignal
from ..RAG import PDFRAG
from ..helpers import get_settings
from langchain_groq import ChatGroq
import os

groq_api_key = get_settings().GROQ_API_KEY
groq_llm = ChatGroq(
    groq_api_key=groq_api_key,  
    model_name="gemma2-9b-it",      
    temperature=0
)

PDF_FILENAME = "Hands_On_Machine_Learning_with_Scikit_Learn,_Keras,_and_TensorFlow 3th edition chapter 1.pdf"
PATH_FILE = os.path.join(os.path.dirname(__file__), "../uploads", PDF_FILENAME)
PATH_FILE = os.path.abspath(PATH_FILE)

rag_process = PDFRAG(PATH_FILE, chunk_size=300, llm=groq_llm)
rag_process.load_or_build_db()

class ChatController(BaseController):
    def __init__(self):
        super().__init__()

    def get_user_question(self, question: str):

        if question.strip() == "":
            return False, ResponseSignal.INVALID_INPUT_QUESTION.value

        answer = rag_process.ask_question(question)

        if not answer:
            return False, ResponseSignal.NO_VALID_ANSWER.value

        if answer.strip() == "":
            return False, ResponseSignal.INVALID_ANSWER_EMPTY.value

        return True, answer
    