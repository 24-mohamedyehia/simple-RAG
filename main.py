from langchain_groq import ChatGroq
from app import PDFRAG
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

groq_llm = ChatGroq(
    groq_api_key=groq_api_key,  
    model_name="gemma2-9b-it",      
    temperature=0
)
PATH_FILE = './uploads/Lecture-1.pdf'

rag_process = PDFRAG(PATH_FILE, chunk_size= 300, llm= groq_llm)

user_question = input("Enter your question: ")

answer = rag_process.ask_question(user_question)

print(answer)
