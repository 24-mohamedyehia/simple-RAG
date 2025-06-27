from langchain_groq import ChatGroq
from .src.RAG import PDFRAG
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

groq_llm = ChatGroq(
    groq_api_key=groq_api_key,  
    model_name="gemma2-9b-it",      
    temperature=0
)
PATH_FILE = './uploads/Hands_On_Machine_Learning_with_Scikit_Learn,_Keras,_and_TensorFlow 3th edition chapter 1.pdf'

rag_process = PDFRAG(PATH_FILE, chunk_size= 300, llm= groq_llm)

rag_process.load_or_build_db()

while True:
    user_question = input("Ask a question about the PDF content (or type 'exit' to quit): ")

    if user_question.lower() == 'exit':
        break

    answer = rag_process.ask_question(user_question)

    print(answer)

    print("="*50)
