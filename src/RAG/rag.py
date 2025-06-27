from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import NLTKTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
load_dotenv()
import os
import nltk
nltk.download('punkt')

cohere_api_key = os.getenv('COHERE_API_KEY')

class PDFRAG:
    """
    Class to handle PDF RAG (Retrieval-Augmented Generation) using Cohere embeddings and Chroma vector store.
    This class loads a PDF, splits it into chunks, builds or loads a vector store, and allows querying the PDF content.
    Attributes:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Size of text chunks to split the PDF content into.
        llm (object): Language model for generating responses.
        db_dir (str): Directory to store the vector database.   
    """
    def __init__(
            self, pdf_path: str, 
            chunk_size: int,
            llm: object,
            db_dir: str = './pdf_chroma_db'
            ):
        self.pdf_path = pdf_path
        self.db_dir = db_dir
        self.chunk_size = chunk_size
        self.vector_db = None
        self.llm = llm

        self.embedding_model = CohereEmbeddings(
            cohere_api_key=cohere_api_key,
            model='embed-multilingual-light-v3.0'
            )
        
    def load_pdf(self):
        """ Load and split the PDF file into pages.
            """
        try:
            pdf_loader = PyPDFLoader(self.pdf_path)
            pages = pdf_loader.load_and_split()
            if not pages:
                raise ValueError('PDF File is Empty !!')
            documents = [page.page_content for page in pages]
            metadatas = [page.metadata for page in pages]
            return documents , metadatas
        except Exception as e:
            print(f"Error For read pdf {e}")
            return None, None
    
    def split_text(self, documents, metadatas):
        """ Split the text content into chunks.
        Args:
            documents (list): List of text content from the PDF.
        """ 
        text_splitter = NLTKTextSplitter(
            chunk_size= self.chunk_size
        )
        return text_splitter.create_documents(documents, metadatas=metadatas)
    
    def build_vectorstore(self, chunks):
        """ Build a vector store from the text chunks.
        Args:
            chunks (list): List of text chunks to be stored in the vector store.
        Returns:
            vector_db (Chroma): The vector store containing the text chunks.
        """
        docs_ids = list( range( len(chunks) ) )
        docs_ids = [ str(d) for d in docs_ids ]

        vector_db = Chroma.from_documents(
                                chunks,
                                self.embedding_model,
                                persist_directory= self.db_dir,
                                ids=docs_ids
                            )
        return vector_db
    
    def load_or_build_db(self):
            """
            Load existing vector DB if present, else build from PDF.
            """
            if os.path.isdir(self.db_dir) and os.listdir(self.db_dir):
                # load stored DB
                self.vector_db = Chroma(
                    persist_directory=self.db_dir,
                    embedding_function=self.embedding_model
                )
                print("♻️ Loaded existing vector DB from:", self.db_dir)
            else:
                print("📦 Vector DB not found, building a new one...")
                docs, metas = self.load_pdf()
                if docs is None or metas is None:
                    raise ValueError(f"Failed to load PDF: {self.pdf_path}")  # <-- add this
                chunks = self.split_text(docs, metas)
                self.vector_db = self.build_vectorstore(chunks)
                print("🎉 New vector DB built and saved to:", self.db_dir)


    def ask_question(self, question):
        """ Ask a question and get an answer using the vector store and LLM.
        Args:
            question (str): The question to ask.
        """
        qna_template = "\n".join([
            "Answer the next question using the provided context.",
            "If the answer is not contained in the context, say 'NO ANSWER IS AVAILABLE'",
            "### Context:",
            "{context}",
            "",
            "### Question:",
            "{question}",
            "",
            "### Answer:",
        ])

        qna_prompt = PromptTemplate(
            template=qna_template,
            input_variables=['context', 'question'],
            verbose=False
        )
        stuff_chain = create_stuff_documents_chain(self.llm, prompt=qna_prompt)
        similar_docs = self.vector_db.similarity_search(
                                            question,
                                            k=3
                                        )
        result = stuff_chain.invoke(
            {
                "context": similar_docs,
                "question": question
            }
        )
        return result

