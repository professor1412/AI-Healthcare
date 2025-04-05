import os
from typing import List
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import Chroma
from langchain_core.documents import Document as LangchainDocument
import sys


load_dotenv()
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")

def load_and_chunk_pdfs(directory_path: str) -> List[Document]:
    """
    Loads all PDF files from the given directory, extracts text, and chunks them.

    Args:
        directory_path (str): Path to the folder containing medical PDF files.

    Returns:
        List[Document]: List of chunked Document objects.
    """
    medical_docs = []

    # Process each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            reader = PdfReader(file_path)
            full_text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text
            
            # Create a document for the entire PDF
            medical_docs.append(Document(page_content=full_text, metadata={"source": filename}))
    
    # Split the documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)  # Reduced chunk size to control token count
    return splitter.split_documents(medical_docs)

def create_healthcare_retriever(documents: List[Document]):
    """
    Creates a vector retriever from given documents using HuggingFace embeddings with ChromaDB.

    Args:
        documents (list): List of LangChain Document objects.

    Returns:
        retriever: A retriever that can be used to query healthcare knowledge,
                   with a limit on the number of returned chunks.
    """
    # Initialize the embeddings
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_token,
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )

    # Use Chroma instead of InMemory store for persistent, scalable vector storage
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)

    # Limit chunks and context window in the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # Limit to top 3 matches

    return retriever
