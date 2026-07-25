from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def build_vectorstore(transcript_text: str):
    """Build a FAISS vectorstore from the provided transcript text."""
    documents = [Document(page_content=transcript_text)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    return FAISS.from_documents(docs, embeddings)
