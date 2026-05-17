# rag_app/rag.py

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

# =========================================================
# LOAD ENV VARIABLES
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================================================
# LOAD WEBSITE DATA
# =========================================================

def load_website(url):
    """
    Load website content from URL
    """

    loader = WebBaseLoader(url)

    documents = loader.load()

    return documents


# =========================================================
# SPLIT DOCUMENTS
# =========================================================

def split_documents(documents):
    """
    Split large text into smaller chunks
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    return chunks


# =========================================================
# CREATE VECTOR STORE
# =========================================================

def create_vector_store(chunks):
    """
    Create FAISS vector database
    """

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    return vector_store


# =========================================================
# CREATE LLM
# =========================================================

def create_llm():
    """
    Create Gemini LLM
    """

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )

    return llm


# =========================================================
# BUILD RAG CHAIN
# =========================================================

def build_rag_chain(vector_store):
    """
    Build RetrievalQA chain
    """

    llm = create_llm()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    return qa_chain


# =========================================================
# MAIN RAG FUNCTION
# =========================================================

def ask_website(url, question):
    """
    Full RAG pipeline:
    1. Load website
    2. Split text
    3. Create embeddings
    4. Store vectors
    5. Ask question
    """

    # Load website
    documents = load_website(url)

    # Split documents
    chunks = split_documents(documents)

    # Create vector DB
    vector_store = create_vector_store(chunks)

    # Build QA chain
    qa_chain = build_rag_chain(vector_store)

    # Ask question
    result = qa_chain.invoke({
        "query": question
    })

    return {
        "answer": result["result"],
        "sources": result["source_documents"]
    }