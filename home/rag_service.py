import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

load_dotenv(ENV_PATH)
os.environ.setdefault("USER_AGENT", os.getenv("USER_AGENT", "WebRAG/1.0"))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    base_url: str
    chat_model: str
    embedding_model: str


def _load_openrouter_config():
    load_dotenv(ENV_PATH, override=False)

    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    base_url = (os.getenv("OPENROUTER_BASE_URL") or "").strip()
    chat_model = (os.getenv("OPENROUTER_MODEL") or os.getenv("MODEL_NAME") or "").strip()
    embedding_model = (os.getenv("OPENROUTER_EMBEDDING_MODEL") or "").strip()

    missing_vars = []
    if not api_key:
        missing_vars.append("OPENROUTER_API_KEY")
    if not base_url:
        missing_vars.append("OPENROUTER_BASE_URL")
    if not chat_model:
        missing_vars.append("OPENROUTER_MODEL")
    if not embedding_model:
        missing_vars.append("OPENROUTER_EMBEDDING_MODEL")

    if missing_vars:
        message = (
            "Missing OpenRouter configuration in webinfo/.env: "
            + ", ".join(missing_vars)
            + "."
        )
        if "OPENROUTER_MODEL" in missing_vars:
            message += " You can temporarily set MODEL_NAME as a fallback for OPENROUTER_MODEL."
        raise ValueError(message)

    return OpenRouterConfig(
        api_key=api_key,
        base_url=base_url,
        chat_model=chat_model,
        embedding_model=embedding_model,
    )


def _create_embeddings(config):
    return OpenAIEmbeddings(
        model=config.embedding_model,
        api_key=config.api_key,
        base_url=config.base_url,
    )


def _create_llm(config):
    return ChatOpenAI(
        model=config.chat_model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=0,
    )


def _create_retriever(documents, embeddings):
    vectorstore = InMemoryVectorStore(embeddings)
    vectorstore.add_documents(documents)
    return vectorstore.as_retriever(search_kwargs={"k": 4})

# step 1: get url and load data
def get_rag_answer(url, query):
    """
    Load the website, process it, and answer the query.
    """
    config = _load_openrouter_config()

    print(f"Loading documents from {url}...")

    loader = WebBaseLoader([url])
    docs = loader.load()

    # step 2: split documents into chunks
    print("Splitting documents...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    splits = text_splitter.split_documents(docs)
    # step 3: create embeddings and vector store
    print("Creating embeddings...")

    embeddings = _create_embeddings(config)
    # step 4: create retriever and RAG chain
    print("Creating retriever...")

    retriever = _create_retriever(splits, embeddings)

    print("Creating LLM...")

    # step 5: create LLM and RAG chain
    llm = _create_llm(config)

    #step 6: building RAG chain and invoking query
    print("Building RAG chain...")

    template = """Answer the question based only on the following 
                    context: {context}
                    Question: {question}"""
    
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("Invoking query...")

    response = chain.invoke(query)

    return response
