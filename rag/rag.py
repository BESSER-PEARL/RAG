import os

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import login

from rag.message import Message


def load_embeddings() -> Embeddings:
    provider = st.session_state['provider']
    if provider == 'OpenAI':
        return OpenAIEmbeddings()
    elif provider == 'HuggingFace':
        embeddings_name = st.session_state['hf_embeddings_name']
        return HuggingFaceEmbeddings(model_name=embeddings_name)


def create_db(source_path: str, db_path: str, embeddings: Embeddings, chunk_size: int = 1000, chunk_overlap: int = 100):
    documents = []
    for file in os.listdir(source_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(source_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    # TODO: RecursiveCharacterTextSplitter vs CharacterTextSplitter
    chunked_documents = text_splitter.split_documents(documents)
    vectordb: Chroma = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory=db_path
    )
    return vectordb


def get_db(path: str, embeddings: Embeddings) -> Chroma:
    if os.path.isdir(path):
        vectordb = Chroma(persist_directory=path, embedding_function=embeddings)
        return vectordb
    else:
        return None


def run_retrieval(db_path: str, embeddings: Embeddings, k: int, query: str) -> list[Document]:
    vectordb: Chroma = get_db(db_path, embeddings)
    if vectordb is None:
        return None
    retriever: VectorStoreRetriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def load_llm(provider: str) -> BaseChatModel:
    if provider == 'OpenAI':
        llm_name = st.session_state['openai_llm_name']
        llm: ChatOpenAI = ChatOpenAI(model=llm_name)
    elif provider == 'HuggingFace':
        llm_name = st.session_state['hf_llm_name']
        max_length = st.session_state['hf_max_length']
        llm: ChatHuggingFace = load_hf_llm(model_id=llm_name, max_length=max_length)
    return llm


@st.cache_resource(show_spinner=f'Loading HuggingFace model...')
def load_hf_llm(model_id: str, max_length=2000) -> ChatHuggingFace:
    login(st.session_state['hf_token'])
    hf = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        pipeline_kwargs={
            "max_length": max_length,
            "return_full_text": False,
            # "max_new_tokens": 10
        },
    )
    llm: ChatHuggingFace = ChatHuggingFace(llm=hf)
    return llm


def create_prompt(history: list[Message], docs: list[Document], question: str, num_context: int) -> PromptValue:
    def format_docs(_docs):
        return "\n\n".join(doc.page_content for doc in _docs)

    conversation = [("human", "You are an assistant for question-answering tasks. You only talk Spanish, even if you receive an English query. Based on the previous messages in the conversation (if provided), and additional context retrieved from a database (if provided), answer the user question. If you don't know the answer, just say that you don't know. Note that if the question refers to a previous message, you may have to ignore the context since it is retrieved from the database based only on the question (the retrieval does not take into account the previous messages). Use three sentences maximum and keep the answer concise")]
    conversation.extend(("human" if message.is_user else "ai", message.content) for message in history[-num_context:])
    conversation.append(("human", "Context: {context}\nQuestion: {question}\nAnswer:"))
    template = ChatPromptTemplate.from_messages(conversation)
    prompt_value = template.invoke(
        {
            "context": format_docs(docs),
            "question": question
        }
    )
    return prompt_value
