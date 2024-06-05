import os
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_db(source_path, db_path, chunk_size=1000, chunk_overlap=100):
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
        embedding=OpenAIEmbeddings(),
        persist_directory=db_path
    )
    return vectordb


def get_db(path) -> Chroma:
    if os.path.isdir(path):
        vectordb = Chroma(persist_directory=path, embedding_function=OpenAIEmbeddings())
        return vectordb
    else:
        return None


def run_retrieval(db_path, k, query) -> list[Document]:
    vectordb: Chroma = get_db(db_path)
    if vectordb is None:
        return None
    retriever: VectorStoreRetriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)


def run_llm(query: str, llm_name: str, docs: list[Document]) -> AIMessage:
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm: ChatOpenAI = ChatOpenAI(model=llm_name)
    prompt = hub.pull("rlm/rag-prompt")  # https://smith.langchain.com/hub/rlm/rag-prompt

    formatted_prompt = prompt.format(context=format_docs(docs), question=query)
    return llm.invoke(formatted_prompt)


