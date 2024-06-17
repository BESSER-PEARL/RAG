import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from rag.message import Message
from rag.rag import run_retrieval, run_llm, load_llm, load_embeddings

user_type = {
    0: 'assistant',
    1: 'user'
}


def chat():
    st.header('RAG Application')

    # User input component. Must be declared before history writing
    user_input = st.chat_input("Ask something")

    for message in st.session_state['history']:
        write_message(message)

    if user_input:
        user_message = Message(content=user_input, is_user=True)
        write_message(user_message)

        with st.spinner('Thinking...'):
            k = st.session_state['k']
            db_path = st.session_state['db_path']
            provider = st.session_state['provider']
            llm = load_llm(provider)
            embeddings = load_embeddings()
            docs: list[Document] = run_retrieval(db_path, embeddings, k, user_input)
            if docs is None:
                st.error('There is no database')
            else:
                llm_response: AIMessage = run_llm(user_input, llm, docs)
                llm_message = Message(content=llm_response.content, is_user=False, docs=docs)
                st.session_state.history.append(user_message)
                st.session_state.history.append(llm_message)
                write_message(llm_message)


def write_message(message: Message):
    with st.chat_message(user_type[message.is_user]):
        st.write(message.content)
        if not message.is_user and message.docs:
            with st.expander('Details'):
                for i, doc in enumerate(message.docs):
                    st.write(f'**Document {i+1}/{len(message.docs)}**')
                    st.write(f'- **Source:** {doc.metadata["source"]}')
                    st.write(f'- **Page:** {doc.metadata["page"]}')
                    st.write(f'- **Content:** {doc.page_content}')
