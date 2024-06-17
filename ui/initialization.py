import streamlit as st

from rag.message import Message

PROVIDERS = ['HuggingFace', 'OpenAI']


def set_defaults():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
        st.session_state['history'].append(Message(content='Hello! I am here to answer your questions', is_user=False))

    if 'k' not in st.session_state:
        st.session_state['k'] = 4

    if 'db_path' not in st.session_state:
        st.session_state['db_path'] = './vector_db'

    if 'num_context' not in st.session_state:
        st.session_state['num_context'] = 2

    if 'provider' not in st.session_state:
        st.session_state['provider'] = 'HuggingFace'

    # OpenAI

    if 'openai_llm_name' not in st.session_state:
        st.session_state['openai_llm_name'] = 'gpt-3.5-turbo'

    if 'openai_api_key' not in st.session_state:
        st.session_state['openai_api_key'] = ''

    # HuggingFace

    if 'hf_token' not in st.session_state:
        st.session_state['hf_token'] = ''

    if 'hf_embeddings_name' not in st.session_state:
        st.session_state['hf_embeddings_name'] = 'all-MiniLM-L6-v2'

    if 'hf_llm_name' not in st.session_state:
        st.session_state['hf_llm_name'] = 'google/gemma-2b-it'

    if 'hf_max_length' not in st.session_state:
        st.session_state['hf_max_length'] = 2000
