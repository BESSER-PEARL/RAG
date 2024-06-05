import streamlit as st

from rag.message import Message


def set_defaults():
    if 'history' not in st.session_state:
        st.session_state['history'] = []
        st.session_state['history'].append(Message(content='Hello! I am here to answer your questions', is_user=False))

    if 'k' not in st.session_state:
        st.session_state['k'] = 4

    if 'llm_name' not in st.session_state:
        st.session_state['llm_name'] = 'gpt-3.5-turbo'

    if 'db_path' not in st.session_state:
        st.session_state['db_path'] = './vector_db'

    if 'num_context' not in st.session_state:
        st.session_state['num_context'] = 2

    if 'openai_api_key' not in st.session_state:
        st.session_state['openai_api_key'] = ''
