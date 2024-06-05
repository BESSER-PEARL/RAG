import os

import streamlit as st


def update_value(key):
    st.session_state[key] = st.session_state[f'_{key}']


def rag_configuration():
    st.header('RAG Configuration')
    st.text_input(label='Database path', value=st.session_state['db_path'], key='_db_path', on_change=update_value, kwargs={'key': 'db_path'})
    st.text_input(label='OpenAI API key', type='password', value=st.session_state['openai_api_key'], key='_openai_api_key', on_change=update_value, kwargs={'key': 'openai_api_key'})
    os.environ["OPENAI_API_KEY"] = st.session_state['openai_api_key']
    st.text_input(label='OpenAI model', value=st.session_state['llm_name'], key='_llm_name', on_change=update_value, kwargs={'key': 'llm_name'})
    st.number_input(label='K', min_value=1, value=st.session_state['k'], key='_k', on_change=update_value, kwargs={'key': 'k'})
    st.number_input(label='Number of previous messages to keep in context', min_value=0, value=st.session_state['num_context'], key='_num_context', on_change=update_value, kwargs={'key': 'num_context'})
