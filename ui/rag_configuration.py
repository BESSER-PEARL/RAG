import os

import streamlit as st

from ui.initialization import PROVIDERS


def update_value(key):
    st.session_state[key] = st.session_state[f'_{key}']


def rag_configuration():
    st.header('RAG Configuration')
    st.text_input(label='Database path', value=st.session_state['db_path'], key='_db_path', on_change=update_value, kwargs={'key': 'db_path'})
    st.number_input(label='K', min_value=1, value=st.session_state['k'], key='_k', on_change=update_value, kwargs={'key': 'k'})
    st.number_input(label='Number of previous messages to keep in context', min_value=0, value=st.session_state['num_context'], key='_num_context', on_change=update_value, kwargs={'key': 'num_context'})

    provider = st.radio(label='Embeddings & LLM Provider', options=PROVIDERS, index=PROVIDERS.index(st.session_state['provider']), key='_provider', on_change=update_value, kwargs={'key': 'provider'})

    if provider == 'OpenAI':
        st.text_input(label='OpenAI API key', type='password', value=st.session_state['openai_api_key'], key='_openai_api_key', on_change=update_value, kwargs={'key': 'openai_api_key'})
        os.environ["OPENAI_API_KEY"] = st.session_state['openai_api_key']
        st.text_input(label='OpenAI model', value=st.session_state['openai_llm_name'], key='_openai_llm_name', on_change=update_value, kwargs={'key': 'openai_llm_name'})
    elif provider == 'HuggingFace':
        st.text_input(label='HuggingFace Token', type='password', value=st.session_state['hf_token'], key='_hf_token', on_change=update_value, kwargs={'key': 'hf_token'})
        st.text_input(label='HuggingFace embeddings', value=st.session_state['hf_embeddings_name'], key='_hf_embeddings_name', on_change=update_value, kwargs={'key': 'hf_embeddings_name'})
        st.text_input(label='HuggingFace model', value=st.session_state['hf_llm_name'], key='_hf_llm_name', on_change=update_value, kwargs={'key': 'hf_llm_name'})
        st.number_input(label='Max input length (tokens)', min_value=1, value=st.session_state['hf_max_length'], key='_hf_max_length', on_change=update_value, kwargs={'key': 'hf_max_length'})
