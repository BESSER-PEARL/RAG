import streamlit as st

from rag.rag import create_db, get_db


def database_config():
    st.header('Create the vectorized database')
    source_path = st.text_input(label='Directory where your documents are located', value='./pdfs')
    db_path = st.text_input(label='Directory of the vector DB', value='./vector_db')
    chunk_size = st.number_input(label='Chunk size', min_value=1, value=1000)
    chunk_overlap = st.number_input(label='Chunk overlap', min_value=0, max_value=chunk_size, value=100)
    if st.button('Create DB'):
        st.session_state['db_path'] = db_path
        with st.spinner('Creating database...'):
            create_db(source_path, db_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    db = get_db(db_path)
    if db:
        st.info(f'The database is created an located in {db_path}')
