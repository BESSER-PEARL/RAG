import sys

import streamlit as st
from streamlit.web import cli as stcli

from ui.database import database_config
from ui.chat import chat
from ui.initialization import set_defaults
from ui.rag_configuration import rag_configuration
from ui.sidebar import sidebar_menu

# https://python.langchain.com/v0.2/docs/tutorials/rag/
# https://diptimanrc.medium.com/rapid-q-a-on-multiple-pdfs-using-langchain-and-chromadb-as-local-disk-vector-store-60678328c0df

st.set_page_config(layout="wide")


if __name__ == "__main__":
    if st.runtime.exists():
        set_defaults()
        with st.sidebar:
            page = sidebar_menu()
        if page == 'Chat':
            chat()
        elif page == 'Database':
            database_config()
        elif page == 'RAG Configuration':
            rag_configuration()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
