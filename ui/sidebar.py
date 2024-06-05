import streamlit as st
import streamlit_antd_components as sac


def sidebar_menu():
    st.header('RAG')
    page = sac.menu([
        sac.MenuItem('Chat', icon='chat-left-dots'),
        sac.MenuItem('Database', icon='database'),
        sac.MenuItem('RAG Configuration', icon='gear'),
    ], open_all=True)
    return page
