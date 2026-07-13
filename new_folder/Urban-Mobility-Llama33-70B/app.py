from __future__ import annotations

import streamlit as st

from rag import UrbanMobilityRAG
from utils import configure_logging, ensure_directories

configure_logging()
ensure_directories()

st.set_page_config(page_title="Urban Mobility AI - LLaMA-3.3-70B-Versatile", layout="wide")
st.title("Urban Mobility AI - LLaMA-3.3-70B-Versatile")
st.caption("Explainable RAG and tool-augmented NYC taxi intelligence.")

if "rag" not in st.session_state:
    st.session_state.rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Pipeline")
    st.write("Intent detection -> Retriever -> FAISS -> LLM -> Tools -> Memory")
    if st.button("Load model and index"):
        st.session_state.rag = UrbanMobilityRAG()
        st.success("Pipeline loaded.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about NYC taxi distance, demand, patterns, or comparisons")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    if st.session_state.rag is None:
        st.session_state.rag = UrbanMobilityRAG()
    answer = st.session_state.rag.answer(question)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
