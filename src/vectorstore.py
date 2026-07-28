"""
Builds a FAISS vector store from document chunks using a local
HuggingFace sentence-transformer embedding model.
"""

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


@st.cache_resource(show_spinner=False)
def get_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """
    Cached so the embedding model is downloaded/loaded only once per
    Streamlit session instead of on every PDF upload.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def create_vectorstore(splits, model_name: str = "all-MiniLM-L6-v2"):
    """
    Embed the given document chunks and build an in-memory FAISS index.
    """
    embeddings = get_embeddings(model_name)
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore
