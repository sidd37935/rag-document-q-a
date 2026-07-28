"""
Handles loading a PDF (given as raw bytes from a Streamlit upload) and
splitting it into retriever-friendly chunks.
"""

import os
import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_and_split_pdf(file_bytes: bytes, chunk_size: int = 1000, chunk_overlap: int = 200,
                           display_name: str | None = None):
    """
    Take raw PDF bytes, load them with PyPDFLoader, and split into chunks.

    Args:
        file_bytes: raw bytes of the uploaded PDF (e.g. uploaded_file.getbuffer())
        chunk_size: max characters per chunk
        chunk_overlap: overlap between consecutive chunks
        display_name: original filename to stamp into chunk metadata (so citations show
            the real filename instead of a temp file path)

    Returns:
        (splits, num_pages) -> list of split Documents, and the original page count
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No readable text found in this PDF (it may be scanned/image-only).")

        if display_name:
            for d in docs:
                d.metadata["source"] = display_name

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        splits = text_splitter.split_documents(docs)
        return splits, len(docs)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
