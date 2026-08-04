import os
import sys
from datetime import datetime

# Ensure Python finds the 'src' directory relative to app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from src.document_loader import process_and_split_pdf
from src.vectorstore import create_vectorstore
from src.rag_chain import build_rag_chain

# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="RAG Bot | Document Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_OPTIONS = {
    "Llama 3.3 70B (best quality)": "llama-3.3-70b-versatile",
    "Llama 3.1 8B (fastest)": "llama-3.1-8b-instant",
    "Gemma 2 9B": "gemma2-9b-it",
}

# --------------------------------------------------------------------------------------
# Theme (light / dark, high-contrast)
# --------------------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"

THEMES = {
    "Dark": {
        "bg": "#0f1117", "bg2": "#14161f", "sidebar_bg": "#12141c",
        "text": "#f3f4f6", "subtext": "#a3a9b7", "border": "rgba(255,255,255,0.10)",
        "card_bg": "rgba(255,255,255,0.05)", "accent": "#a78bfa",
        "chip_bg": "rgba(167,139,250,0.18)", "chip_text": "#ddd6fe", "chip_border": "rgba(167,139,250,0.4)",
    },
    "Light": {
        "bg": "#f7f7fb", "bg2": "#ffffff", "sidebar_bg": "#ffffff",
        "text": "#1a1a2e", "subtext": "#52525b", "border": "rgba(0,0,0,0.10)",
        "card_bg": "#ffffff", "accent": "#6d28d9",
        "chip_bg": "rgba(109,40,217,0.10)", "chip_text": "#5b21b6", "chip_border": "rgba(109,40,217,0.3)",
    },
}
T = THEMES[st.session_state.theme]

st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(180deg, {T['bg']} 0%, {T['bg2']} 100%);
        color: {T['text']};
    }}

    /* Force readable text everywhere, regardless of Streamlit's own theme */
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] span,
    h1, h2, h3, h4, h5, h6, label, .stCaption, [data-testid="stCaptionContainer"] {{
        color: {T['text']} !important;
    }}
    [data-testid="stCaptionContainer"], .stCaption, small {{
        color: {T['subtext']} !important;
    }}

    .hero {{
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        margin-bottom: 1.4rem;
        box-shadow: 0 8px 30px rgba(99,102,241,0.25);
    }}
    .hero h1 {{ color: white !important; margin: 0; font-size: 2rem; }}
    .hero p {{ color: rgba(255,255,255,0.92) !important; margin: 0.3rem 0 0 0; font-size: 1rem; }}

    .metric-card {{
        background: {T['card_bg']};
        border: 1px solid {T['border']};
        border-radius: 14px;
        padding: 0.9rem 1rem;
        text-align: center;
    }}
    .metric-card .val {{ font-size: 1.6rem; font-weight: 700; color: {T['accent']} !important; }}
    .metric-card .lbl {{ font-size: 0.8rem; color: {T['subtext']} !important; text-transform: uppercase; letter-spacing: 0.04em; }}

    section[data-testid="stSidebar"] {{
        background: {T['sidebar_bg']};
        border-right: 1px solid {T['border']};
    }}
    section[data-testid="stSidebar"] * {{ color: {T['text']} !important; }}
    section[data-testid="stSidebar"] [data-baseweb="select"] * ,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * ,
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] * ,
    section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {{
        color: {T['text']} !important;
        opacity: 1 !important;
    }}

    .source-chip {{
        display: inline-block;
        background: {T['chip_bg']};
        color: {T['chip_text']} !important;
        border: 1px solid {T['chip_border']};
        border-radius: 999px;
        padding: 0.15rem 0.7rem;
        font-size: 0.75rem;
        margin: 0.15rem;
    }}

    div[data-testid="stChatMessage"] {{
        border-radius: 14px;
        padding: 0.4rem 0.2rem;
        background: {T['card_bg']};
    }}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------------------
defaults = {
    "chat_history": [],       # [{"role": "user"/"assistant", "content": str, "sources": [...]}]
    "rag_chain": None,
    "doc_stats": None,        # {"files": int, "pages": int, "chunks": int, "names": [...]}
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# --------------------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------------------
with st.sidebar:
    theme_choice = st.radio("🎨 Theme", ["Dark", "Light"],
                             index=0 if st.session_state.theme == "Dark" else 1, horizontal=True)
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()

    st.header("⚙️ Setup")

    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        st.success("Groq API key loaded from server (.env). ✅")
        with st.expander("Use a different key instead"):
            override_key = st.text_input("Your own Groq API key", type="password",
                                          help="Overrides the server key for this session only.")
        groq_api_key = override_key or env_key
    else:
        groq_api_key = st.text_input(
            "Groq API key",
            type="password",
            help="Not set on the server. Paste your own — get a free key at console.groq.com. "
                 "It's only kept in your browser session, never written to disk.",
        )

    model_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
    model_name = MODEL_OPTIONS[model_label]

    with st.expander("🔧 Advanced settings"):
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
        top_k = st.slider("Chunks retrieved per question (k)", 2, 10, 4, 1)
        chunk_size = st.slider("Chunk size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk overlap", 0, 500, 200, 50)

    st.divider()
    st.header("📄 Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_clicked = st.button("🚀 Process documents", type="primary", use_container_width=True)

    if process_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        elif not groq_api_key:
            st.error("A Groq API key is required. Add it above or set GROQ_API_KEY in .env.")
        else:
            progress = st.progress(0, text="Starting...")
            try:
                all_splits = []
                total_pages = 0
                names = []
                n = len(uploaded_files)
                for i, f in enumerate(uploaded_files):
                    progress.progress((i) / n, text=f"Reading {f.name}...")
                    splits, num_pages = process_and_split_pdf(
                        f.getvalue(), chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                        display_name=f.name,
                    )
                    all_splits.extend(splits)
                    total_pages += num_pages
                    names.append(f.name)

                progress.progress(0.7, text="Building vector index...")
                vectorstore = create_vectorstore(all_splits)

                progress.progress(0.9, text="Initializing chat model...")
                st.session_state.rag_chain = build_rag_chain(
                    vectorstore, groq_api_key, model_name=model_name,
                    temperature=temperature, k=top_k,
                )
                st.session_state.doc_stats = {
                    "files": len(uploaded_files),
                    "pages": total_pages,
                    "chunks": len(all_splits),
                    "names": names,
                }
                st.session_state.chat_history = []
                progress.progress(1.0, text="Done!")
                st.success(f"Processed {len(uploaded_files)} file(s) successfully!")
            except Exception as e:
                st.error(f"Error processing documents: {e}")
            finally:
                progress.empty()

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col_b:
        if st.button("♻️ Reset all", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.rag_chain = None
            st.session_state.doc_stats = None
            st.rerun()

    if st.session_state.chat_history:
        transcript = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history
        )
        st.download_button(
            "⬇️ Download chat",
            data=transcript,
            file_name=f"rag_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            use_container_width=True,
        )

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🤖 RAG Question-Answering Bot</h1>
    <p>Upload PDFs, then ask questions and get grounded answers with cited sources.</p>
</div>
""", unsafe_allow_html=True)

stats = st.session_state.doc_stats
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, stats["files"] if stats else 0, "Documents"),
    (c2, stats["pages"] if stats else 0, "Pages"),
    (c3, stats["chunks"] if stats else 0, "Chunks indexed"),
    (c4, len(st.session_state.chat_history), "Messages"),
]:
    with col:
        st.markdown(f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                     unsafe_allow_html=True)

st.write("")

# --------------------------------------------------------------------------------------
# Main chat
# --------------------------------------------------------------------------------------
if st.session_state.rag_chain is None:
    st.info("👈 Upload one or more PDFs in the sidebar and click **Process documents** to start chatting.")
else:
    if stats:
        with st.expander(f"📚 Indexed documents ({stats['files']})"):
            for name in stats["names"]:
                st.markdown(f'<span class="source-chip">{name}</span>', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"🔎 {len(msg['sources'])} source excerpt(s)"):
                    for s in msg["sources"]:
                        st.markdown(f'<span class="source-chip">{s["label"]}</span>', unsafe_allow_html=True)
                        st.caption(s["snippet"])

    if prompt := st.chat_input("Ask a question about your document(s)..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build proper LangChain message objects from prior turns (before this new question)
        lc_history = []
        for m in st.session_state.chat_history[:-1]:
            if m["role"] == "user":
                lc_history.append(HumanMessage(content=m["content"]))
            else:
                lc_history.append(AIMessage(content=m["content"]))

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke({
                        "input": prompt,
                        "chat_history": lc_history,
                    })
                    answer = response.get("answer", response) if isinstance(response, dict) else response

                    sources = []
                    for doc in response.get("context", []) if isinstance(response, dict) else []:
                        page = doc.metadata.get("page", "?")
                        source_name = doc.metadata.get("source", "document")
                        source_name = os.path.basename(str(source_name))
                        snippet = doc.page_content[:280] + ("..." if len(doc.page_content) > 280 else "")
                        sources.append({"label": f"{source_name} · page {page}", "snippet": snippet})

                    st.markdown(answer)
                    if sources:
                        with st.expander(f"🔎 {len(sources)} source excerpt(s)"):
                            for s in sources:
                                st.markdown(f'<span class="source-chip">{s["label"]}</span>', unsafe_allow_html=True)
                                st.caption(s["snippet"])

                    st.session_state.chat_history.append({
                        "role": "assistant", "content": answer, "sources": sources,
                    })
                except Exception as e:
                    st.error(f"Error generating response: {e}")
