# 🤖 RAG Bot — PDF Question-Answering Assistant

A Retrieval-Augmented Generation (RAG) chatbot built with **Streamlit**, **LangChain**, **FAISS**, and **Groq's LLaMA/Gemma models**. Upload one or more PDFs and ask natural-language questions — the bot retrieves the most relevant passages and answers grounded in your documents, with page-level source citations.

---

## Table of contents
- [What this project does](#what-this-project-does)
- [How it works (architecture)](#how-it-works-architecture)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Setup — run it locally](#setup--run-it-locally)
- [Configuration](#configuration)
- [Usage guide](#usage-guide)
- [Deploying to Streamlit Community Cloud](#deploying-to-streamlit-community-cloud)
- [Security notes](#security-notes)
- [Troubleshooting](#troubleshooting)
- [Roadmap / ideas for later](#roadmap--ideas-for-later)

---

## What this project does

You upload a PDF (or several). The app:
1. Extracts and splits the text into overlapping chunks.
2. Embeds each chunk into a vector using a local sentence-transformer model.
3. Stores the vectors in an in-memory FAISS index.
4. When you ask a question, it retrieves the most relevant chunks and passes them, along with your chat history, to a Groq-hosted LLM (Llama 3.3 70B by default) to generate a grounded answer.
5. Shows you exactly which document/page each answer was pulled from.

This is the classic **RAG (Retrieval-Augmented Generation)** pattern: instead of relying only on what the LLM memorized during training, you give it the specific, relevant context it needs at question time — so answers stay accurate and traceable to your own documents.

## How it works (architecture)

```
                 ┌─────────────┐
   Upload PDF →  │ PyPDFLoader │  (src/document_loader.py)
                 └──────┬──────┘
                        │ raw pages
                 ┌──────▼──────────────┐
                 │ RecursiveCharacter  │  chunk_size / chunk_overlap
                 │ TextSplitter        │  (configurable in sidebar)
                 └──────┬──────────────┘
                        │ chunks
                 ┌──────▼──────────────┐
                 │ HuggingFaceEmbeddings│  all-MiniLM-L6-v2 (local, free)
                 └──────┬──────────────┘
                        │ vectors
                 ┌──────▼──────┐
                 │ FAISS index │  (src/vectorstore.py, cached per session)
                 └──────┬──────┘
                        │ similarity search (top-k)
Question →  ┌───────────▼───────────────────────┐
            │ History-aware retriever            │  rewrites follow-up
            │  + Groq LLM (Llama/Gemma)          │  questions using
            │ (src/rag_chain.py)                 │  chat history
            └───────────┬───────────────────────┘
                        │ retrieved chunks + question + history
                 ┌──────▼──────┐
                 │  QA prompt   │  "answer using ONLY this context"
                 │  + Groq LLM  │
                 └──────┬──────┘
                        │
                 Answer + cited source snippets → Streamlit UI
```

## Features

- 📄 **Multi-PDF upload** — combine several documents into a single searchable index
- 💬 **Conversational memory** — follow-up questions are understood in context (e.g. "what about page 2?")
- 🔎 **Source citations** — every answer links back to the file name, page number, and a text snippet
- 🎛️ **Configurable retrieval** — pick the Groq model, temperature, retrieved-chunks count (`k`), chunk size/overlap, all from the sidebar
- 🎨 **Light / Dark theme toggle** — high-contrast palettes, switchable at runtime
- 📊 **Live stats dashboard** — documents, pages, chunks, and message counters
- ⬇️ **Chat export** — download the full transcript as `.txt`
- 🧹 **Clear chat** / **Reset all** controls
- 🔐 **Safe key handling** — server-side `.env` key is never exposed in the UI; visitors can optionally bring their own key

## Tech stack

| Layer            | Tool                                                        |
|-------------------|--------------------------------------------------------------|
| UI                | [Streamlit](https://streamlit.io)                            |
| Orchestration     | [LangChain](https://python.langchain.com) (`langchain`, `langchain-community`, `langchain-core`, `langchain-text-splitters`) |
| LLM               | [Groq](https://groq.com) — Llama 3.3 70B / Llama 3.1 8B / Gemma 2 9B |
| Embeddings        | `sentence-transformers` (`all-MiniLM-L6-v2`, runs locally, no API cost) |
| Vector store      | [FAISS](https://github.com/facebookresearch/faiss) (in-memory, per session) |
| PDF parsing       | `pypdf` via `PyPDFLoader`                                     |
| Config            | `python-dotenv`                                               |

## Project structure

```
RAG bot/
├── app.py                  # Streamlit UI, session state, orchestration
├── requirements.txt        # Python dependencies
├── .env.example             # Template for required environment variables
├── .env                     # Your real secrets (gitignored, not committed)
├── .gitignore
├── pyrightconfig.json       # Editor config so Pylance resolves src/ correctly
├── .vscode/
│   └── settings.json         # Same fix, VS Code-specific
└── src/
    ├── __init__.py
    ├── document_loader.py   # PDF → text chunks
    ├── vectorstore.py       # chunks → FAISS index (cached embeddings model)
    └── rag_chain.py         # retriever + Groq LLM → history-aware RAG chain
```

## Setup — run it locally

**Prerequisites:** Python 3.10+, a free [Groq API key](https://console.groq.com).

```bash
# 1. Clone / unzip the project, then move into it
cd "RAG bot"

# 2. (recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
cp .env.example .env
# open .env and paste: GROQ_API_KEY=your_key_here

# 5. Run the app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Configuration

| Variable        | Required | Description                              |
|------------------|----------|--------------------------------------------|
| `GROQ_API_KEY`   | Yes      | Your Groq API key, from console.groq.com   |

All other settings (model choice, temperature, chunk size, retrieval `k`) are adjustable live from the sidebar — no code changes or restarts needed.

## Usage guide

1. Open the app — the sidebar shows a green **"key loaded ✅"** badge if `.env` is set up correctly.
2. Upload one or more PDFs under **Documents**.
3. Click **🚀 Process documents**. A progress bar shows reading → embedding → indexing.
4. Once processed, ask questions in the chat box at the bottom.
5. Expand **🔎 source excerpt(s)** under any answer to see exactly which page/file it came from.
6. Use **🧹 Clear chat** to start a new conversation on the same documents, or **♻️ Reset all** to remove the documents too.
7. Use **⬇️ Download chat** in the sidebar to save the conversation as a `.txt` file.

## Deploying to Streamlit Community Cloud

1. Push this project to a GitHub repo (make sure `.env` stays out of the repo — it's already in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick your repo/branch → set the main file to `app.py`.
3. Before the first run, open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```
   Streamlit Cloud exposes this both as `st.secrets["GROQ_API_KEY"]` and as a regular environment variable, so `os.getenv("GROQ_API_KEY")` (what this app uses) works without any code changes.
4. Deploy. You'll get a permanent URL like `https://your-app-name.streamlit.app` — this is what you'd put on a resume.

## Security notes

- Your Groq key lives only in `.env` (local) or the Streamlit Cloud **Secrets** panel (deployed) — never in the code or the UI.
- The sidebar never pre-fills a text box with the real key; it only shows a status badge, so screen-sharing or screenshots of the app don't leak it.
- If you deploy publicly without a server-side key, every visitor is prompted to paste their **own** key — no shared quota risk.

## Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `Import "src.xxx" could not be resolved` in VS Code | Pylance does static analysis and doesn't see the runtime `sys.path.insert(...)` in `app.py` | Already fixed via `.vscode/settings.json` / `pyrightconfig.json`. If it still shows, run **"Python: Restart Language Server"** from the command palette. |
| `A Groq API key is required` error | No key in `.env` and none pasted in the sidebar | Add `GROQ_API_KEY` to `.env` or paste one in the sidebar field |
| `No readable text found in this PDF` | The PDF is scanned/image-only with no extractable text layer | Run it through OCR first, or use a text-based PDF |
| App is slow on first upload | The embedding model (`all-MiniLM-L6-v2`) downloads on first use | This only happens once per environment; subsequent runs are cached |
| Colors look washed out | Old cached CSS from a previous run | Hard refresh the browser tab (Ctrl/Cmd+Shift+R) |

## Roadmap / ideas for later

- Streaming (token-by-token) responses
- Persistent vector store (save/reload FAISS index instead of re-embedding every session)
- In-PDF citation viewer (click a source → jump to the highlighted page)
- Support for `.docx`, `.txt`, and web URLs as sources
- 👍/👎 feedback logging per answer
- Usage analytics (most-asked questions, response time, token cost)
