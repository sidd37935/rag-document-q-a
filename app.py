"""
=============================================================
 Dynamic RAG-Based Document Q&A System
 Tech Stack: Streamlit + LangChain + Groq (Llama-3.3) + Google Embeddings + FAISS
 
 Kya karta hai ye project?
 - User koi bhi PDF upload karta hai
 - System us PDF ko chhote chhote chunks mein tod deta hai
 - Har chunk ka embedding (numeric representation) banta hai
 - Jab user question karta hai, relevant chunks dhundhke Groq LLM ko deta hai
 - LLM un chunks ke basis pe accurate answer deta hai
=============================================================
"""

import os
import streamlit as st
from dotenv import load_dotenv

# PDF padhne ke liye
from langchain_community.document_loaders import PyPDFLoader

# Document ko chhote pieces mein todne ke liye
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector database - embeddings store karne ke liye (local, fast)
from langchain_community.vectorstores import FAISS

# Groq ka LLM (Llama-3.3) - actual jawab dene ke liye
from langchain_groq import ChatGroq

# RAG chain banane ke liye - retrieval + answer generation
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Prompt template - LLM ko instructions dene ke liye
from langchain_core.prompts import ChatPromptTemplate

# Google ka Embedding model - text ko numbers mein convert karta hai
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ─────────────────────────────────────────────
# STEP 1: Environment Variables Load Karo
# .env file se API keys uthata hai
# ─────────────────────────────────────────────
load_dotenv()

# ─────────────────────────────────────────────
# STEP 2: Streamlit Web Page Setup
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A System",
    page_icon="📄",
    layout="wide"
)
st.title("📄 Dynamic RAG-Based Document Q&A System")
st.subheader("Upload any PDF and ask questions directly to Groq Llama-3.3")

# Agar koi API key missing hai to app band kar do
if not os.getenv("GOOGLE_API_KEY") or not os.getenv("GROQ_API_KEY"):
    st.error("❌ Configuration Error: GOOGLE_API_KEY or GROQ_API_KEY is missing in the .env file.")
    st.stop()

# ─────────────────────────────────────────────
# STEP 3: PDF Processing Function
# Ye function poora RAG pipeline setup karta hai:
# PDF Load → Chunks → Embeddings → FAISS Vector Store → Retriever
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def process_uploaded_pdf(uploaded_file):
    """
    PDF ko process karke ek 'retriever' banata hai.
    Retriever ka kaam hai: question ke basis pe relevant chunks dhundhna.
    
    @st.cache_resource: Ek baar process hone ke baad dobara process nahi karta
    """
    
    # PDF ko temporarily disk pe save karo (PyPDFLoader ko file path chahiye)
    temp_filename = f"temp_{uploaded_file.name}"
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # ── 3a. PDF Load Karo ──
        loader = PyPDFLoader(temp_filename)
        documents = loader.load()
        # Ab 'documents' mein puri PDF ka text hai, page by page
        
        # ── 3b. Chunks Banao ──
        # Poori PDF ko 1000 character ke pieces mein todo
        # overlap=200 matlab: har chunk ke end ke 200 characters
        # agle chunk mein bhi rahenge (context loss na ho)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        final_chunks = text_splitter.split_documents(documents)
        
        # ── 3c. Embeddings Setup Karo ──
        # Google ka embedding model text ko numbers (vectors) mein badalta hai
        # Ye numbers similar content ko similar numbers deta hai
        # models/gemini-embedding-001: available model tera Google account mein
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # ── 3d. FAISS Vector Store Banao ──
        # FAISS: Facebook ka fast similarity search library
        # Har chunk ka embedding calculate karke store karta hai
        vector_store = FAISS.from_documents(final_chunks, embeddings)
        
        # ── 3e. Retriever Banao ──
        # k=3 matlab: question ke liye top 3 relevant chunks dhundhega
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        return retriever
    
    except Exception as e:
        st.error(f"Error while building vector index: {e}")
        return None
    
    finally:
        # Kaam ho gaya, temporary file delete karo
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# ─────────────────────────────────────────────
# STEP 4: Sidebar - PDF Upload Section
# ─────────────────────────────────────────────
st.sidebar.header("📁 Document Management")
uploaded_file = st.sidebar.file_uploader(
    "Upload your PDF file:",
    type=["pdf"]
)

# ─────────────────────────────────────────────
# STEP 5: Main App Logic - PDF Upload Hone Par
# ─────────────────────────────────────────────
if uploaded_file is not None:
    
    # PDF process karo aur retriever banao
    with st.spinner("Processing PDF and building vector index..."):
        retriever = process_uploaded_pdf(uploaded_file)
    
    if retriever is not None:
        st.sidebar.success(f"✅ '{uploaded_file.name}' Indexed Successfully!")

        # ── 5a. LLM Setup (Groq - Llama 3.3) ──
        # Ye actual question ka jawab dega
        # temperature=0.3: thoda creative but mostly factual answers
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )

        # ── 5b. System Prompt ──
        # LLM ko batao ki wo kaise behave kare
        # {context} mein retriever ke dhundhe hue chunks aayenge
        system_prompt = (
            "You are an advanced AI assistant. Analyze the provided pieces of retrieved context "
            "to deliver a comprehensive and accurate answer to the user's query. If the context "
            "does not contain enough information, explicitly state that you do not know.\n\n"
            "Context:\n{context}"
        )

        # Prompt template: system instructions + user ka question
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),  # {input} mein user ka question aayega
        ])

        # ── 5c. RAG Chain Build Karo ──
        # Chain 1: Retrieved chunks + prompt ko LLM ko deta hai
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        # Chain 2: Pehle retriever se chunks dhundhta hai, phir Chain 1 chalata hai
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # ── 5d. User Input ──
        st.write("---")
        user_query = st.text_input(
            "💬 Ask a question based on your uploaded document:",
            placeholder="e.g., Summary of the document"
        )

        # ── 5e. Answer Generate Karo ──
        if user_query:
            with st.spinner("Searching database..."):
                try:
                    # RAG chain invoke karo
                    # Flow: user_query → retriever → relevant chunks → LLM → answer
                    response = rag_chain.invoke({"input": user_query})
                    
                    # Answer display karo
                    st.markdown("### 📝 Response:")
                    st.write(response["answer"])
                    
                    # Retrieved chunks bhi dikhao (optional, debugging ke liye useful)
                    with st.expander("📚 View Retrieved Document Context Blocks"):
                        for i, doc in enumerate(response.get("context", [])):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.caption(doc.page_content)
                            st.write("---")
                
                except Exception as e:
                    st.error(f"Execution Error: {e}")

else:
    # Jab tak PDF upload na ho, ye message dikhao
    st.info("💡 Please upload a PDF file from the sidebar to activate the RAG Q&A session.")
