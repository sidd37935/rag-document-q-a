"""
Builds a history-aware RAG (retrieval-augmented generation) chain
backed by Groq's hosted LLMs.
"""

from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_rag_chain(vectorstore, groq_api_key: str, model_name: str = "llama-3.3-70b-versatile",
                     temperature: float = 0.2, k: int = 4):
    """
    Args:
        vectorstore: a FAISS (or other) vector store with .as_retriever()
        groq_api_key: Groq API key (required)
        model_name: Groq-hosted model id
        temperature: sampling temperature
        k: number of chunks to retrieve per query
    """
    if not groq_api_key:
        raise ValueError("A Groq API key is required to build the RAG chain.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_name,
        temperature=temperature,
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given the chat history and the latest user question, formulate a standalone "
                   "question which can be understood without the chat history. Do NOT answer the "
                   "question, just reformulate it if needed, otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an intelligent document assistant. Answer the user's question using ONLY the "
         "context below. Be concise and accurate. If the answer isn't in the context, say you "
         "don't know rather than guessing.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
