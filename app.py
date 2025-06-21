import os
import re
from dotenv import load_dotenv
import fitz
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers.bm25 import BM25Retriever
from flashrank import Ranker
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Streamlit UI
st.set_page_config(layout="wide")
st.title("chatbot with memory")

# Sidebar: compact PDF upload
with st.sidebar.expander("📄 Upload PDF (optional)", expanded=False):
    uploaded_file = st.file_uploader("PDF file", type=["pdf"])

# Initialize session state for history and memory
if "history" not in st.session_state:
    st.session_state.history = []
if "memory_general" not in st.session_state:
    st.session_state.memory_general = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "memory_docs" not in st.session_state:
    st.session_state.memory_docs = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# PDF processing functions
@st.cache_data(show_spinner=False)
def extract_pdf_text(pdf_bytes):
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for i, page in enumerate(doc, 1):
            text += f"\n--- Page {i} ---\n" + page.get_text()
    return text

@st.cache_data(show_spinner=False)
def split_documents(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    return splitter.create_documents([text])

@st.cache_resource(show_spinner=False)
def get_vector_store(_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(_docs, embedding=embeddings)

# Retrieval functions
def hybrid_retriever(query, docs, vectordb, k=5):
    vec = vectordb.as_retriever(search_kwargs={"k": k})
    sparse = BM25Retriever.from_documents(documents=docs, bm25_params={"k1":1.2, "b":0.75}, k=k)
    return vec.get_relevant_documents(query) + sparse.get_relevant_documents(query)

# QA chains with memory
def respond_with_docs(query, docs, vectordb):
    results = hybrid_retriever(query, docs, vectordb)
    reranker = FlashrankRerank(client=Ranker(model_name="ms-marco-MiniLM-L-12-v2"), model="ms-marco-MiniLM-L-12-v2", top_n=5)
    reranked = reranker.compress_documents(results, query)
    doc_text = "\n\n".join(f"[Score: {d.metadata.get('relevance_score')}]: {d.page_content}" for d in reranked)
    prompt = PromptTemplate(
        input_variables=["documents","query"],
        template=("You are an expert assistant with context memory. Use the following excerpts to answer. "
                  "If not present, use your knowledge. Include conversation history where relevant.\n\n"
                  "Chat History:\n{chat_history}\n\nDocuments:\n{documents}\n\nQuestion:\n{query}\n\nAnswer:\n")
    )
    chain = LLMChain(
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0),
        prompt=prompt,
        memory=st.session_state.memory_docs
    )
    return chain.run({"documents": doc_text, "query": query})


def respond_general(query):
    prompt = PromptTemplate(
        input_variables=["query"],
        template=("You are a helpful assistant with short-term context. Chat History:\n{chat_history}\n\nAnswer the question: {query}")
    )
    chain = LLMChain(
        llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0),
        prompt=prompt,
        memory=st.session_state.memory_general
    )
    return chain.run({"query": query})

# Process PDF
docs = vectordb = None
if uploaded_file:
    with st.spinner("Processing PDF..."):
        text = extract_pdf_text(uploaded_file.read())
        docs = split_documents(text)
        vectordb = get_vector_store(docs)

# Display chat history
for q, a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown("---")

# Chat form
with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input("Enter your question:")
    submit = st.form_submit_button("Send")

if submit and query:
    if docs and vectordb and "document" in query.lower():
        answer = respond_with_docs(query, docs, vectordb)
    else:
        answer = respond_general(query)
    st.session_state.history.append((query, answer))
    st.rerun()
