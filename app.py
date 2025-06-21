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

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Streamlit UI
st.set_page_config(layout="wide")
st.title("chatbot")

# Sidebar upload (compact)
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("PDF file", type=["pdf"], help="Optional: for document-based QA")

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []  # list of (query, answer)

# Process PDF when uploaded
@st.cache_data(show_spinner=False)
def extract_pdf_text(pdf_bytes):
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            text += f"\n--- Page {page_num} ---\n"
            text += page.get_text()
    return text

@st.cache_data(show_spinner=False)
def split_documents(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
    return splitter.create_documents([text])

@st.cache_resource(show_spinner=False)
def get_vector_store(_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(_docs, embedding=embeddings)

# Hybrid retrieval (dense + sparse)
def hybrid_retriever(query, docs, vectordb, k=5):
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": k})
    sparse_retriever = BM25Retriever.from_documents(
        documents=docs,
        bm25_params={"k1": 1.2, "b": 0.75},
        k=k
    )
    return vector_retriever.get_relevant_documents(query) + sparse_retriever.get_relevant_documents(query)

# Build QA chains
def respond_with_docs(query, docs, vectordb):
    initial = hybrid_retriever(query, docs, vectordb, k=5)
    reranker = FlashrankRerank(client=Ranker(model_name="ms-marco-MiniLM-L-12-v2"),
                                model="ms-marco-MiniLM-L-12-v2", top_n=5)
    reranked = reranker.compress_documents(initial, query)
    docs_text = "\n\n".join(f"[Score: {d.metadata.get('relevance_score')}]: {d.page_content}" for d in reranked)
    prompt = PromptTemplate(
        input_variables=["documents", "query"],
        template=("You are an expert assistant. Use the following excerpts to answer. "
                  "If not in excerpts, use your general knowledge.\n\nDocuments:\n{documents}\n\nQuestion:\n{query}\n\nAnswer:\n")
    )
    chain = LLMChain(llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0), prompt=prompt)
    return chain.run({"documents": docs_text, "query": query})

def respond_general(query):
    prompt = PromptTemplate(input_variables=["query"], template="You are a helpful assistant. Answer: {query}")
    chain = LLMChain(llm=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0), prompt=prompt)
    return chain.run({"query": query})

# If PDF uploaded, process once
docs = vectordb = None
if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        text = extract_pdf_text(uploaded_file.read())
        docs = split_documents(text)
        vectordb = get_vector_store(docs)

# Display chat history
for q, a in st.session_state.history:
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Bot:** {a}")
    st.markdown("---")

# Input
query = st.text_input("Enter your question:")
if st.button("Send") and query:
    if vectordb and docs:
        # simple keyword check: treat as doc QA if 'document' keyword present
        if "document" in query.lower():
            answer = respond_with_docs(query, docs, vectordb)
        else:
            answer = respond_general(query)
    else:
        answer = respond_general(query)
    # Save and display
    st.session_state.history.append((query, answer))
    st.experimental_rerun()
