import os
import re
from dotenv import load_dotenv
import fitz
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
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
st.title("PDF QA App with Hybrid Retrieval and Reranking")

# Upload PDF
uploaded_file = st.file_uploader("Drag and drop a PDF file here", type=["pdf"])

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
    # vectordb = Chroma.from_documents(documents=_docs, embedding=embeddings)
    vectordb = FAISS.from_documents(_docs, embedding=embeddings)
    return vectordb

# hybrid retrieval(dense+sparse)
def hybrid_retriever(query, docs, vectordb, k=5):
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": k})
    sparse_retriever = BM25Retriever.from_documents(
        documents=docs,
        bm25_params={"k1": 1.2, "b": 0.75},
        k=k
    )
    vector_results = vector_retriever.get_relevant_documents(query)
    sparse_results = sparse_retriever.get_relevant_documents(query)
    return vector_results + sparse_results

if uploaded_file is not None:
    # Process PDF
    pdf_bytes = uploaded_file.read()
    with st.spinner("Extracting text..."):
        full_text = extract_pdf_text(pdf_bytes)
    with st.spinner("Splitting into chunks..."):
        docs = split_documents(full_text)
    with st.spinner("Building vector store..."):
        vectordb = get_vector_store(docs)

    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("Retrieving relevant documents..."):
            initial_docs = hybrid_retriever(query, docs, vectordb, k=5)

        # Rerank
        flashrank_client = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
        reranker = FlashrankRerank(client=flashrank_client, model="ms-marco-MiniLM-L-12-v2", top_n=5)
        reranked = reranker.compress_documents(initial_docs, query)

        # Prepare prompt
        template = """
You are an expert assistant. Use the following retrieved document excerpts to answer the user's question.
If the answer isn't contained in the excerpts, say "I don't know."

Documents:
{documents}

Question:
{query}

Answer:
"""
        prompt = PromptTemplate(input_variables=["documents", "query"], template=template)
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
        chain = LLMChain(llm=llm, prompt=prompt)

        # Format documents for chain
        docs_text = "\n\n".join(
            f"[Score: {d.metadata.get('relevance_score', 'N/A')}] {d.page_content}"
            for d in reranked
        )
        with st.spinner("Generating answer..."):
            answer = chain.run({"documents": docs_text, "query": query})

        st.subheader("Answer")
        st.write(answer)

        if st.checkbox("Show retrieved passages"):
            st.subheader("Top Passages")
            for i, d in enumerate(reranked, start=1):
                st.markdown(f"**Passage {i} (Score: {d.metadata.get('relevance_score', 'N/A')})**")
                st.write(d.page_content)
                st.markdown("---")
