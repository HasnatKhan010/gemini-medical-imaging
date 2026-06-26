# pyright: reportMissingImports=false

import os
import time

import streamlit as st

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key


st.set_page_config(page_title="RAG Assistant", page_icon="📚", layout="centered")

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            text-align: center;
            padding: 1rem 0 1.25rem;
        }
        .hero h1 {
            margin-bottom: 0.25rem;
            color: #111827;
        }
        .hero p {
            color: #4b5563;
            margin: 0;
        }
        .card {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(148, 163, 184, 0.3);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin: 0.75rem 0;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        }
    </style>
    <div class="hero">
        <h1>RAG Document Assistant</h1>
        <p>Ask questions from your PDF documents and get grounded answers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if not groq_api_key:
    st.error("GROQ_API_KEY not found. Add it to your environment or .env file.")
    st.stop()


llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
Please answer the question strictly based on the provided context.
Ensure the response is accurate, concise, and directly addresses the question.

<context>
{context}
</context>

<question>
{input}
</question>
"""
)




def vector_embedding(docs=None) -> None:
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Load from uploaded files or from directory
        if docs is None:
            loader = PyPDFDirectoryLoader("./ed_pdf")
            docs = loader.load()
        
        if not docs:
            st.error("No PDF documents found. Please upload PDFs or add them to the ed_pdf folder.")
            return
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        final_documents = text_splitter.split_documents(docs)
        st.session_state.vectors = FAISS.from_documents(
            final_documents,
            st.session_state.embeddings,
        )


# File uploader
st.markdown("### Upload PDF Documents")
uploaded_files = st.file_uploader(
    "Drag and drop your PDF files here or click to select",
    type="pdf",
    accept_multiple_files=True,
)

prompt_text = st.text_input("Enter the question from your documents")

if st.button("Load"):
    try:
        docs = []
        
        # Process uploaded files
        if uploaded_files:
            from PyPDF2 import PdfReader
            from langchain_core.documents import Document
            
            for uploaded_file in uploaded_files:
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    text += page.extract_text()
                
                if text:
                    doc = Document(page_content=text, metadata={"source": uploaded_file.name})
                    docs.append(doc)
            
            if docs:
                st.success(f"Loaded {len(uploaded_files)} PDF(s) from uploads")
        else:
            st.info("No files uploaded. Attempting to load from ed_pdf folder...")
        
        vector_embedding(docs if docs else None)
        st.success("Database is ready")
    except Exception as exc:
        st.error(f"Failed to load documents: {exc}")


if prompt_text:
    if "vectors" not in st.session_state:
        st.warning("Click Load first to build the document index.")
    else:
        retriever = st.session_state.vectors.as_retriever()
        
        # Create a LCEL chain for RAG
        chain = (
            {"context": retriever, "input": lambda x: x}
            | {"context": lambda x: "\n\n".join([doc.page_content for doc in x["context"]]), "input": lambda x: x["input"]}
            | {"context": lambda x: x["context"], "input": lambda x: x["input"], "answer": prompt | llm | StrOutputParser()}
            | (lambda x: {"answer": x["answer"], "context": []})
        )
        
        start = time.process_time()
        try:
            # Simple retrieval and generation approach
            docs = retriever.invoke(prompt_text)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Create a prompt with context
            formatted_prompt = f"""Please answer the question strictly based on the provided context.
Ensure the response is accurate, concise, and directly addresses the question.

<context>
{context_text}
</context>

<question>
{prompt_text}
</question>"""
            
            # Invoke the LLM
            response_text = llm.invoke(formatted_prompt).content
            response_time = time.process_time() - start

            st.markdown("### AI Response")
            st.success(response_text)
            st.write(f"Response time: {response_time:.2f} seconds")

            with st.expander("Document similarity results"):
                st.markdown("Below are the most relevant document chunks.")
                for index, doc in enumerate(docs, start=1):
                    st.markdown(
                        f"""
                        <div class="card">
                            <strong>Chunk {index}</strong>
                            <p>{doc.page_content}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        except Exception as e:
            st.error(f"Error generating response: {e}")




