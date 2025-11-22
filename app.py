import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI P&ID Search Engine", page_icon="⚙️", layout="wide")

# --- HEADER ---
st.title("⚙️ AI Engineering Search Engine")
st.caption("Upload P&IDs, Manuals, or Datasheets -> Ask Questions. Powered by Llama 3 & Groq (Free).")

# --- SIDEBAR: API CONFIGURATION ---
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Check if API Key is in secrets, otherwise ask user
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("✅ API Key loaded from Secrets")
    else:
        api_key = st.text_input("Enter Groq API Key:", type="password")
        if not api_key:
            st.warning("Get a free key at: https://console.groq.com/keys")

# --- MAIN APP LOGIC ---
if api_key:
    # 1. File Uploader
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])

    if uploaded_file:
        try:
            with st.spinner("🔍 Processing PDF (OCR & Vectorizing)..."):
                
                # Save uploaded file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                # Load PDF using PyMuPDF
                loader = PyMuPDFLoader(tmp_file_path)
                data = loader.load()

                # Split text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(data)

                # Create Embeddings (Local CPU)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                # Create Vector Store (FAISS)
                vector_store = FAISS.from_documents(chunks, embeddings)
                
                # Setup Retriever
                retriever = vector_store.as_retriever()

                # Setup LLM (Groq)
                llm = ChatGroq(
                    temperature=0, 
                    groq_api_key=api_key, 
                    model_name="llama3-8b-8192"
                )

                # Create QA Chain
                qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever, chain_type="stuff")

                st.success(f"✅ Processed {len(chunks)} text chunks from the PDF!")

            # 2. User Question Area
            query = st.text_input("Ask a question about your document:")
            
            if query:
                with st.spinner("🤖 Thinking..."):
                    response = qa_chain.run(query)
                    st.markdown("### 💡 AI Answer")
                    st.write(response)
                    
                    with st.expander("View Source Text (Verification)"):
                        docs = retriever.get_relevant_documents(query)
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Source {i+1}:** {doc.page_content[:300]}...")
                            st.markdown("---")

            # Cleanup
            os.remove(tmp_file_path)

        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("👈 Please enter your Groq API key in the sidebar to start.")
