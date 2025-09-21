import streamlit as st
import fitz  # PyMuPDF for PDFs
import docx  # python-docx for DOCX
from bs4 import BeautifulSoup # for HTML
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import io

# Use Streamlit's cache to load models only once
@st.cache_resource
def load_models():
    print("Loading models...")
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
    model_path = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    generator_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
    print("Models loaded successfully.")
    return retriever_model, tokenizer, generator_model

# --- Text Extraction Functions for each file type ---
def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = " ".join([page.get_text() for page in doc])
    doc.close()
    return text

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    return " ".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(file_bytes):
    return file_bytes.decode('utf-8')

def extract_text_from_html(file_bytes):
    soup = BeautifulSoup(file_bytes, "html.parser")
    return soup.get_text()

# --- Main App Logic ---
st.title("🧠 Deep Researcher Agent (Multi-Format)")
st.write("Upload a PDF, DOCX, TXT, or HTML file to begin.")

retriever, tokenizer, generator = load_models()

if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
    st.session_state.documents = []

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt", "html"])

if uploaded_file is not None:
    st.info(f"Processing {uploaded_file.name}... Please wait.")
    file_bytes = uploaded_file.getvalue()
    
    # Dispatcher to select the correct text extraction function
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(file_bytes)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(file_bytes)
    elif uploaded_file.type == "text/plain":
        text = extract_text_from_txt(file_bytes)
    elif uploaded_file.type == "text/html":
        text = extract_text_from_html(file_bytes)

    # For simplicity, we'll treat the whole text as one document.
    # A more advanced version would split it into smaller chunks.
    st.session_state.documents = [text]
    
    doc_embeddings = retriever.encode(st.session_state.documents)
    dimension = doc_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(doc_embeddings)
    st.session_state.faiss_index = faiss_index
    
    st.success(f"{uploaded_file.name} processed successfully! You can now ask questions.")

if st.session_state.faiss_index is not None:
    query = st.text_input("Ask a question about the document:")

    if query:
        query_embedding = retriever.encode([query])
        distances, indices = st.session_state.faiss_index.search(query_embedding, k=1)
        retrieved_chunk = st.session_state.documents[indices[0][0]]

        prompt_template = f"""
        Answer the following question using only the context provided.

        Context:
        "{retrieved_chunk}"

        Question:
        "{query}"

        Answer:
        """

        with st.spinner("Generating answer..."):
            input_ids = tokenizer(prompt_template, return_tensors="pt").to("cpu")
            outputs = generator.generate(**input_ids, max_new_tokens=200)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("Answer:")[1].strip()

        st.markdown("---")
        st.write("### Answer")
        st.write(answer)