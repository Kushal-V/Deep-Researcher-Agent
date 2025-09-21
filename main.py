import streamlit as st
import fitz
import docx
from bs4 import BeautifulSoup
import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import io
import os # Import the os library

@st.cache_resource
def load_models():
    print("--- SCRIPT: Attempting to load all models. ---")
    # Get the token from Streamlit secrets
    hf_token = os.environ.get("HUGGING_FACE_TOKEN")

    print("--- SCRIPT: Loading SentenceTransformer retriever model. ---")
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("--- SCRIPT: Retriever loaded successfully. ---")

    model_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    print(f"--- SCRIPT: Loading tokenizer for {model_path}. ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
    print("--- SCRIPT: Tokenizer loaded successfully. ---")

    print(f"--- SCRIPT: Loading generator model {model_path}. This is the memory-intensive step. ---")
    generator_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu", token=hf_token)
    print("--- SCRIPT: Generator model loaded successfully. ---")

    print("--- SCRIPT: All models loaded. Application should be ready. ---")
    return retriever_model, tokenizer, generator_model

# --- The rest of your code remains exactly the same ---
# (Text Extraction Functions and Main App Logic)
# ...
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

st.title("🧠 Deep Researcher Agent (Multi-Format)")
st.write("Upload a PDF, DOCX, TXT, or HTML file to begin.")

retriever, tokenizer, generator = load_models()

if 'hnsw_index' not in st.session_state:
    st.session_state.hnsw_index = None
    st.session_state.documents = []

uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt", "html"])

if uploaded_file is not None:
    st.info(f"Processing {uploaded_file.name}... Please wait.")
    file_bytes = uploaded_file.getvalue()
    
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(file_bytes)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        text = extract_text_from_docx(file_bytes)
    elif uploaded_file.type == "text/plain":
        text = extract_text_from_txt(file_bytes)
    elif uploaded_file.type == "text/html":
        text = extract_text_from_html(file_bytes)

    st.session_state.documents = [text]
    
    doc_embeddings = retriever.encode(st.session_state.documents)
    dimension = doc_embeddings.shape[1]
    
    hnsw_index = hnswlib.Index(space='l2', dim=dimension)
    hnsw_index.init_index(max_elements=len(st.session_state.documents), ef_construction=200, M=16)
    hnsw_index.add_items(doc_embeddings, np.arange(len(st.session_state.documents)))
    st.session_state.hnsw_index = hnsw_index
    
    st.success(f"{uploaded_file.name} processed successfully! You can now ask questions.")

if st.session_state.hnsw_index is not None:
    query = st.text_input("Ask a question about the document:")

    if query:
        query_embedding = retriever.encode([query])
        labels, distances = st.session_state.hnsw_index.knn_query(query_embedding, k=1)
        retrieved_chunk = st.session_state.documents[labels[0][0]]

        prompt_template = f"""
        Answer the following question using only the context provided.
        Context: "{retrieved_chunk}"
        Question: "{query}"
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