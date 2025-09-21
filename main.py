import streamlit as st
import fitz
import docx
from bs4 import BeautifulSoup
import hnswlib
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import io

# Use Streamlit's cache to load models only once
@st.cache_resource
def load_models():
    print("Loading models...")
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Using a smaller model that doesn't require a token and fits in memory
    generator_pipe = pipeline('text-generation', model='distilgpt2')
    print("Models loaded successfully.")
    return retriever_model, generator_pipe

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

retriever, generator = load_models()

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
            # Use the pipeline for generation
            response = generator(prompt_template, max_new_tokens=100, num_return_sequences=1)
            # Extract the generated text and then the answer part
            full_text = response[0]['generated_text']
            if "Answer:" in full_text:
                answer = full_text.split("Answer:")[1].strip()
            else:
                answer = "Could not find a specific answer in the context."


        st.markdown("---")
        st.write("### Answer")
        st.write(answer)