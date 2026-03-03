import streamlit as st
import os
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai

# Cargar variables de entorno
load_dotenv()

st.set_page_config(page_title="Academic RAG Assistant", layout="wide", page_icon="🎓")
st.title("🎓 Academic RAG: Trazabilidad Total")

# --- 1. CONFIGURACIÓN DE EMBEDDINGS LOCALES ---
@st.cache_resource
def get_embedder():
    # Esto corre en tu PC, no gasta cuota y evita el error 404/429 al subir el PDF
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Inicializar cliente de Gemini
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- 2. PROCESAMIENTO DE PDF ---
def process_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    chunks_data = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            page_chunks = text_splitter.split_text(text)
            for chunk in page_chunks:
                chunks_data.append({
                    "text": chunk,
                    "metadata": {"page": i + 1}
                })
    
    texts = [c["text"] for c in chunks_data]
    metadatas = [c["metadata"] for c in chunks_data]
    vector_store = FAISS.from_texts(texts, get_embedder(), metadatas=metadatas)
    return vector_store

# --- 3. LÓGICA DE GENERACIÓN CON TUS MODELOS DISPONIBLES ---
def generate_answer_with_fallback(client, prompt):
    # Usamos los nombres exactos de tu lista de modelos
    # Nota: Probamos SIN el prefijo "models/" primero, que es como lo quiere la nueva SDK
    models_to_try = [
        "gemini-2.0-flash",       # El más rápido y nuevo
        "gemini-flash-latest",    # El más estable de la serie Flash
        "gemini-pro-latest"       # El más potente para razonamiento
    ]
    
    last_err = None
    for model_id in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_id, 
                contents=prompt
            )
            return response.text, model_id
        except Exception as e:
            last_err = e
            # Si es error de cuota (429) o no encuentra el modelo (404), saltamos al siguiente
            if "429" in str(e) or "404" in str(e):
                continue
            else:
                raise e
    raise last_err

# --- 4. INTERFAZ DE USUARIO ---
uploaded_file = st.file_uploader("Sube un Paper Académico (PDF)", type="pdf")

if uploaded_file:
    if "vector_db" not in st.session_state:
        with st.spinner("Indexando documento localmente..."):
            st.session_state.vector_db = process_pdf(uploaded_file)
            st.success("✅ Documento indexado con éxito.")

    # Fila de pregunta y formato de cita
    col_q, col_fmt = st.columns([3, 1])
    
    with col_q:
        question = st.text_input("Haz una pregunta técnica sobre el paper:")
    
    with col_fmt:
        format_choice = st.selectbox(
            "Formato de cita:",
            ["APA (Página X)", "IEEE [Página X]", "Vancouver (p. X)", "Harvard (Pág. X)"]
        )

    if question:
        # Recuperamos los 3 fragmentos más relevantes
        docs = st.session_state.vector_db.similarity_search(question, k=3)
        
        context_text = ""
        for d in docs:
            context_text += f"\n[PÁGINA {d.metadata['page']}]: {d.page_content}\n"

        prompt = f"""Responde a la pregunta basándote SOLO en el contexto proporcionado.
Es obligatorio usar el formato de cita: {format_choice}.

CONTEXTO:
{context_text}

PREGUNTA:
{question}
"""

        with st.spinner("Consultando IA..."):
            try:
                answer, model_used = generate_answer_with_fallback(client, prompt)
                
                if answer:
                    st.divider()
                    st.info(f"⚡ Modelo utilizado: {model_used} | 📚 Citas en: {format_choice}")
                    
                    res_col, src_col = st.columns([2, 1])
                    with res_col:
                        st.markdown("### 💬 Respuesta Analítica")
                        st.write(answer)
                    with src_col:
                        st.markdown("### 📍 Evidencia del PDF")
                        for d in docs:
                            with st.expander(f"Página {d.metadata['page']}"):
                                st.caption(d.page_content)
            except Exception as e:
                st.error(f"Error de conexión: {e}")
                st.info("💡 Intenta esperar 10 segundos. Si el error persiste, verifica tu conexión o API Key.")