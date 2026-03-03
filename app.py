import streamlit as st
import os
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai

load_dotenv()

st.set_page_config(page_title="Academic RAG Assistant", layout="wide", page_icon="🎓")
st.title("🎓 Academic RAG")

# ── 1. EMBEDDINGS LOCALES ────────────────────────────────────────────────────
@st.cache_resource
def get_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ── 2. EXTRACCIÓN DE METADATOS DEL PDF ──────────────────────────────────────
_NOISE_PATTERNS = re.compile(
    r"""(
        \d{1,2}[-]\d{1,2}
      | ^[\d\s\-,\.]+$
      | proceedings|conference|workshop|journal|volume|vol\.|pages?|arxiv
      | http|www\.|doi:|isbn|issn
      | abstract|introduction|keywords|copyright|all\s+rights
      | received|accepted|published|revised
    )""",
    re.IGNORECASE | re.VERBOSE,
)

def _clean_lines(text: str, max_lines: int = 30) -> list:
    lines = []
    for line in text.splitlines()[:max_lines]:
        line = line.strip()
        if len(line) < 4:
            continue
        if _NOISE_PATTERNS.search(line):
            continue
        lines.append(line)
    return lines

def extract_pdf_metadata(pdf_reader: PdfReader) -> dict:
    info   = pdf_reader.metadata or {}
    title  = (info.get("/Title",  "") or "").strip()
    author = (info.get("/Author", "") or "").strip()
    year   = ""

    for date_key in ("/CreationDate", "/ModDate"):
        raw = info.get(date_key, "")
        m = re.search(r"(20\d{2}|19\d{2})", str(raw))
        if m:
            year = m.group(1)
            break

    first_text = ""
    if pdf_reader.pages:
        first_text = pdf_reader.pages[0].extract_text() or ""
        if len(pdf_reader.pages) > 1 and len(first_text) < 200:
            first_text += "\n" + (pdf_reader.pages[1].extract_text() or "")

    clean = _clean_lines(first_text)

    if not title and clean:
        title = max(clean[:10], key=len)[:150]

    if not author and clean:
        author_pat = re.compile(
            r"^[A-Z][a-z]+([\s,]+[A-Z][a-z.]+){1,5}$"
        )
        for line in clean:
            if author_pat.match(line) and line != title:
                author = line[:150]
                break
        if not author:
            for line in clean:
                if line != title and len(line) > 5:
                    author = line[:150]
                    break

    if not year and first_text:
        m = re.search(r"\b(20\d{2}|19\d{2})\b", first_text)
        if m:
            year = m.group(1)

    return {"title": title or "", "author": author or "", "year": year or ""}

# ── 3. PROCESAMIENTO DE PDF ──────────────────────────────────────────────────
def process_pdf(pdf):
    pdf_reader = PdfReader(pdf)
    metadata   = extract_pdf_metadata(pdf_reader)
    splitter   = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    texts, metas = [], []
    for i, page in enumerate(pdf_reader.pages):
        text = page.extract_text()
        if text:
            for chunk in splitter.split_text(text):
                texts.append(chunk)
                metas.append({"page": i + 1})
    vector_store = FAISS.from_texts(texts, get_embedder(), metas)
    return vector_store, metadata

# ── 4. FORMATOS DE CITA ──────────────────────────────────────────────────────
CITE_FORMATS = {
    "APA":       "APA 7.a — (Jiang, 2023, p. 7)",
    "IEEE":      "IEEE — [1]",
    "Vancouver": "Vancouver — (1)",
    "Harvard":   "Harvard — (Jiang 2023, p. 7)",
    "Chicago":   "Chicago — (Jiang 2023, 7)",
}

def _first_surname(author: str) -> str:
    if not author or author in ("Autor desconocido", ""):
        return "Autor"
    return author.split(",")[0].strip().split()[-1]

def build_cite_prompt(fmt_key: str, meta: dict) -> str:
    author  = meta.get("author", "Autor desconocido")
    year    = meta.get("year",   "s.f.")
    title   = meta.get("title",  "Sin titulo")
    surname = _first_surname(author)

    prompts = {

        "APA": f"""FORMATO DE CITA: APA 7.a edicion

CITA EN TEXTO — inmediatamente despues de cada afirmacion:
  ({surname}, {year}, p. N)   <- sustituye N por el numero de pagina real
  Con 2 autores: (Apellido1 y Apellido2, {year}, p. N)
  Con 3 o mas:   ({surname} et al., {year}, p. N)
  Paginas consecutivas: pp. N-N  /  No consecutivas: pp. N, N

REGLA DE CITAS REPETIDAS:
  - Si todo el parrafo proviene de la misma fuente: cita al inicio del parrafo y no repitas en cada oracion
  - Si cambias de pagina o de tema dentro del parrafo: nueva cita con la pagina actualizada
  - Si hay ambiguedad sobre la fuente: repite la cita
  - NUNCA omitas el ano en ninguna cita en texto

SECCION "Referencias" AL FINAL de tu respuesta:
  {author} ({year}). {title}.

REGLAS:
- USA coma entre apellido y ano: (Apellido, ano) <- distingue APA de Harvard
- Indicador de pagina: "p." singular, "pp." para rango o paginas multiples
- Cada parrafo que use el contexto DEBE tener al menos una cita en texto""",

        "IEEE": f"""FORMATO DE CITA: IEEE

CITA EN TEXTO — inmediatamente despues de cada afirmacion:
  [N, p. X]   <- numero secuencial entre corchetes + pagina especifica
  Ejemplo: La recuperacion iterativa mejora los resultados [1, p. 7].
  Rango de paginas: [1, pp. 7-9]
  NUNCA uses (Autor, ano): IEEE solo admite numeros entre corchetes

REGLA CRITICA DE NUMEROS:
  - Misma fuente, paginas distintas = SIEMPRE el MISMO numero: [1, p. 7] ... [1, p. 14]
  - NUNCA crees [1], [2], [3] distintos para el mismo articulo con distintas paginas
  - Solo se crea un numero nuevo si es una fuente DIFERENTE

SECCION "References" AL FINAL de tu respuesta (UNA sola entrada por fuente):
  [1] {author}, "{title}," {year}.
  Formato completo: [N] I. Apellido, "Titulo articulo," en Nombre Revista/Conf., ano.

REGLAS:
- Numera referencias en ORDEN DE APARICION en el texto
- Titulo del articulo entre comillas; revista/conferencia en cursiva
- Las paginas especificas van en el texto [1, p. X], NO en la referencia final""",

        "Vancouver": f"""FORMATO DE CITA: Vancouver

CITA EN TEXTO — inmediatamente despues de cada afirmacion:
  (N)   <- numero secuencial entre parentesis, desde (1)
  Ejemplo: El umbral de confianza controla la recuperacion (1).
  NUNCA uses (Autor, ano): Vancouver solo admite numeros entre parentesis
  NUNCA añadas paginas en la cita en texto: solo el numero, nada mas

REGLA CRITICA DE NUMEROS:
  - Misma fuente repetida = SIEMPRE el MISMO numero: si ya es (1), siempre es (1)
  - Solo se asigna numero nuevo si es fuente DIFERENTE

SECCION "Referencias" AL FINAL de tu respuesta (UNA sola entrada por fuente):
  1. {author}. {title}. {year}.
  Formato completo: N. Apellido Iniciales. Titulo sin cursiva ni comillas. ano.

REGLAS:
- Numera referencias en ORDEN DE APARICION en el texto
- El titulo NO va en cursiva ni entre comillas
- Las iniciales del autor van DESPUES del apellido sin punto entre ellas: Jiang Z
- Las paginas especificas NO se incluyen en la cita en texto en Vancouver""",

        "Harvard": f"""FORMATO DE CITA: Harvard (autor-fecha)

CITA EN TEXTO — inmediatamente despues de cada afirmacion:
  ({surname} {year}, p. N)   <- SIN coma entre apellido y ano
  Con 3 o mas autores: ({surname} et al. {year}, p. N)
  Paginas consecutivas: pp. N-N  /  No consecutivas: pp. N, N
  Si el autor ya aparece en la oracion: {surname} ({year}, p. N) sostiene que...

REGLA DE CITAS REPETIDAS:
  - Misma fuente, oraciones distintas: repite la cita completa cada vez con su pagina
  - Paginas distintas en una sola cita: ({surname} {year}, pp. 7, 14)
  - NUNCA omitas el ano en citas repetidas

SECCION "Reference List" AL FINAL de tu respuesta:
  {author} {year}, {title}.
  Formato completo: Apellido, Iniciales ano, Titulo en cursiva, Ciudad: Editorial.

REGLAS:
- NO hay coma entre apellido y ano: (Apellido ano) <- diferencia clave con APA
- Cada oracion que use el contexto DEBE llevar su cita""",

        "Chicago": f"""FORMATO DE CITA: Chicago 17.a edicion, estilo autor-fecha (exportacion por defecto de Zotero)

CITA EN TEXTO — inmediatamente despues de cada afirmacion:
  ({surname} {year}, N)   <- SIN "p." antes del numero, solo el numero de pagina
  Con 4 o mas autores: ({surname} et al. {year}, N)
  Paginas consecutivas: ({surname} {year}, N-N)  /  No consecutivas: ({surname} {year}, N, N)

REGLA DE CITAS REPETIDAS (CMOS 17 oficial):
  - Primera mencion en el parrafo: cita completa ({surname} {year}, 7)
  - Misma fuente, mismo parrafo, pagina distinta: solo el numero de pagina (14)
  - Parrafo nuevo: repite siempre la cita completa ({surname} {year}, N)

SECCION "Bibliography" AL FINAL de tu respuesta (UNA sola entrada por fuente):
  {author}. {year}. "{title}."
  Formato completo: Apellido, Nombre completo. ano. "Titulo articulo." Nombre Revista vol (num): pp. doi.

REGLAS:
- Chicago autor-fecha NUNCA escribe "p." en la cita en texto (diferencia con APA y Harvard)
- Titulo del articulo entre comillas; revista/libro en cursiva
- En la bibliografia el NOMBRE COMPLETO del autor va tras el apellido, no solo inicial""",
    }

    return prompts.get(fmt_key, prompts["APA"])

# ── 5. GENERACION CON FALLBACK ───────────────────────────────────────────────
MODELS = [
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
]

def generate_answer(client, prompt: str):
    last_err = None
    for model_id in MODELS:
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            return response.text, model_id
        except Exception as e:
            last_err = e
            if "429" in str(e) or "404" in str(e):
                continue
            raise e
    raise last_err

# ── 6. ESTADO INICIAL ────────────────────────────────────────────────────────
for key, default in [
    ("chat_history",   []),
    ("is_loading",     False),
    ("vector_db",      None),
    ("pdf_meta",       None),
    ("meta_confirmed", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── 7. SUBIDA DE PDF ─────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Sube un Paper Academico (PDF)", type="pdf",
    disabled=st.session_state.is_loading,
)

if uploaded_file and st.session_state.vector_db is None:
    with st.spinner("Indexando documento localmente..."):
        vdb, meta = process_pdf(uploaded_file)
        st.session_state.vector_db      = vdb
        st.session_state.pdf_meta       = meta
        st.session_state.meta_confirmed = False
    st.success("Documento indexado. Revisa los metadatos antes de consultar.")

# ── 8. PANEL DE METADATOS ────────────────────────────────────────────────────
if st.session_state.vector_db and st.session_state.pdf_meta is not None:
    meta      = st.session_state.pdf_meta
    confirmed = st.session_state.meta_confirmed

    with st.expander(
        "Metadatos del paper — revisa y corrige si es necesario",
        expanded=not confirmed,
    ):
        st.caption("Datos extraidos automaticamente del PDF. Corrigelos si alguno esta mal.")
        c1, c2, c3 = st.columns([3, 2, 1])
        new_title  = c1.text_input("Titulo",   value=meta.get("title",  ""), key="meta_title")
        new_author = c2.text_input(
            "Autor/es", value=meta.get("author", ""), key="meta_author",
            help="Ej: Jiang, Z.; Xu, S.  /  Smith, J. et al.",
        )
        new_year = c3.text_input("Ano", value=meta.get("year", ""), key="meta_year", max_chars=4)

        if st.button("Confirmar metadatos"):
            st.session_state.pdf_meta = {
                "title":  new_title  or "Titulo desconocido",
                "author": new_author or "Autor desconocido",
                "year":   new_year   or "s.f.",
            }
            st.session_state.meta_confirmed = True
            st.rerun()

    if confirmed:
        m = st.session_state.pdf_meta
        st.info(f"Documento: {m['title']} — {m['author']} ({m['year']})")

# ── 9. HISTORIAL DE CHAT ─────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown("### Historial de conversacion")
    for i, entry in enumerate(st.session_state.chat_history):
        label = entry["question"][:80] + ("..." if len(entry["question"]) > 80 else "")
        with st.expander(
            f"P{i+1}: {label}",
            expanded=(i == len(st.session_state.chat_history) - 1),
        ):
            col_a, col_s = st.columns([2, 1])
            with col_a:
                st.markdown(f"**Pregunta:** {entry['question']}")
                st.markdown("**Respuesta:**")
                st.write(entry["answer"])
                st.caption(f"Modelo: {entry['model']} | Formato: {entry['fmt']}")
            with col_s:
                st.markdown("**Paginas consultadas:**")
                for page_num, snippet in entry["sources"]:
                    with st.expander(f"Pagina {page_num}"):
                        st.caption(snippet)

    if st.button("Limpiar historial", disabled=st.session_state.is_loading):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

# ── 10. FORMULARIO DE PREGUNTA ───────────────────────────────────────────────
if st.session_state.vector_db and st.session_state.meta_confirmed:
    col_q, col_fmt = st.columns([3, 1])

    with col_q:
        question = st.text_input(
            "Haz una pregunta tecnica sobre el paper:",
            disabled=st.session_state.is_loading,
            key="question_input",
        )

    with col_fmt:
        fmt_key = st.radio(
            "Formato de cita:",
            options=list(CITE_FORMATS.keys()),
            format_func=lambda k: CITE_FORMATS[k],
            disabled=st.session_state.is_loading,
            key="cite_format",
        )

    ask_btn = st.button(
        "Consultar",
        disabled=st.session_state.is_loading or not question,
    )

    if ask_btn and question:
        st.session_state.is_loading = True
        st.rerun()

# ── 11. PROCESAMIENTO ────────────────────────────────────────────────────────
if st.session_state.is_loading and st.session_state.vector_db:
    question = st.session_state.get("question_input", "")
    fmt_key  = st.session_state.get("cite_format", "APA")
    meta     = st.session_state.pdf_meta

    docs = st.session_state.vector_db.similarity_search(question, k=3)
    context_text = "\n".join(
        f"[PAGINA {d.metadata['page']}]: {d.page_content}" for d in docs
    )

    cite_instructions = build_cite_prompt(fmt_key, meta)

    prompt = f"""Eres un asistente academico experto. Responde SOLO con la informacion del contexto. No inventes datos.

{cite_instructions}

CONTEXTO DEL PAPER (cada fragmento indica su pagina de origen):
{context_text}

PREGUNTA:
{question}
"""

    with st.spinner("Consultando IA..."):
        try:
            answer, model_used = generate_answer(client, prompt)
            sources = [(d.metadata["page"], d.page_content) for d in docs]
            st.session_state.chat_history.append({
                "question": question,
                "answer":   answer,
                "model":    model_used,
                "fmt":      CITE_FORMATS[fmt_key],
                "sources":  sources,
            })
        except Exception as e:
            st.error(f"Error de conexion: {e}")
            st.info("Espera unos segundos y vuelve a intentarlo, o verifica tu API Key.")

    st.session_state.is_loading = False
    st.rerun()

elif not st.session_state.vector_db:
    st.info("Sube un PDF para comenzar.")
elif not st.session_state.meta_confirmed:
    st.info("Confirma los metadatos del paper antes de hacer preguntas.")