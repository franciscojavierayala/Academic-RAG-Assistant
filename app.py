import streamlit as st
import streamlit.components.v1 as components
import os
import re
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai

# ── TRADUCCIONES ─────────────────────────────────────────────────────────────
LANGS = {
    "es": {
        "page_title": "Asistente Academico RAG",
        "title": "🎓 RAG Academico",
        "upload_label": "Sube un Paper Academico (PDF)",
        "indexing": "Indexando documento localmente...",
        "indexed_ok": "Documento indexado. Revisa los metadatos antes de consultar.",
        "meta_expander": "Metadatos del paper — revisa y corrige si es necesario",
        "meta_caption": "Datos extraidos automaticamente del PDF. Corrigelos si alguno esta mal.",
        "meta_title": "Titulo", "meta_author": "Autor/es", "meta_year": "Ano",
        "meta_author_help": "Ej: Jiang, Z.; Xu, S.  /  Smith, J. et al.",
        "meta_confirm_btn": "Confirmar metadatos",
        "meta_info": "Documento: {title} — {author} ({year})",
        "chat_history": "### Historial de conversacion",
        "question_label": "Pregunta:", "answer_label": "Respuesta:",
        "pages_label": "**Paginas consultadas:**", "page_label": "Pagina {n}",
        "clear_btn": "Limpiar historial",
        "question_input": "Haz una pregunta tecnica sobre el paper:",
        "cite_format_label": "Formato de cita:",
        "ask_btn": "Consultar",
        "consulting": "Consultando IA...",
        "conn_error": "Error de conexion: {e}",
        "retry_info": "Espera unos segundos y vuelve a intentarlo, o verifica tu API Key.",
        "upload_info": "Sube un PDF para comenzar.",
        "meta_info_pending": "Confirma los metadatos del paper antes de hacer preguntas.",
        "lang_btn": "🌐 English",
        "model_fmt": "Modelo: {model} | Formato: {fmt}",
        "entry_label": "P{i}: {label}",
        "unknown_author": "Autor desconocido",
        "unknown_title": "Titulo desconocido",
        "unknown_year": "s.f.",
        "system_prompt": "Eres un asistente academico experto. Responde SOLO con la informacion del contexto. No inventes datos.",
        "context_label": "CONTEXTO DEL PAPER (cada fragmento indica su pagina de origen):",
        "question_prompt": "PREGUNTA:",
        "lang_instruction": "Responde siempre en español.",
    },
    "en": {
        "page_title": "Academic RAG Assistant",
        "title": "🎓 Academic RAG",
        "upload_label": "Upload an Academic Paper (PDF)",
        "indexing": "Indexing document locally...",
        "indexed_ok": "Document indexed. Review the metadata before querying.",
        "meta_expander": "Paper metadata — review and correct if needed",
        "meta_caption": "Data extracted automatically from the PDF. Fix any errors.",
        "meta_title": "Title", "meta_author": "Author(s)", "meta_year": "Year",
        "meta_author_help": "E.g: Jiang, Z.; Xu, S.  /  Smith, J. et al.",
        "meta_confirm_btn": "Confirm metadata",
        "meta_info": "Document: {title} — {author} ({year})",
        "chat_history": "### Conversation history",
        "question_label": "Question:", "answer_label": "Answer:",
        "pages_label": "**Pages consulted:**", "page_label": "Page {n}",
        "clear_btn": "Clear history",
        "question_input": "Ask a technical question about the paper:",
        "cite_format_label": "Citation format:",
        "ask_btn": "Ask",
        "consulting": "Querying AI...",
        "conn_error": "Connection error: {e}",
        "retry_info": "Wait a few seconds and try again, or check your API Key.",
        "upload_info": "Upload a PDF to get started.",
        "meta_info_pending": "Confirm the paper metadata before asking questions.",
        "lang_btn": "🌐 Español",
        "model_fmt": "Model: {model} | Format: {fmt}",
        "entry_label": "Q{i}: {label}",
        "unknown_author": "Unknown author",
        "unknown_title": "Unknown title",
        "unknown_year": "n.d.",
        "system_prompt": "You are an expert academic assistant. Answer ONLY using the information from the context. Do not invent data.",
        "context_label": "PAPER CONTEXT (each fragment indicates its source page):",
        "question_prompt": "QUESTION:",
        "lang_instruction": "Always respond in English.",
    },
}

load_dotenv()

st.set_page_config(page_title="Academic RAG Assistant", layout="wide", page_icon="🎓")

# ── ESTADO INICIAL ────────────────────────────────────────────────────────────
for key, default in [
    ("chat_history",      []),
    ("is_loading",        False),
    ("vector_db",         None),
    ("pdf_meta",          None),
    ("meta_confirmed",    False),
    ("lang",              "en"),
    ("cite_format",       "APA"),
    ("last_pdf_name",     None),
    ("scroll_to_answer",  False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

T = LANGS[st.session_state.lang]

# ── TITULO + BOTON DE IDIOMA ──────────────────────────────────────────────────
col_title, col_lang = st.columns([9, 1])
with col_title:
    st.title(T["title"])
with col_lang:
    st.write("")  # espaciado vertical
    if st.button(T["lang_btn"]):
        st.session_state.lang = "en" if st.session_state.lang == "es" else "es"
        st.rerun()

# ── 1. EMBEDDINGS LOCALES ─────────────────────────────────────────────────────
@st.cache_resource
def get_embedder():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# ── 2. EXTRACCIÓN DE METADATOS DEL PDF ───────────────────────────────────────
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
        author_pat = re.compile(r"^[A-Z][a-z]+([\s,]+[A-Z][a-z.]+){1,5}$")
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

# ── 3. PROCESAMIENTO DE PDF ───────────────────────────────────────────────────
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

# ── 4. FORMATOS DE CITA ───────────────────────────────────────────────────────
CITE_FORMATS = {
    "APA":       "APA 7.a — (Jiang, 2023, p. 7)",
    "IEEE":      "IEEE — [1]",
    "Vancouver": "Vancouver — (1)",
    "Harvard":   "Harvard — (Jiang 2023, p. 7)",
    "Chicago":   "Chicago — (Jiang 2023, 7)",
}

def _first_surname(author: str) -> str:
    if not author or author in ("Autor desconocido", "Unknown author", ""):
        return "Autor"
    return author.split(",")[0].strip().split()[-1]

def build_cite_prompt(fmt_key: str, meta: dict) -> str:
    author  = meta.get("author", T["unknown_author"])
    year    = meta.get("year",   T["unknown_year"])
    title   = meta.get("title",  T["unknown_title"])
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

# ── 5. GENERACION CON FALLBACK ────────────────────────────────────────────────
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

# ── 7. SUBIDA DE PDF ──────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    T["upload_label"], type="pdf",
    disabled=st.session_state.is_loading,
)

# ── FIX: re-procesar si el archivo cambia (segunda, tercera vez...) ───────────
last_name = st.session_state.get("last_pdf_name")
if uploaded_file and (st.session_state.vector_db is None or uploaded_file.name != last_name):
    st.session_state.chat_history   = []
    st.session_state.meta_confirmed = False
    with st.spinner(T["indexing"]):
        vdb, meta = process_pdf(uploaded_file)
        st.session_state.vector_db     = vdb
        st.session_state.pdf_meta      = meta
        st.session_state.last_pdf_name = uploaded_file.name
    st.success(T["indexed_ok"])

# ── 8. PANEL DE METADATOS ─────────────────────────────────────────────────────
if st.session_state.vector_db and st.session_state.pdf_meta is not None:
    meta      = st.session_state.pdf_meta
    confirmed = st.session_state.meta_confirmed

    with st.expander(T["meta_expander"], expanded=not confirmed):
        st.caption(T["meta_caption"])
        c1, c2, c3 = st.columns([3, 2, 1])
        _pdf_key = st.session_state.get("last_pdf_name", "_")
        new_title  = c1.text_input(T["meta_title"],  value=meta.get("title",  ""), key=f"meta_title_{_pdf_key}")
        new_author = c2.text_input(
            T["meta_author"], value=meta.get("author", ""), key=f"meta_author_{_pdf_key}",
            help=T["meta_author_help"],
        )
        new_year = c3.text_input(T["meta_year"], value=meta.get("year", ""), key=f"meta_year_{_pdf_key}", max_chars=4)

        if st.button(T["meta_confirm_btn"]):
            st.session_state.pdf_meta = {
                "title":  new_title  or T["unknown_title"],
                "author": new_author or T["unknown_author"],
                "year":   new_year   or T["unknown_year"],
            }
            st.session_state.meta_confirmed = True
            st.rerun()

    if confirmed:
        m = st.session_state.pdf_meta
        st.info(T["meta_info"].format(title=m["title"], author=m["author"], year=m["year"]))

# ── 9. HISTORIAL DE CHAT ──────────────────────────────────────────────────────
if st.session_state.chat_history:
    st.markdown(T["chat_history"])
    for i, entry in enumerate(st.session_state.chat_history):
        label = entry["question"][:80] + ("..." if len(entry["question"]) > 80 else "")
        with st.expander(
            T["entry_label"].format(i=i + 1, label=label),
            expanded=(i == len(st.session_state.chat_history) - 1),
        ):
            col_a, col_s = st.columns([2, 1])
            with col_a:
                st.markdown(f"**{T['question_label']}** {entry['question']}")
                st.markdown(f"**{T['answer_label']}**")
                st.write(entry["answer"])
                st.caption(T["model_fmt"].format(model=entry["model"], fmt=entry["fmt"]))
            with col_s:
                st.markdown(T["pages_label"])
                for page_num, snippet in entry["sources"]:
                    with st.expander(T["page_label"].format(n=page_num)):
                        st.caption(snippet)

    if st.button(T["clear_btn"], disabled=st.session_state.is_loading):
        st.session_state.chat_history = []
        st.rerun()

    # ── FIX: scroll automático a la última respuesta ──────────────────────────
    if st.session_state.pop("scroll_to_answer", False):
        components.html("""
            <script>
                window.parent.document
                    .querySelector('section.main')
                    .scrollTo({top: 999999, behavior: 'smooth'});
            </script>
        """, height=0)

    st.divider()

# ── 10. FORMULARIO DE PREGUNTA ────────────────────────────────────────────────
if st.session_state.vector_db and st.session_state.meta_confirmed and not st.session_state.is_loading:
    with st.form(key="question_form", clear_on_submit=False):
        col_q, col_fmt = st.columns([3, 1])

        with col_q:
            question = st.text_input(
                T["question_input"],
                disabled=st.session_state.is_loading,
                key="question_input",
            )

        with col_fmt:
            fmt_key = st.radio(
                T["cite_format_label"],
                options=list(CITE_FORMATS.keys()),
                format_func=lambda k: CITE_FORMATS[k],
                disabled=st.session_state.is_loading,
                key="cite_format",
            )

        ask_btn = st.form_submit_button(
            T["ask_btn"],
            disabled=st.session_state.is_loading,
        )

    if ask_btn and question:
        st.session_state.is_loading       = True
        st.session_state.pending_question = question
        st.session_state.pending_fmt      = fmt_key or "APA"
        st.rerun()

# ── 11. PROCESAMIENTO ─────────────────────────────────────────────────────────
if st.session_state.is_loading and st.session_state.vector_db:
    question = st.session_state.get("pending_question", "")
    fmt_key  = st.session_state.get("pending_fmt", "APA")
    meta     = st.session_state.pdf_meta

    docs = st.session_state.vector_db.similarity_search(question, k=3)
    context_text = "\n".join(
        f"[PAGINA {d.metadata['page']}]: {d.page_content}" for d in docs
    )

    cite_instructions = build_cite_prompt(fmt_key, meta)

    prompt = f"""{T["system_prompt"]}
{T["lang_instruction"]}

{cite_instructions}

{T["context_label"]}
{context_text}

{T["question_prompt"]}
{question}
"""

    with st.spinner(T["consulting"]):
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
            st.session_state.scroll_to_answer = True   # ← activar scroll
        except Exception as e:
            st.error(T["conn_error"].format(e=e))
            st.info(T["retry_info"])

    st.session_state.is_loading = False
    st.rerun()

elif not st.session_state.vector_db:
    st.info(T["upload_info"])
elif not st.session_state.meta_confirmed:
    st.info(T["meta_info_pending"])