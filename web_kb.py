import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Knowledge Base AI Dashboard", page_icon="🧠", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.top-box {
    background: linear-gradient(135deg, #1e3a8a, #06b6d4);
    padding: 25px;
    border-radius: 20px;
    color: white;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 5px 15px rgba(0,0,0,0.1);
}
.answer-box {
    background: #87CEEB;   /* Sky Blue */
    color: #00008B;        /* Deep Blue */
    padding: 15px;
    border-radius: 12px;
    font-family: Arial;
    font-size: 16px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 Knowledge Base AI")
st.sidebar.markdown("### System Panel")
st.sidebar.write("📌 Academic Assistant")
st.sidebar.write("📌 RAG Knowledge System")
st.sidebar.write("📌 PDF Notes Analyzer")
st.sidebar.markdown("---")
st.sidebar.success("System Status: ONLINE")
st.sidebar.info("Developed by Sreyoshee Saha | ECE 3rd Year")

# ---------------- HEADER DASHBOARD ----------------
st.markdown("""
<div class="top-box">
<h1>🧠 Knowledge Base AI Dashboard</h1>
<p>Academic Intelligent Assistant for PDF Knowledge Retrieval</p>
</div>
""", unsafe_allow_html=True)

# ---------------- METRICS ----------------
col1, col2, col3 = st.columns(3)
col1.metric("📄 Total Documents", "3 PDFs")
col2.metric("🔍 Vector Database", "ChromaDB Active")
col3.metric("🤖 AI Mode", "Knowledge Retrieval")

# ---------------- LOAD DATABASE ----------------
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="kb_db", embedding_function=embeddings)

db = load_db()

# ---------------- TABS UI ----------------
tab1, tab2 = st.tabs(["🔎 Ask Question", "📊 System Info"])

# ================= TAB 1 =================
with tab1:
    st.markdown("<div class='card'><h3>Ask Your Knowledge Base</h3></div>", unsafe_allow_html=True)
    query = st.text_area("Enter your academic question:", height=100, placeholder="Explain neural networks...")

    if st.button("🚀 Get Answer"):
        docs = db.similarity_search(query, k=2)
        if docs:
            st.markdown("### AI Retrieved Knowledge")
            for d in docs:
                st.markdown(f"<div class='answer-box'>{d.page_content}</div>", unsafe_allow_html=True)
        else:
            st.error("❌ No relevant information found.")

# ================= TAB 2 =================
with tab2:
    st.markdown("<div class='card'><h3>System Information</h3></div>", unsafe_allow_html=True)
    st.write("Model: Sentence Transformers")
    st.write("Database: Chroma Vector Store")
    st.write("Project Type: Academic AI Assistant")
    st.write("Year: ECE 3rd Year")

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<p style='text-align:center;'>Knowledge Base AI Dashboard | Mini Project</p>
""", unsafe_allow_html=True)