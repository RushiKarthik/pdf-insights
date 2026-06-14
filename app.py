import os
import streamlit as st
import numpy as np
from groq import Groq
from pdf_processor import extract_text, create_chunks
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="PDF Insight", page_icon="📄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #f0f4ff 0%, #faf5ff 50%, #f0fffe 100%);
    min-height: 100vh;
}
[data-testid="stAppViewContainer"] > .main > div {
    max-width: 800px;
    margin: 0 auto;
    padding: 1rem 2rem;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #c7d2fe;
    border-radius: 16px;
    padding: 0.5rem;
}
.hero { text-align: center; padding: 3rem 0 2.5rem; }
.hero-badge {
    display: inline-block;
    background: #eef2ff;
    color: #4f46e5;
    font-size: 13px;
    font-weight: 600;
    padding: 5px 16px;
    border-radius: 20px;
    margin-bottom: 16px;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4f46e5, #7c3aed, #0891b2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}
.hero p {
    color: #6b7280;
    font-size: 1.1rem;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}
.card {
    background: white;
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.08);
    border: 1px solid #e0e7ff;
    margin-bottom: 1.2rem;
}
.section-title {
    font-size: 13px;
    font-weight: 600;
    color: #4f46e5;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 1rem;
}
.answer-box {
    background: #f8f7ff;
    border-left: 4px solid #6366f1;
    border-radius: 0 12px 12px 0;
    padding: 1.4rem 1.6rem;
    margin-top: 0.8rem;
}
.q-chip {
    background: #eef2ff;
    color: #4f46e5;
    font-size: 14px;
    font-weight: 500;
    padding: 7px 16px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 1rem;
}
.answer-text { color: #1f2937; font-size: 16px; line-height: 1.8; }
.success-pill {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 12px 18px;
    color: #166534;
    font-size: 14px;
    font-weight: 500;
}
.stTextInput > div > div > input {
    background: #f8faff !important;
    border: 2px solid #e0e7ff !important;
    border-radius: 12px !important;
    color: #1f2937 !important;
    font-size: 16px !important;
    padding: 12px 18px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    height: 50px !important;
    width: 100% !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
}
.stButton > button:hover { opacity: 0.9 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-badge">✨ AI Powered</div>
    <h1>PDF Insight</h1>
    <p>Upload any PDF and instantly get answers to your questions using AI</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
    return model, groq_client


model, groq_client = load_models()

st.markdown('<div class="card"><div class="section-title">📄 Upload your PDF</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    with st.spinner("Reading your PDF..."):
        text = extract_text(uploaded_file)
        chunks = create_chunks(text)

        if not chunks:
            st.error("❌ Could not extract text from this PDF. Try a different file.")
            st.stop()

        embeddings = model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        st.session_state.embeddings = embeddings
        st.session_state.chunks = chunks

    st.markdown(f'<div class="success-pill">✅ <b>{uploaded_file.name}</b> is ready — ask your questions below!</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="card"><div class="section-title">💬 Ask a question</div>', unsafe_allow_html=True)
    question = st.text_input("", placeholder="e.g. What are the key skills mentioned?", label_visibility="collapsed")
    ask = st.button("Ask AI →")
    st.markdown('</div>', unsafe_allow_html=True)

    if ask and question.strip():
        query_embedding = model.encode([question]).astype("float32")

        dot_scores = np.dot(st.session_state.embeddings, query_embedding[0])
        top_indices = np.argsort(dot_scores)[::-1][:3]
        top_chunks = [st.session_state.chunks[i] for i in top_indices]
        context = "\n\n---\n\n".join(top_chunks)

        prompt = f"""You are a helpful assistant. Answer ONLY using the context below.
If the answer is not in the context, reply exactly: "I couldn't find that in the uploaded PDF."

Context:
{context}

Question: {question}"""

        with st.spinner("Thinking..."):
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )
            answer = response.choices[0].message.content

        st.markdown(f"""
        <div class="card">
            <div class="section-title">💡 Answer</div>
            <div class="q-chip">Q: {question}</div>
            <div class="answer-box">
                <div class="answer-text">{answer}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center; color:#9ca3af; padding: 1rem 0; font-size: 14px;">
        👆 Start by uploading a PDF above
    </div>
    """, unsafe_allow_html=True)