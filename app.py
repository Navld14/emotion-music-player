import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import os
import io
import pickle

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Search Engine",
    page_icon="🔍",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.hero {
    text-align: center;
    padding: 2rem 0 0.5rem 0;
}

.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366f1, #a78bfa, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero p {
    color: #888;
    font-size: 1rem;
    margin-top: -8px;
}

.result-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 12px;
    padding: 8px;
    text-align: center;
}

.score-badge {
    background: #1e1e2e;
    color: #a78bfa;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
    display: inline-block;
    margin-top: 6px;
}

.section-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 8px;
}

.stat-box {
    background: #111;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}

.stat-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #a78bfa;
}

.stat-label {
    font-size: 0.75rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)


# ── Load CLIP model (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CLIP model...")
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


# ── Encode images ──────────────────────────────────────────────────────────────
def encode_images(uploaded_files, model, preprocess, device):
    embeddings = []
    names = []
    images = []

    for f in uploaded_files:
        try:
            img = Image.open(f).convert("RGB")
            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())
            names.append(f.name)
            images.append(img)
        except Exception:
            continue

    return np.vstack(embeddings) if embeddings else None, names, images


# ── Encode text query ──────────────────────────────────────────────────────────
def encode_text(query, model, device):
    tokens = clip.tokenize([query]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()


# ── Search ─────────────────────────────────────────────────────────────────────
def search(query_emb, image_embs, names, images, top_k=6):
    scores = (image_embs @ query_emb.T).squeeze()
    if scores.ndim == 0:
        scores = np.array([float(scores)])
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(names[i], images[i], float(scores[i])) for i in top_indices]


# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>🔍 Multimodal Search Engine</h1>
    <p>Search your images using natural language — powered by OpenAI CLIP</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

model, preprocess, device = load_model()

# ── Sidebar — upload images ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📁 Upload Images")
    st.caption("Upload any images (JPG, PNG, WEBP). The more you upload, the better the search!")

    uploaded_files = st.file_uploader(
        "Choose images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} image(s) loaded")
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        top_k = st.slider("Results to show", min_value=1, max_value=min(12, len(uploaded_files)), value=min(6, len(uploaded_files)))
        threshold = st.slider("Min similarity score", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        st.markdown("""
        - `a person smiling`
        - `outdoor landscape`
        - `food on a table`
        - `something red`
        - `a group of people`
        - `night scene with lights`
        """)


# ── Main area ──────────────────────────────────────────────────────────────────
if not uploaded_files:
    st.markdown("""
    <div style='text-align:center; padding:4rem 0; color:#333'>
        <div style='font-size:5rem'>🖼️</div>
        <div style='font-size:1.2rem; margin-top:1rem; color:#555'>Upload images in the sidebar to get started</div>
        <div style='color:#444; margin-top:0.5rem; font-size:0.9rem'>Then search them with plain English — no labels needed</div>
    </div>
    """, unsafe_allow_html=True)

else:
    # Encode images
    with st.spinner("🧠 Encoding images with CLIP..."):
        image_embs, names, images = encode_images(uploaded_files, model, preprocess, device)

    # Stats
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-value'>{len(uploaded_files)}</div>
            <div class='stat-label'>Images Loaded</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-value'>512</div>
            <div class='stat-label'>Embedding Dims</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='stat-box'>
            <div class='stat-value'>ViT-B/32</div>
            <div class='stat-label'>CLIP Model</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Search bar
    st.markdown("### 🔍 Search your images")
    query = st.text_input(
        label="",
        placeholder='Try: "a smiling person", "sunset over water", "something colorful"...',
        label_visibility="collapsed"
    )

    if query:
        with st.spinner("Searching..."):
            query_emb = encode_text(query, model, device)
            results = search(query_emb, image_embs, names, images, top_k=top_k)
            filtered = [(n, img, s) for n, img, s in results if s >= threshold]

        if not filtered:
            st.warning(f"No images matched above similarity threshold ({threshold}). Try lowering the threshold or a different query.")
        else:
            st.markdown(f"<div class='section-label'>Top {len(filtered)} results for: \"{query}\"</div>", unsafe_allow_html=True)

            cols = st.columns(3)
            for idx, (name, img, score) in enumerate(filtered):
                with cols[idx % 3]:
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.image(img, use_column_width=True)
                    score_pct = round(score * 100, 1)
                    bar_color = "#22c55e" if score_pct > 25 else "#f59e0b" if score_pct > 18 else "#ef4444"
                    st.markdown(f"""
                    <div style='padding:4px 6px'>
                        <div style='font-size:0.75rem; color:#888; overflow:hidden; text-overflow:ellipsis; white-space:nowrap'>{name}</div>
                        <div class='score-badge' style='background:#1e1e2e; color:{bar_color}'>Match: {score_pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

    else:
        # Show all uploaded images in a grid
        st.markdown("<div class='section-label'>All uploaded images</div>", unsafe_allow_html=True)
        cols = st.columns(4)
        for idx, (f, img) in enumerate(zip(uploaded_files, images)):
            with cols[idx % 4]:
                st.image(img, caption=f.name, use_column_width=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#333; font-size:0.8rem'>Built with Python · OpenAI CLIP · Streamlit · PyTorch</p>", unsafe_allow_html=True)
