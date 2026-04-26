import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from sentence_transformers import CrossEncoder

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Q&A",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RAG Q&A with Reranking")
st.caption("Retrieval-Augmented Generation using HuggingFace + FAISS + CrossEncoder")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    url = st.text_input(
        "Source URL",
        value="https://en.wikipedia.org/wiki/Paracetamol",
        help="Web page to load as the knowledge source",
    )

    chunk_size = st.slider("Chunk size", 200, 1000, 500, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 200, 100, step=10)
    top_k_retrieve = st.slider("Docs to retrieve (k)", 5, 30, 20)
    top_k_rerank = st.slider("Docs after reranking", 1, 10, 5)

    st.divider()
    st.markdown("**Models**")
    st.code("LLM: openai/gpt-oss-20b\nEmbedding: all-MiniLM-L6-v2\nReranker: ms-marco-MiniLM-L-6-v2", language="text")

# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading LLM…")
def load_llm():
    return ChatHuggingFace(
        llm=HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b")
    )

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Loading reranker…")
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource(show_spinner="Loading & indexing documents…")
def build_vectorstore(_embeddings, url: str, chunk_size: int, chunk_overlap: int):
    loader = WebBaseLoader(url)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, _embeddings)
    return vectorstore, len(split_docs)

# ── Load resources ────────────────────────────────────────────────────────────
llm        = load_llm()
embeddings = load_embeddings()
reranker   = load_reranker()

with st.spinner("Building vector index…"):
    vectorstore, n_chunks = build_vectorstore(
        embeddings, url, chunk_size, chunk_overlap
    )

st.success(f"✅ Indexed **{n_chunks} chunks** from `{url}`", icon="📄")

# ── Query interface ───────────────────────────────────────────────────────────
st.divider()
query = st.text_input(
    "💬 Ask a question",
    placeholder="What are the key side effects mentioned?",
)

if st.button("Ask", type="primary", disabled=not query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k_retrieve})

    with st.status("Thinking…", expanded=True) as status:
        # Step 1 – Retrieve
        st.write(f"🔎 Retrieving top **{top_k_retrieve}** chunks…")
        retrieved_docs = retriever.invoke(query)

        # Step 2 – Rerank
        st.write(f"⚡ Reranking → keeping top **{top_k_rerank}**…")
        pairs  = [(query, doc.page_content) for doc in retrieved_docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in ranked[:top_k_rerank]]

        # Step 3 – Generate
        st.write("🤖 Generating answer…")
        context = "\n\n".join([doc.page_content for doc in top_docs])
        prompt  = f"""Answer the question using ONLY the context below.

Context: {context}

Question: {query}"""

        response = llm.invoke(prompt)
        status.update(label="Done!", state="complete", expanded=False)

    # ── Answer ────────────────────────────────────────────────────────────────
    st.subheader("📝 Answer")
    st.markdown(response.content)

    # ── Source chunks ─────────────────────────────────────────────────────────
    with st.expander("📚 Source chunks used"):
        for i, (doc, score) in enumerate(ranked[:top_k_rerank], 1):
            st.markdown(f"**Chunk {i}** — rerank score: `{score:.4f}`")
            st.caption(doc.page_content)
            st.divider()