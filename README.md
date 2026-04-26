# 🔍 RAG Q&A with Reranking

A production-ready **Retrieval-Augmented Generation (RAG)** system built with LangChain, FAISS, HuggingFace, and Streamlit. It loads any webpage as a knowledge source, indexes it into a vector store, and answers questions using a reranking pipeline to ensure high-quality, grounded responses.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Setup & Installation](#-setup--installation)
- [Running the App](#-running-the-app)
- [How It Works](#-how-it-works)
- [Configuration](#-configuration)
- [Environment Variables](#-environment-variables)
- [Troubleshooting](#-troubleshooting)

---

## 🧠 Overview

Standard RAG systems retrieve document chunks based on **embedding similarity**, which often returns chunks that are semantically close but not truly relevant to the query. This project solves that problem by adding a **CrossEncoder reranker** step after retrieval — dramatically improving answer quality and reducing hallucinations.

The app lets you:
- Point at **any public webpage** as your knowledge source
- Ask natural language questions
- Get answers grounded **strictly** in that content

---

## ✨ Features

- 🌐 Load any public webpage as a knowledge base
- ✂️ Configurable chunking (size & overlap)
- ⚡ Fast vector similarity search using FAISS
- 🔁 Two-stage retrieval with CrossEncoder reranking
- 🤖 LLM-powered answer generation via HuggingFace
- 🖥️ Interactive Streamlit UI with live configuration
- 📦 Dockerized and fully reproducible

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────┐
│   Web Page Loader   │  ← Load any URL
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Text Splitter     │  ← Chunk into segments (configurable size & overlap)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  HuggingFace        │  ← Embed chunks (all-MiniLM-L6-v2)
│  Embeddings         │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   FAISS VectorStore │  ← Index & retrieve top-k candidates
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  CrossEncoder       │  ← Rerank candidates by true relevance
│  Reranker           │     (ms-marco-MiniLM-L-6-v2)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  HuggingFace LLM    │  ← Generate answer from top-N chunks
│  (gpt-oss-20b)      │
└────────┬────────────┘
         │
         ▼
      Answer
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Framework | [LangChain](https://www.langchain.com/) |
| LLM | HuggingFace — `openai/gpt-oss-20b` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | [FAISS](https://github.com/facebookresearch/faiss) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| UI | [Streamlit](https://streamlit.io/) |
| Document Loader | LangChain `WebBaseLoader` |
| Environment | Python 3.12+, Docker |

---

## 📁 Project Structure

```
rag-docker/
│
├── rag_app.py          # Main Streamlit application (UI + pipeline)
├── Reranker_web.py     # Standalone script version (no UI)
├── main.py             # Entry point placeholder
│
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project metadata & dependencies (uv)
├── uv.lock             # Locked dependency versions
│
├── .env                # Environment variables (HF_TOKEN etc.) — not committed
├── .gitignore          # Git ignore rules
├── .python-version     # Python version pin
│
└── README.md           # This file
```

---

## ✅ Prerequisites

Make sure you have the following installed:

- **Python 3.12+** — [Download](https://www.python.org/downloads/)
- **pip** or **uv** (recommended) — [uv install guide](https://github.com/astral-sh/uv)
- **Git** — [Download](https://git-scm.com/)
- A **HuggingFace account** with an API token — [Get yours here](https://huggingface.co/settings/tokens)

> Optional: **Docker** if you want to run the containerized version.

---

## ⚙️ Setup & Installation

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/rag-docker.git
cd rag-docker
```

### Step 2 — Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env   # if an example exists
# OR create it manually:
touch .env
```

Open `.env` and add your HuggingFace token:

```env
HF_TOKEN=hf_your_token_here
```

> ⚠️ Never commit your `.env` file. It's already in `.gitignore`.

### Step 3 — Create a Virtual Environment

**Using `uv` (recommended):**

```bash
uv venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

**Using standard `venv`:**

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### Step 4 — Install Dependencies

**Using `uv`:**

```bash
uv pip install -r requirements.txt
```

**Using `pip`:**

```bash
pip install -r requirements.txt
```

> ⏳ First-time install may take a few minutes — it downloads PyTorch, Transformers, and FAISS.

---

## ▶️ Running the App

### Option 1 — Streamlit Web App (Recommended)

```bash
streamlit run rag_app.py
```

Then open your browser at: **http://localhost:8501**

### Option 2 — Standalone Script (No UI)

```bash
python Reranker_web.py
```

This runs the full RAG pipeline in the terminal with a hardcoded query against the Paracetamol Wikipedia page.

---

## 🔄 How It Works

The pipeline follows 5 key steps:

**1. Load** — `WebBaseLoader` fetches and parses any public webpage.

**2. Chunk** — `RecursiveCharacterTextSplitter` breaks the content into overlapping segments for better context preservation.

**3. Embed & Index** — Each chunk is converted to a dense vector using `all-MiniLM-L6-v2` and stored in a FAISS index for fast retrieval.

**4. Retrieve & Rerank** — On a query:
   - FAISS retrieves the top-k most similar chunks (e.g., top 20)
   - The CrossEncoder reranker scores each `(query, chunk)` pair together, capturing true relevance beyond surface similarity
   - Only the top-N reranked chunks are passed forward (e.g., top 5)

**5. Generate** — The LLM receives a prompt containing only the top-N chunks and is instructed to answer using that context alone — minimizing hallucinations.

---

## 🎛️ Configuration

All parameters are adjustable via the **Streamlit sidebar**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Source URL | Wikipedia/Paracetamol | Any public webpage |
| Chunk size | 500 | Characters per chunk |
| Chunk overlap | 100 | Overlap between chunks |
| Docs to retrieve (k) | 20 | Candidates fetched from FAISS |
| Docs after reranking | 5 | Final chunks sent to LLM |

> 💡 Tip: A higher `k` with a lower rerank count gives better coverage while keeping the LLM prompt focused.

---

## 🔐 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | ✅ Yes | HuggingFace API token for LLM & model access |

---

## 🐛 Troubleshooting

**`ModuleNotFoundError`** — Make sure your virtual environment is activated and dependencies are installed.

**`HF_TOKEN not found`** — Ensure your `.env` file exists in the root directory and contains a valid token.

**Slow first run** — The embedding and reranker models are downloaded on first use (~100–300 MB). Subsequent runs use the cache.

**`FAISS` install issues on Windows** — Try `pip install faiss-cpu` explicitly, or use WSL2.

**Streamlit port conflict** — Run on a different port: `streamlit run rag_app.py --server.port 8502`

---

## 🙌 Acknowledgements

- [LangChain](https://www.langchain.com/) for the orchestration framework
- [HuggingFace](https://huggingface.co/) for open-source models
- [Facebook AI Research](https://github.com/facebookresearch/faiss) for FAISS
- [Streamlit](https://streamlit.io/) for the rapid UI framework
