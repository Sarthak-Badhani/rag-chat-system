# Chat with Your Documents

A production-quality **Retrieval-Augmented Generation (RAG)** system that allows users to upload PDF documents and ask questions about them. 

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

##  Project Overview

This project implements a complete RAG pipeline for document question-answering. Users can:
- Upload PDF documents
- Ask natural language questions
- Receive accurate answers with source citations
- Evaluate retrieval quality

**Key Feature:** Answers are generated using *only* the content from uploaded documents, minimizing LLM hallucination.

---

##  What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture that enhances Large Language Model (LLM) responses by grounding them in external knowledge sources.

### Traditional LLM Approach
```
User Question → LLM → Answer (may hallucinate)
```

### RAG Approach
```
User Question → Retrieve Relevant Documents → LLM + Context → Grounded Answer
```

### Benefits of RAG
- **Reduced Hallucination**: Answers based on actual documents
- **Up-to-date Information**: No need to retrain the model
- **Source Attribution**: Citations provide transparency
- **Domain Specificity**: Works with specialized documents

---

##  Architecture

The system follows a two-phase architecture:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           INDEXING PHASE                                │
│                     (One-time document processing)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────────┐   │
│   │   PDF   │───▶│  Extract │───▶│  Chunk   │───▶│    Generate     │   │
│   │ Upload  │    │   Text   │    │   Text   │    │   Embeddings    │   │
│   └─────────┘    └──────────┘    └──────────┘    └────────┬────────┘   │
│                                                           │             │
│                                                           ▼             │
│                                               ┌─────────────────────┐   │
│                                               │   FAISS Vector DB   │   │
│                                               │   (Persisted)       │   │
│                                               └─────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                            QUERY PHASE                                  │
│                    (Real-time question answering)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌─────────────────┐   │
│   │  User   │───▶│  Embed   |───▶│ Retrieve │───▶│     Build       │   │
│   │ Question│    │  Query   │    │  Top-K   │    │    Prompt       │   │
│   └─────────┘    └──────────┘    └──────────┘    └────────┬────────┘   │
│                                                           │             │
│                                                           ▼             │
│   ┌─────────┐    ┌──────────────────────────────────────────────────┐   │
│   │ Answer  │◀───│              LLM (GPT-3.5/4)                     │   │
│   │ + Cites │    │   "Answer using ONLY the provided context..."   │   │
│   └─────────┘    └──────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Orchestration** | LangChain | RAG pipeline management |
| **Vector Store** | FAISS | Similarity search |
| **Embeddings** | OpenAI / HuggingFace | Text vectorization |
| **LLM** | GPT-3.5/4 | Answer generation |
| **UI** | Streamlit | Web interface |
| **PDF Processing** | PyPDF | Document loading |
| **Configuration** | python-dotenv | Environment management |

---

## 📂 Project Structure

```
rag-chat-system/
│
├── app.py                    # Streamlit web application
├── config.py                 # Centralized configuration
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
├── README.md                # This file
│
├── rag/                     # Core RAG modules
│   ├── __init__.py          # Package exports
│   ├── loader.py            # PDF loading & text extraction
│   ├── chunker.py           # Text splitting with overlap
│   ├── embeddings.py        # Embedding generation & caching
│   ├── vector_store.py      # FAISS index management
│   ├── retriever.py         # Similarity search
│   ├── prompt_builder.py    # Prompt engineering
│   └── generator.py         # LLM answer generation
│
└── data/                    # Data directory (auto-created)
    ├── index/               # Persisted FAISS indexes
    └── cache/               # Embedding cache
```

---

##  Quick Start

### Prerequisites
- Python 3.9 or higher
- OpenAI API key (for OpenAI models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sarthak-Badhani/rag-chat-system.git
   cd rag-chat-system
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   .\venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   # Copy example environment file
   cp .env.example .env
   
   # Edit .env and add your API key
   # OPENAI_API_KEY=sk-your-key-here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**
   Navigate to `http://localhost:8501`

---

## 📖 Usage Guide

### 1. Upload Documents
- Click the **Upload** tab
- Drag and drop PDF files or click to browse
- Click **Process Documents** to index

### 2. Ask Questions
- Switch to the **Chat** tab
- Type your question in the input box
- Click **Ask** to get an answer

### 3. Review Sources
- Expand **Sources** to see cited documents
- Expand **Retrieved Chunks** to see the context used

### 4. Evaluate Quality
- Use the **Evaluate** tab for testing
- Enter test questions to assess retrieval

---

## ⚙️ Configuration

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `CHUNK_SIZE` | 800 | Characters per chunk |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `RETRIEVER_TOP_K` | 3 | Chunks to retrieve |
| `LLM_TEMPERATURE` | 0 | Generation randomness |
| `EMBEDDING_PROVIDER` | huggingface | Embedding source |

### How Chunk Size Affects Quality

| Chunk Size | Pros | Cons |
|------------|------|------|
| Small (200-400) | Precise retrieval | May lose context |
| Medium (600-800) | Balanced | Good default |
| Large (1000+) | More context | May include noise |

**Recommendation:** Start with 800 characters and adjust based on your documents.

---

## 🧪 Evaluation

The evaluation section helps assess retrieval quality:

1. **Enter test questions** relevant to your documents
2. **Run evaluation** to see retrieved chunks and scores
3. **Analyze results** to tune chunk size and top-k

### Quality Metrics to Consider
- **Relevance**: Are retrieved chunks related to the question?
- **Coverage**: Does the context contain the answer?
- **Precision**: Is there irrelevant information?

---

## 🔧 Module Documentation

### loader.py - PDF Loading
Handles PDF text extraction with page-level metadata.

```python
from rag import PDFLoader

loader = PDFLoader()
documents = loader.load("document.pdf")

for doc in documents:
    print(f"Page {doc.metadata.page_number}: {doc.content[:100]}...")
```

### chunker.py - Text Splitting
Splits documents into overlapping chunks for embedding.

```python
from rag import TextChunker

chunker = TextChunker(chunk_size=800, chunk_overlap=150)
chunks = chunker.chunk_documents(documents)
```

### embeddings.py - Embedding Generation
Generates vector embeddings with optional caching.

```python
from rag import EmbeddingManager

# HuggingFace (local, no API key)
manager = EmbeddingManager(provider="huggingface")

# OpenAI (requires API key)
manager = EmbeddingManager(provider="openai", api_key="sk-...")

embedding = manager.embed_text("Hello world")
```

### vector_store.py - FAISS Management
Manages vector index creation, persistence, and search.

```python
from rag import VectorStoreManager

store = VectorStoreManager(embedding_manager)
store.create_index(documents)
store.save()  # Persist to disk

# Later...
store.load()  # Reload from disk
results = store.similarity_search("query", k=3)
```

### retriever.py - Document Retrieval
High-level retrieval interface with metadata.

```python
from rag import DocumentRetriever

retriever = DocumentRetriever(vector_store, top_k=5)
result = retriever.retrieve("What is machine learning?")

for chunk in result.chunks:
    print(f"{chunk.to_citation()}: {chunk.content[:100]}")
```

### prompt_builder.py - Prompt Engineering
Constructs prompts that minimize hallucination.

```python
from rag import PromptBuilder

builder = PromptBuilder()
prompt = builder.build(retrieval_result, "What is RAG?")
```

### generator.py - Answer Generation
Generates answers using the LLM.

```python
from rag import AnswerGenerator

generator = AnswerGenerator(provider="openai", api_key="sk-...")
answer = generator.generate(prompt)

print(answer.answer)
print(answer.get_formatted_sources())
```

---

## 🚀 Future Improvements

- [ ] **Multi-format support**: Word, PowerPoint, HTML
- [ ] **Hybrid search**: Combine semantic + keyword search
- [ ] **Chat memory**: Maintain conversation context
- [ ] **Streaming responses**: Real-time answer generation
- [ ] **Authentication**: User accounts and document isolation
- [ ] **Analytics**: Track usage and improve retrieval
- [ ] **API endpoint**: REST API for integration
- [ ] **Docker deployment**: Containerized deployment

---


---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [OpenAI](https://openai.com/) for embedding and LLM APIs
- [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- [Streamlit](https://streamlit.io/) for the web interface

---

