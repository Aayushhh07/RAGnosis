# Complete Step-by-Step Guide: Building a RAG Application

## Overview
This guide will walk you through building a complete Retrieval-Augmented Generation (RAG) application using open-source tools. The system answers questions strictly based on provided documents.

---

## Step 0: Project Setup and Environment

### 0.1 Verify Python Environment
```bash
python --version  # Should be Python 3.12
```

### 0.2 Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 0.3 Install Dependencies
```bash
pip install -e .
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

**What gets installed:**
- `sentence-transformers`: For generating text embeddings (BGE models)
- `faiss-cpu`: Vector database for similarity search
- `streamlit`: Web UI framework
- `pypdf2`: PDF document processing
- `python-docx`: Word document support (optional)
- `markdown`: Markdown parsing
- `ollama`: Python client for Ollama LLM
- `numpy`: Numerical operations
- `tiktoken`: Token counting utilities

### 0.4 Install and Setup Ollama (for LLM)

1. **Download Ollama:**
   - Visit https://ollama.ai
   - Download and install for your OS

2. **Start Ollama:**
   ```bash
   ollama serve
   ```

3. **Pull a model:**
   ```bash
   ollama pull llama3
   # OR
   ollama pull mistral
   ```

**Why Ollama?** It allows running open-source LLMs locally without API keys.

---

## Step 1: Document Ingestion

### 1.1 Create Documents Directory
```bash
mkdir documents
```

### 1.2 Add Your Documents
Place 5-20 documents in the `documents/` folder. Supported formats:
- **PDF** (.pdf)
- **Text files** (.txt)
- **Markdown** (.md, .markdown)

### 1.3 How Document Ingestion Works

**File: `document_processor.py`**

The `DocumentProcessor` class handles:
- **PDF extraction**: Uses PyPDF2 to extract text from PDF pages
- **Text loading**: Direct UTF-8 text file reading
- **Markdown parsing**: Converts Markdown to plain text

**Key Methods:**
- `load_document()`: Automatically detects file type and loads content
- `process_directory()`: Processes all supported files in a directory
- `process_files()`: Processes specific file paths

---

## Step 2: Document Chunking

### 2.1 Why Chunking?
- LLMs have token limits
- Smaller chunks improve retrieval precision
- Overlapping chunks preserve context

### 2.2 Chunking Strategy

**Parameters:**
- `chunk_size`: 512 characters (default)
- `chunk_overlap`: 50 characters (default)

**Algorithm:**
1. Split text into chunks of `chunk_size` characters
2. Try to break at sentence boundaries (`.`, `!`, `?`)
3. If no sentence boundary, break at word boundaries
4. Overlap chunks by `chunk_overlap` characters

**Implementation:** See `chunk_text()` method in `document_processor.py`

### 2.3 Metadata Preservation
Each chunk stores:
- Original text
- Source document name
- File path
- Chunk index
- Character positions

---

## Step 3: Embedding Generation

### 3.1 Model Selection

**File: `embeddings.py`**

We use **BAAI/bge-base-en-v1.5** (BGE - BAAI General Embedding):
- High-quality open-source embeddings
- 768-dimensional vectors
- Optimized for retrieval tasks
- Supports instruction-based query encoding

**Alternatives:**
- `BAAI/bge-small-en-v1.5`: Faster, smaller (384 dimensions)
- `sentence-transformers/all-MiniLM-L6-v2`: Lightweight option

### 3.2 How Embeddings Work

1. **Document Embeddings:**
   - Each chunk is converted to a dense vector
   - Vectors capture semantic meaning
   - Normalized for cosine similarity

2. **Query Embeddings:**
   - Queries use instruction-based encoding
   - Format: "Represent this sentence for searching relevant passages: {query}"
   - Improves query-document matching

**Implementation:** See `EmbeddingGenerator` class in `embeddings.py`

---

## Step 4: Vector Database (FAISS)

### 4.1 Why FAISS?
- **Fast**: Optimized C++ backend
- **Open-source**: Facebook AI Research
- **Scalable**: Handles millions of vectors
- **No server needed**: Runs locally

### 4.2 Index Types

**File: `vector_store.py`**

- **Cosine Similarity** (default): Best for normalized embeddings
- **L2 Distance**: Euclidean distance
- **Flat Index**: Exact search (accurate but slower)

### 4.3 How It Works

1. **Storage:**
   - Vectors stored in FAISS index
   - Metadata stored separately (Python pickle)

2. **Search:**
   - Query vector compared to all stored vectors
   - Returns top-K most similar vectors
   - Returns similarity scores

3. **Persistence:**
   - Index saved to disk (`vector_store/faiss.index`)
   - Metadata saved as pickle (`vector_store/metadata.pkl`)
   - Can be reloaded without reprocessing documents

**Implementation:** See `VectorStore` class in `vector_store.py`

---

## Step 5: Complete RAG Pipeline

### 5.1 Pipeline Flow

**File: `rag_pipeline.py`**

```
User Query
    ↓
Query Embedding (Step 3)
    ↓
Vector Similarity Search (Step 4)
    ↓
Context Retrieval (Top-K chunks)
    ↓
Prompt Augmentation (Add contexts to prompt)
    ↓
LLM Generation (Ollama)
    ↓
Guardrails Validation (Step 6)
    ↓
Response with Sources
```

### 5.2 Prompt Engineering

**System Instructions:**
```
You are a helpful knowledge assistant that answers questions 
STRICTLY based on the provided documents.

CRITICAL RULES:
1. Answer ONLY using information from the provided context documents
2. If the answer is not in the provided documents, explicitly state 
   "Based on the provided documents, I cannot find information about [topic]"
3. Do NOT make up information or use external knowledge
4. Cite the source document when referencing specific information
5. If multiple sources contain relevant information, synthesize them clearly
```

**Why This Works:**
- Explicit instructions prevent hallucinations
- Forces model to acknowledge when information is missing
- Encourages source citation

**Implementation:** See `build_prompt()` method in `rag_pipeline.py`

---

## Step 6: Guardrails and Safety Controls

### 6.1 Confidence Thresholding

**File: `guardrails.py`**

- **Minimum Confidence**: Default 0.5 (0-1 scale)
- Filters out low-similarity results
- Prevents answering from irrelevant documents

**How it works:**
```python
if max_similarity_score < min_confidence:
    return "I couldn't find relevant documents..."
```

### 6.2 Context-Only Enforcement

**Methods:**
1. **Prompt Engineering**: System instructions enforce context-only answers
2. **Response Validation**: Checks for refusal phrases
3. **Source Verification**: Ensures sources are cited

### 6.3 Output Filtering

**Hallucination Detection:**
- Detects phrases like "I don't know", "not in the provided"
- Validates these are appropriate (not false refusals)
- Adds warnings if confidence is low

**Implementation:** See `Guardrails` class in `guardrails.py`

---

## Step 7: User Interface (Streamlit)

### 7.1 Features

**File: `app.py`**

- **Document Management**: Load/reload documents
- **Query Interface**: Ask questions
- **Response Display**: Shows answer with sources
- **Metadata View**: Confidence scores, validation results
- **Configuration**: Adjust models, thresholds, top-K

### 7.2 Running the UI

```bash
streamlit run app.py
```

Access at: http://localhost:8501

### 7.3 UI Components

1. **Sidebar:**
   - Model selection
   - Confidence threshold slider
   - Top-K slider
   - Document loading button
   - System statistics

2. **Main Area:**
   - Query input
   - Answer display
   - Source citations
   - Retrieval details (expandable)

---

## Step 8: Testing and Usage

### 8.1 Setup Documents

1. Create `documents/` folder
2. Add 5-20 documents (PDF, TXT, or Markdown)
3. Run setup:
   ```bash
   python main.py setup
   ```

### 8.2 Using CLI

```bash
python main.py query "What is the main topic?"
```

### 8.3 Using Streamlit UI

1. Start UI: `streamlit run app.py`
2. Click "Load/Reload Documents" in sidebar
3. Enter query and click "Search"
4. View answer with sources

---

## Architecture Summary

```
┌─────────────────┐
│  Documents      │ (PDF, TXT, MD)
│  (documents/)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Document        │ Chunking (512 chars, 50 overlap)
│ Processor       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Embedding       │ BGE Model (768-dim)
│ Generator       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vector Store    │ FAISS Index
│ (FAISS)         │
└────────┬────────┘
         │
         │ Query
         │
         ▼
┌─────────────────┐
│ RAG Pipeline    │ Retrieve → Augment → Generate
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Guardrails      │ Confidence + Validation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Response        │ Answer + Sources
└─────────────────┘
```

---

## Model Choices Explained

### Embeddings: BGE (BAAI/bge-base-en-v1.5)
- **Why**: State-of-the-art open-source embeddings
- **Performance**: High retrieval accuracy
- **Size**: 768 dimensions (good balance)
- **License**: Apache 2.0 (open-source)

### LLM: Ollama (Llama 3 / Mistral)
- **Why**: Fully open-source, runs locally
- **Privacy**: No data sent to external APIs
- **Cost**: Free (no API costs)
- **Performance**: Good quality for RAG tasks

### Vector DB: FAISS
- **Why**: Fast, scalable, no server needed
- **Performance**: Handles millions of vectors efficiently
- **Simplicity**: Easy to integrate and persist

---

## Troubleshooting

### Issue: "Ollama connection error"
**Solution**: Ensure Ollama is running: `ollama serve`

### Issue: "Model not found"
**Solution**: Pull the model: `ollama pull llama3`

### Issue: "No documents found"
**Solution**: Check `documents/` folder has PDF, TXT, or MD files

### Issue: "Low confidence warnings"
**Solution**: Lower confidence threshold or add more relevant documents

### Issue: "Memory errors"
**Solution**: Use smaller embedding model (`bge-small`) or reduce chunk size

---

## Next Steps

1. **Add more documents** to improve knowledge base
2. **Experiment with chunk sizes** for your document types
3. **Try different models** (Mistral, Phi, Gemma)
4. **Adjust confidence thresholds** based on your use case
5. **Add more guardrails** (e.g., NLI models for validation)

---

## Project Structure

```
RAG/
├── app.py                    # Streamlit UI
├── main.py                   # CLI entry point
├── rag_pipeline.py           # Core RAG pipeline
├── document_processor.py     # Document ingestion & chunking
├── embeddings.py             # Embedding generation
├── vector_store.py           # FAISS vector database
├── guardrails.py             # Safety controls
├── documents/                # Input documents (you add these)
├── vector_store/             # Saved FAISS index (auto-generated)
├── requirements.txt          # Dependencies
├── pyproject.toml           # Project config
└── README.md                # Project overview
```

---

## Summary

You've built a complete RAG system with:
✅ Open-source embeddings (Sentence Transformers)
✅ Open-source LLM (Ollama)
✅ Vector database (FAISS)
✅ Document ingestion (PDF, TXT, MD)
✅ Chunking and embedding
✅ Complete RAG pipeline
✅ Guardrails and safety controls
✅ Source references
✅ User interface (Streamlit)

The system is production-ready and demonstrates all required components!
