# RAG Application - Knowledge Assistant

A complete Retrieval-Augmented Generation (RAG) application built with open-source tools that answers user questions strictly based on provided documents.

## Features

- **Document Ingestion**: Supports PDF, TXT, and Markdown files
- **Open-Source Embeddings**: Uses Sentence Transformers (BGE models)
- **Open-Source LLM**: Uses Ollama with Llama 3 or Mistral models
- **Vector Database**: FAISS for efficient similarity search
- **Complete RAG Pipeline**: Query embedding → Vector search → Context retrieval → Response generation
- **Guardrails**: Hallucination prevention, confidence thresholding, context-only answers
- **Source References**: All answers include document sources
- **Streamlit UI**: User-friendly web interface

## Architecture

```
User Query → Query Embedding → Vector Search → Context Retrieval → 
Prompt Augmentation → LLM Generation → Guardrails → Response with Sources
```

## Quick Start

See [QUICK_START.md](QUICK_START.md) for a quick setup guide.

For detailed step-by-step instructions, see [STEP_BY_STEP_GUIDE.md](STEP_BY_STEP_GUIDE.md).

**New to RAG or Python?** Check out [COMPLETE_BEGINNER_GUIDE.md](COMPLETE_BEGINNER_GUIDE.md) - explains every line of code, all concepts, and tech stack choices in beginner-friendly detail!

## Installation

1. Create a virtual environment (if not already created):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -e .
```

3. Install and start Ollama (for LLM):
   - Download from https://ollama.ai
   - Start server: `ollama serve`
   - Pull a model: `ollama pull llama3` or `ollama pull mistral`

## Usage

### Streamlit UI (Recommended)

1. Place your documents in the `documents/` folder (PDF, TXT, or Markdown)
   - Sample documents are included for testing!

2. Run the Streamlit app:
```bash
streamlit run app.py
```

3. Access the UI at http://localhost:8501
4. Click "Load/Reload Documents" in the sidebar
5. Enter your question and click "Search"

### Command Line Interface

```bash
# Setup documents and build vector store
python main.py setup

# Query the system
python main.py query "What is machine learning?"
```

## Project Structure

```
RAG/
├── app.py                 # Streamlit UI
├── rag_pipeline.py        # Core RAG pipeline
├── document_processor.py  # Document ingestion and chunking
├── embeddings.py          # Embedding generation
├── vector_store.py        # FAISS vector database
├── guardrails.py          # Safety controls and filtering
├── documents/             # Input documents folder
├── vector_store/          # Saved FAISS index
└── requirements.txt       # Dependencies
```

## Model Choices

- **Embeddings**: `BAAI/bge-base-en-v1.5` - High-quality open-source embeddings
- **LLM**: Ollama with Llama 3 or Mistral - Open-source, locally runnable models

## License

MIT
