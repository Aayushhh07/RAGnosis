# Quick Start Guide

## Prerequisites

1. Python 3.12 installed
2. Virtual environment activated
3. Ollama installed and running

## Installation Steps

### 1. Install Dependencies

```bash
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
```

### 2. Install and Start Ollama

**Download Ollama:**
- Visit https://ollama.ai and download for your OS
- Install it

**Start Ollama Server:**
```bash
ollama serve
```

**Pull a Model (in a new terminal):**
```bash
ollama pull llama3
# OR
ollama pull mistral
```

### 3. Add Documents

Place your documents in the `documents/` folder:
- PDF files (.pdf)
- Text files (.txt)
- Markdown files (.md, .markdown)

Sample documents are already included for testing!

### 4. Run the Application

**Option A: Streamlit UI (Recommended)**
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

**Option B: Command Line Interface**
```bash
# First, setup documents
python main.py setup

# Then query
python main.py query "What is machine learning?"
```

## First Run

1. Start Ollama: `ollama serve`
2. Pull model: `ollama pull llama3`
3. Run Streamlit: `streamlit run app.py`
4. In the UI sidebar, click "Load/Reload Documents"
5. Enter a question and click "Search"

## Example Questions

Try these with the sample documents:

- "What is machine learning?"
- "Explain the RAG pipeline"
- "What are vector databases?"
- "What types of embedding models exist?"

## Troubleshooting

**"Ollama connection error"**
- Make sure Ollama is running: `ollama serve`
- Check if model is installed: `ollama list`

**"No documents found"**
- Ensure files are in `documents/` folder
- Check file extensions (.pdf, .txt, .md)

**"Model not found"**
- Pull the model: `ollama pull llama3`

**Slow first run**
- First run downloads the embedding model (~400MB)
- Subsequent runs are faster

## Next Steps

- Add your own documents to `documents/` folder
- Experiment with different models (Mistral, Phi, Gemma)
- Adjust confidence threshold in the UI
- Try different chunk sizes for your document types
