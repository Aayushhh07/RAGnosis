# Complete Beginner's Guide to RAG Project
## From Zero to Pro - Every Line Explained

---

## Table of Contents

1. [What is RAG?](#what-is-rag)
2. [Project Overview](#project-overview)
3. [Folder Structure Explained](#folder-structure-explained)
4. [Tech Stack Deep Dive](#tech-stack-deep-dive)
5. [Key Terms Dictionary](#key-terms-dictionary)
6. [File-by-File Code Explanation](#file-by-file-code-explanation)
7. [How Everything Works Together](#how-everything-works-together)
8. [Common Questions](#common-questions)

---

## What is RAG?

**RAG = Retrieval-Augmented Generation**

Think of RAG like a smart research assistant:

1. **You ask a question** → "What is machine learning?"
2. **Assistant searches documents** → Finds relevant pages/chapters
3. **Assistant reads those sections** → Gets context
4. **Assistant answers** → Based ONLY on what it found

**Why RAG?**
- Regular AI chatbots can "hallucinate" (make up facts)
- RAG forces the AI to answer only from your documents
- You can update knowledge by adding new documents (no retraining needed!)

---

## Project Overview

### What This Project Does

You have documents (PDFs, text files, etc.). Users ask questions. The system:
1. Finds relevant parts of your documents
2. Uses those parts to generate accurate answers
3. Shows which documents were used (source citations)

### The Complete Flow

```
Documents → Chunk → Embed → Store in Vector DB
                                    ↓
User Query → Embed → Search Vector DB → Retrieve Top Chunks
                                    ↓
Retrieved Chunks → Build Prompt → LLM → Answer + Sources
```

---

## Folder Structure Explained

```
RAG/                              # Root folder (your project)
│
├── documents/                    # 📁 YOUR DOCUMENTS GO HERE
│   ├── sample1.txt               # Example: Machine Learning intro
│   ├── sample2.txt               # Example: NLP basics
│   ├── sample3.txt               # Example: Vector databases
│   ├── sample4.md                # Example: RAG overview
│   └── sample5.txt               # Example: Embedding models
│
├── vector_store/                 # 📁 AUTO-GENERATED (don't edit)
│   ├── faiss.index               # Vector database file
│   ├── metadata.pkl              # Document metadata
│   └── config.pkl                 # Configuration
│
├── document_processor.py         # 📄 Processes PDF/TXT/MD files
├── embeddings.py                 # 📄 Converts text to numbers
├── vector_store.py               # 📄 Manages the search database
├── guardrails.py                 # 📄 Safety checks
├── rag_pipeline.py               # 📄 Main RAG logic
├── app.py                        # 📄 Web UI (Streamlit)
├── main.py                       # 📄 Command-line interface
│
├── requirements.txt              # 📄 Python packages needed
├── pyproject.toml                # 📄 Project configuration
├── README.md                     # 📄 Project overview
├── QUICK_START.md                # 📄 Quick setup guide
├── STEP_BY_STEP_GUIDE.md         # 📄 Detailed implementation guide
└── COMPLETE_BEGINNER_GUIDE.md    # 📄 This file!
```

### What Each Folder Does

**`documents/`**
- Put your PDF, TXT, or Markdown files here
- The system reads everything in this folder
- You can add/remove files anytime

**`vector_store/`**
- Created automatically when you load documents
- Stores the "searchable version" of your documents
- Don't delete this if you want to keep your indexed documents!

**Python Files (.py)**
- Each file has a specific job (we'll explain each one)

---

## Tech Stack Deep Dive

### Why These Technologies?

#### 1. **Python 3.12**
**What:** Programming language
**Why:** 
- Easy to learn
- Great libraries for AI/ML
- Widely used in data science

#### 2. **Sentence Transformers**
**What:** Library for converting text to numbers (embeddings)
**Why:**
- Open-source (free!)
- Pre-trained models (no training needed)
- High quality embeddings
- Easy to use

**Alternative:** OpenAI embeddings (but costs money, not open-source)

#### 3. **FAISS (Facebook AI Similarity Search)**
**What:** Vector database for fast similarity search
**Why:**
- Very fast (millions of searches per second)
- Open-source
- No server needed (runs locally)
- Made by Facebook AI Research (reliable)

**Alternatives:** 
- Chroma (easier but slower)
- Qdrant (needs server)
- Pinecone (cloud, costs money)

#### 4. **Ollama**
**What:** Tool to run LLMs (Large Language Models) locally
**Why:**
- Completely free
- Runs on your computer (privacy!)
- No API keys needed
- Supports many models (Llama, Mistral, etc.)

**Alternatives:**
- OpenAI GPT (costs money, sends data to cloud)
- Hugging Face (more complex setup)
- Local LLM files (harder to manage)

#### 5. **Streamlit**
**What:** Framework for building web UIs in Python
**Why:**
- Super easy (write Python, get a website)
- No HTML/CSS/JavaScript needed
- Perfect for data science projects
- Free and open-source

**Alternatives:**
- Flask/FastAPI (more control, more complex)
- React/Vue (need to learn web development)

#### 6. **PyPDF2**
**What:** Library to read PDF files
**Why:**
- Simple API
- Handles most PDFs
- Lightweight

**Alternatives:**
- pdfplumber (better for complex PDFs)
- pymupdf (faster but more complex)

---

## Key Terms Dictionary

### Core Concepts

**RAG (Retrieval-Augmented Generation)**
- A technique where AI answers questions using retrieved documents
- Combines search + AI generation

**Embedding**
- Converting text into a list of numbers (vector)
- Example: "cat" → [0.2, -0.5, 0.8, ...]
- Similar words have similar numbers
- Used for semantic search

**Vector**
- A list of numbers representing something
- Example: [0.1, 0.5, -0.3, 0.9] is a 4-dimensional vector
- In our case, each text chunk becomes a 768-dimensional vector

**Chunking**
- Splitting long documents into smaller pieces
- Why? LLMs have token limits, and smaller chunks = better search precision
- Example: 10-page document → 20 chunks of ~500 characters each

**Chunk Overlap**
- When chunks share some text at boundaries
- Why? Prevents losing context at chunk boundaries
- Example: Chunk 1 ends at char 500, Chunk 2 starts at char 450 (50 char overlap)

**Vector Database**
- Database optimized for storing and searching vectors
- Like a search engine, but for numbers instead of keywords
- FAISS is our vector database

**Similarity Search**
- Finding vectors that are "close" to a query vector
- Uses math (cosine similarity, Euclidean distance)
- Returns most similar documents/chunks

**Cosine Similarity**
- Way to measure how similar two vectors are
- Range: -1 (opposite) to 1 (identical)
- 0 = unrelated
- We normalize to 0-1 scale

**Top-K**
- Retrieving the K most similar results
- Example: Top-5 means get the 5 best matches
- K is configurable (usually 3-10)

**Prompt**
- Text instructions given to an LLM
- Includes: system instructions + context + user question
- Well-crafted prompts = better answers

**Prompt Augmentation**
- Adding retrieved context to the prompt
- Example: "Answer this question using: [context from documents]"

**Hallucination**
- When AI makes up information not in the documents
- RAG prevents this by forcing answers from context only

**Guardrails**
- Safety checks to prevent errors
- Examples: confidence thresholding, hallucination detection

**Confidence Score**
- How sure the system is about a match (0-1)
- 0.9 = very confident, 0.3 = not confident
- We filter low-confidence results

**Metadata**
- Information about data
- Example: chunk metadata = {source: "file.pdf", chunk_index: 5, start_char: 1000}

**Session State**
- Streamlit's way to remember data between interactions
- Like variables that persist while the app runs

---

## File-by-File Code Explanation

### 1. document_processor.py

**Purpose:** Loads documents and splits them into chunks

#### Imports (Lines 1-12)

```python
import os                    # Operating system functions (file paths, etc.)
import re                     # Regular expressions (pattern matching in text)
from typing import List, Dict # Type hints (tells Python what types to expect)
from pathlib import Path      # Better way to handle file paths
import PyPDF2                 # Library to read PDF files
import markdown               # Library to convert Markdown to text
from docx import Document     # Library to read Word docs (not used but available)
```

**Why these imports?**
- `os` and `Path`: Handle file system operations
- `re`: Find sentence boundaries for smart chunking
- `PyPDF2`: Extract text from PDFs
- `markdown`: Convert `.md` files to plain text

#### Class Definition (Lines 15-30)

```python
class DocumentProcessor:
    """
    Processes documents from various formats (PDF, TXT, Markdown) 
    and splits them into chunks for embedding.
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size      # Store chunk size (default: 512 chars)
        self.chunk_overlap = chunk_overlap # Store overlap (default: 50 chars)
```

**What's happening:**
- `__init__`: Constructor (runs when you create a `DocumentProcessor` object)
- `self.chunk_size`: Instance variable (each object has its own)
- Default values: 512 chars per chunk, 50 char overlap
- These can be changed: `processor = DocumentProcessor(chunk_size=256, chunk_overlap=25)`

#### PDF Loading (Lines 32-42)

```python
def load_pdf(self, file_path: str) -> str:
    """Extract text from a PDF file."""
    text = ""                              # Start with empty string
    try:                                   # Try to do this, catch errors
        with open(file_path, 'rb') as file:  # Open file in binary mode ('rb')
            pdf_reader = PyPDF2.PdfReader(file)  # Create PDF reader object
            for page in pdf_reader.pages:       # Loop through each page
                text += page.extract_text() + "\n"  # Extract text, add newline
    except Exception as e:                # If error occurs
        raise Exception(f"Error reading PDF {file_path}: {str(e)}")  # Show error
    return text                            # Return extracted text
```

**Line-by-line:**
- `text = ""`: Initialize empty string to collect text
- `try/except`: Error handling (if PDF is corrupted, show error instead of crashing)
- `'rb'`: Read binary mode (PDFs are binary files)
- `PdfReader`: PyPDF2's class to read PDFs
- `for page in pdf_reader.pages`: Iterate through all pages
- `extract_text()`: Get text from page
- `+ "\n"`: Add newline between pages
- `raise Exception`: If error, create new exception with message

**Why binary mode?** PDFs contain binary data (images, fonts), not just text.

#### Text File Loading (Lines 44-50)

```python
def load_txt(self, file_path: str) -> str:
    """Load text from a TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:  # Open in read mode, UTF-8 encoding
            return file.read()                                 # Read entire file, return it
    except Exception as e:
        raise Exception(f"Error reading TXT file {file_path}: {str(e)}")
```

**Key differences from PDF:**
- `'r'`: Text mode (not binary)
- `encoding='utf-8'`: Handle special characters (emojis, accents, etc.)
- `file.read()`: Read entire file at once (simple!)

**Why UTF-8?** Supports all languages and special characters.

#### Markdown Loading (Lines 52-63)

```python
def load_markdown(self, file_path: str) -> str:
    """Load and convert Markdown to plain text."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            md_content = file.read()                    # Read Markdown content
            # Convert markdown to HTML then extract text
            html = markdown.markdown(md_content)        # Convert MD → HTML
            # Simple HTML tag removal
            text = re.sub(r'<[^>]+>', '', html)        # Remove HTML tags
            return text
    except Exception as e:
        raise Exception(f"Error reading Markdown file {file_path}: {str(e)}")
```

**Process:**
1. Read `.md` file
2. Convert Markdown to HTML (markdown library)
3. Remove HTML tags (regex `r'<[^>]+>'` means "anything between < and >")
4. Return plain text

**Why this approach?** Markdown has formatting (`**bold**`, `# heading`). We convert to HTML then strip tags to get clean text.

#### Universal Document Loader (Lines 65-85)

```python
def load_document(self, file_path: str) -> str:
    """
    Load a document based on its file extension.
    """
    file_path = Path(file_path)              # Convert to Path object
    extension = file_path.suffix.lower()     # Get extension (.pdf, .txt, etc.), lowercase
    
    if extension == '.pdf':
        return self.load_pdf(str(file_path))      # Call PDF loader
    elif extension == '.txt':
        return self.load_txt(str(file_path))      # Call TXT loader
    elif extension in ['.md', '.markdown']:
        return self.load_markdown(str(file_path)) # Call Markdown loader
    else:
        raise ValueError(f"Unsupported file type: {extension}")  # Error if unknown type
```

**What it does:**
- Checks file extension
- Calls appropriate loader method
- One function handles all file types!

**Why `.lower()`?** Windows might have `.PDF` (uppercase), we normalize to lowercase.

#### Chunking Function (Lines 87-147)

This is the most complex function! Let's break it down:

```python
def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
    """
    Split text into overlapping chunks.
    """
    if not text.strip():          # If text is empty/whitespace only
        return []                 # Return empty list
    
    chunks = []                   # List to store chunks
    start = 0                     # Start position (character index)
    text_length = len(text)       # Total text length
    
    # Use sentence boundaries when possible for better chunking
    sentence_endings = re.compile(r'[.!?]\s+')  # Pattern: period/exclamation/question + space
```

**Initialization:**
- Check if text is empty
- Initialize variables
- Create regex pattern for sentence endings

**Regex explanation:** `r'[.!?]\s+'`
- `[.!?]`: Match any of these: period, exclamation, question mark
- `\s+`: One or more whitespace characters
- Example matches: ". ", "! ", "?\n"

```python
    while start < text_length:                    # Loop until we've processed all text
        end = start + self.chunk_size             # Calculate end position
        
        if end >= text_length:                    # If we've reached the end
            # Last chunk
            chunk_text = text[start:].strip()    # Get remaining text
        else:
            # Try to break at sentence boundary
            chunk_text = text[start:end]         # Get text from start to end
            last_sentence = sentence_endings.search(chunk_text[::-1])  # Search backwards
```

**Chunking logic:**
- Calculate where chunk should end
- If at end of text, take remaining text
- Otherwise, try to find sentence boundary

**Why search backwards?** `chunk_text[::-1]` reverses string. We search from the end to find the last sentence boundary.

```python
            if last_sentence:
                # Adjust end to sentence boundary
                end = start + len(chunk_text) - last_sentence.start()
                chunk_text = text[start:end].strip()
            else:
                # Break at word boundary
                last_space = chunk_text.rfind(' ')  # Find last space
                if last_space > self.chunk_size * 0.5:  # If space is not too far from start
                    end = start + last_space
                    chunk_text = text[start:end].strip()
                else:
                    chunk_text = chunk_text.strip()
```

**Smart breaking:**
1. Try sentence boundary (best)
2. If not found, try word boundary (good)
3. If word boundary too far, just cut (acceptable)

**Why 0.5 threshold?** If last space is before halfway point, cutting there would make chunk too small. Better to cut at exact position.

```python
        if chunk_text:                            # If chunk has content
            chunk_metadata = {
                'text': chunk_text,               # The actual text
                'chunk_index': len(chunks),       # Which chunk number (0, 1, 2...)
                'start_char': start,              # Where it starts in original text
                'end_char': end,                  # Where it ends
            }
            if metadata:                          # If metadata provided
                chunk_metadata.update(metadata)    # Add it (source, file_path, etc.)
            
            chunks.append(chunk_metadata)         # Add chunk to list
        
        # Move start position with overlap
        start = end - self.chunk_overlap          # Next chunk starts before this one ends
```

**Metadata creation:**
- Store chunk text and position info
- Add source metadata if provided
- Append to chunks list
- Move start position backwards by overlap amount

**Overlap example:**
- Chunk 1: chars 0-512
- Chunk 2: chars 462-974 (starts at 512-50=462)
- Overlap: chars 462-512 (shared between chunks)

#### Directory Processing (Lines 149-183)

```python
def process_directory(self, directory_path: str) -> List[Dict]:
    """
    Process all supported documents in a directory.
    """
    directory = Path(directory_path)             # Convert to Path object
    all_chunks = []                              # Collect all chunks
    
    supported_extensions = ['.pdf', '.txt', '.md', '.markdown']  # Allowed file types
    
    for file_path in directory.iterdir():       # Loop through files in directory
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                print(f"Processing: {file_path.name}")  # Show progress
                text = self.load_document(str(file_path))  # Load document
                
                metadata = {
                    'source': file_path.name,    # Just filename (not full path)
                    'file_path': str(file_path),  # Full path
                }
                
                chunks = self.chunk_text(text, metadata)  # Chunk the text
                all_chunks.extend(chunks)       # Add chunks to master list
                print(f"  Created {len(chunks)} chunks from {file_path.name}")
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {str(e)}")
                continue                         # Skip this file, continue with next
```

**What it does:**
- Scans directory for supported files
- Processes each file
- Collects all chunks from all files
- Handles errors gracefully (continues if one file fails)

**Why `extend()` not `append()`?** `extend()` adds all items from list, `append()` would add the list itself.

---

### 2. embeddings.py

**Purpose:** Converts text into numerical vectors (embeddings)

#### Imports (Lines 1-8)

```python
from typing import List                    # Type hints
from sentence_transformers import SentenceTransformer  # The embedding model
import numpy as np                         # Numerical operations
```

**Why numpy?** Embeddings are arrays of numbers. NumPy handles arrays efficiently.

#### Class Definition (Lines 11-31)

```python
class EmbeddingGenerator:
    """
    Generates embeddings for text using open-source Sentence Transformers models.
    Uses BGE (BAAI General Embedding) models for high-quality embeddings.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        """
        Initialize the embedding generator.
        """
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)  # Load the model (downloads if first time)
        self.model_name = model_name                   # Store model name
        print("Embedding model loaded successfully!")
```

**What happens:**
- `SentenceTransformer(model_name)`: Loads pre-trained model
- First time: Downloads model (~400MB)
- Subsequent times: Loads from cache
- Model is ready to encode text!

**Why BGE?**
- State-of-the-art performance
- Optimized for retrieval (finding similar texts)
- 768 dimensions (good balance of quality/speed)

#### Batch Embedding Generation (Lines 33-55)

```python
def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    """
    if not texts:                          # If empty list
        return np.array([])                # Return empty array
    
    embeddings = self.model.encode(        # Encode texts to embeddings
        texts,                             # List of text strings
        batch_size=32,                     # Process 32 at a time (for memory efficiency)
        show_progress_bar=True,            # Show progress bar
        convert_to_numpy=True,             # Return as numpy array (not PyTorch tensor)
        normalize_embeddings=True          # Normalize vectors (length = 1)
    )
    
    return embeddings                      # Return array of shape (num_texts, 768)
```

**Key parameters:**
- `batch_size=32`: Process 32 texts at once (faster than one-by-one, uses less memory than all at once)
- `normalize_embeddings=True`: Makes all vectors unit length (for cosine similarity)

**Why normalize?** Cosine similarity works best with normalized vectors. Also prevents longer texts from having larger vectors.

**Output shape:** If you have 100 texts, output is (100, 768) - 100 rows, 768 columns.

#### Query Embedding (Lines 57-85)

```python
def generate_query_embedding(self, query: str) -> np.ndarray:
    """
    Generate embedding for a single query.
    Uses instruction-based encoding for better query-document matching.
    
    """
    # BGE models support instruction-based encoding for queries
    if "bge" in self.model_name.lower():   # If using BGE model
        # Format query with instruction for better retrieval
        instruction = "Represent this sentence for searching relevant passages:"
        query_with_instruction = f"{instruction} {query}"  # Add instruction prefix
        embedding = self.model.encode(
            query_with_instruction,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    else:
        # For other models, encode directly
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    return embedding                       # Returns shape (768,)
```

**Why instruction prefix?**

- BGE models are trained to understand this instruction
- Makes query embeddings better match document embeddings
- Improves retrieval accuracy!

**Example:**
- Query: "What is machine learning?"
- With instruction: "Represent this sentence for searching relevant passages: What is machine learning?"
- Model understands: "This is a search query, encode it for retrieval"

#### Embedding Dimension Property (Lines 87-92)

```python
@property
def embedding_dimension(self) -> int:
    """Get the dimension of embeddings produced by this model."""
    # Get dimension by encoding a dummy text
    dummy_embedding = self.model.encode(["dummy"], convert_to_numpy=True)
    return dummy_embedding.shape[1]        # Return the second dimension (768)
```

**What's a property?** Python decorator that makes a method callable like an attribute.
- Instead of: `generator.embedding_dimension()` (method call)
- You can do: `generator.embedding_dimension` (attribute access)

**Why dummy encoding?** Different models have different dimensions. This dynamically detects the dimension.

**Shape explanation:** `dummy_embedding.shape` = (1, 768)
- `[0]` = 1 (one text)
- `[1]` = 768 (embedding dimension)

---

### 3. vector_store.py

**Purpose:** Manages FAISS vector database for similarity search

#### Imports (Lines 1-10)

```python
import os
import pickle                    # Save/load Python objects to files
import numpy as np
import faiss                    # Facebook AI Similarity Search library
from typing import List, Dict, Tuple
```

**Why pickle?** Need to save metadata (Python dictionaries) to disk. Pickle can serialize any Python object.

#### Class Initialization (Lines 13-44)

```python
class VectorStore:
    """
    Manages vector storage and similarity search using FAISS.
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embeddings (e.g., 768)
            index_type: Type of FAISS index
        """
        self.dimension = dimension
        self.index_type = index_type
        
        # Create FAISS index
        if index_type == "cosine" or index_type == "flat":
            # Use inner product for cosine similarity (with normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)  # Inner Product index
        elif index_type == "l2":
            self.index = faiss.IndexFlatL2(dimension)   # L2 distance index
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Store metadata for each vector
        self.metadata = []                              # List to store chunk metadata
```

**FAISS Index Types:**
- `IndexFlatIP`: Inner Product (for normalized vectors = cosine similarity)
- `IndexFlatL2`: Euclidean distance (L2 norm)
- `IndexFlatIP` is faster and better for normalized embeddings

**Why store metadata separately?** FAISS only stores vectors. We need to remember which vector belongs to which document/chunk.

#### Adding Vectors (Lines 46-68)

```python
def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict]):
    """
    Add vectors and their metadata to the index.
    """
    if len(vectors) != len(metadata_list):
        raise ValueError("Number of vectors must match number of metadata entries")
    
    if vectors.shape[1] != self.dimension:
        raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}")
    
    # Ensure vectors are normalized for cosine similarity
    if self.index_type == "cosine" or self.index_type == "flat":
        faiss.normalize_L2(vectors)              # Normalize vectors in-place
    
    # Add to FAISS index
    self.index.add(vectors.astype('float32'))    # FAISS requires float32
    
    # Store metadata
    self.metadata.extend(metadata_list)         # Add metadata to list
```

**Validation:**
- Check vector count matches metadata count
- Check vector dimension matches index dimension
- Normalize vectors (if using cosine similarity)

**Why `astype('float32')`?** FAISS requires 32-bit floats (not 64-bit doubles). Saves memory and is faster.

**Why `extend()`?** Adds all metadata items to the list (one per vector).

#### Search Function (Lines 70-109)

```python
def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
    """
    Search for the k most similar vectors.
    """
    if self.index.ntotal == 0:                  # If index is empty
        return []                              # Return empty list
    
    # Reshape query vector for FAISS
    query_vector = query_vector.reshape(1, -1).astype('float32')  # Shape: (1, 768)
    
    # Normalize query vector for cosine similarity
    if self.index_type == "cosine" or self.index_type == "flat":
        faiss.normalize_L2(query_vector)       # Normalize query too
    
    # Search
    k = min(k, self.index.ntotal)              # Don't request more than available
    distances, indices = self.index.search(query_vector, k)  # Search!
```

**Reshape explanation:**
- Query vector comes in as shape (768,) - 1D array
- FAISS expects (1, 768) - 2D array with 1 row
- `reshape(1, -1)` means: 1 row, auto-calculate columns

**Search returns:**
- `distances`: Similarity scores (higher = more similar for IP)
- `indices`: Which vectors matched (indices into our metadata list)

```python
    # Format results
    results = []
    for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
        if idx < len(self.metadata):           # Safety check
            # For cosine similarity, distance is inner product (higher is better)
            # Convert to similarity score between 0 and 1
            if self.index_type == "cosine" or self.index_type == "flat":
                similarity = float((distance + 1) / 2)  # Normalize to [0, 1]
            else:
                # For L2, lower distance is better, convert to similarity
                similarity = float(1 / (1 + distance))
            
            results.append((self.metadata[idx], similarity))  # (metadata, score)
    
    return results
```

**Score conversion:**
- Inner Product: Range is roughly [-1, 1]. We convert to [0, 1] with `(distance + 1) / 2`
- L2 Distance: Lower is better. Convert to similarity: `1 / (1 + distance)` (closer = higher similarity)

**Why `indices[0]`?** Search returns shape (1, k) - one query, k results. We take the first (and only) row.

#### Save Function (Lines 111-136)

```python
def save(self, directory: str):
    """
    Save the index and metadata to disk.
    """
    os.makedirs(directory, exist_ok=True)      # Create directory if doesn't exist
    
    # Save FAISS index
    index_path = os.path.join(directory, "faiss.index")
    faiss.write_index(self.index, index_path)   # Save FAISS index
    
    # Save metadata
    metadata_path = os.path.join(directory, "metadata.pkl")
    with open(metadata_path, 'wb') as f:       # 'wb' = write binary
        pickle.dump(self.metadata, f)          # Serialize metadata to file
    
    # Save configuration
    config_path = os.path.join(directory, "config.pkl")
    with open(config_path, 'wb') as f:
        pickle.dump({
            'dimension': self.dimension,
            'index_type': self.index_type,
            'num_vectors': self.index.ntotal    # Total vectors in index
        }, f)
```

**What gets saved:**
1. FAISS index file (vectors)
2. Metadata pickle file (chunk info)
3. Config pickle file (settings)

**Why pickle for metadata?** Can save complex Python objects (dictionaries, lists) easily.

**`exist_ok=True`:** Don't error if directory already exists.

#### Load Function (Lines 138-166)

```python
@classmethod
def load(cls, directory: str) -> 'VectorStore':
    """
    Load the index and metadata from disk.
    """
    # Load configuration
    config_path = os.path.join(directory, "config.pkl")
    with open(config_path, 'rb') as f:          # 'rb' = read binary
        config = pickle.load(f)                # Deserialize config
    
    # Create instance
    store = cls(config['dimension'], config['index_type'])  # Create new VectorStore
    
    # Load FAISS index
    index_path = os.path.join(directory, "faiss.index")
    store.index = faiss.read_index(index_path)  # Load FAISS index
    
    # Load metadata
    metadata_path = os.path.join(directory, "metadata.pkl")
    with open(metadata_path, 'rb') as f:
        store.metadata = pickle.load(f)        # Load metadata
    
    return store                                # Return loaded store
```

**What's `@classmethod`?** Method that belongs to the class, not an instance. Called like: `VectorStore.load("path")` instead of `store.load("path")`.

**Process:**
1. Load config to know dimensions/index type
2. Create new VectorStore with those settings
3. Load FAISS index into it
4. Load metadata
5. Return ready-to-use store

#### Stats Function (Lines 168-174)

```python
def get_stats(self) -> Dict:
    """Get statistics about the vector store."""
    return {
        'num_vectors': self.index.ntotal,       # How many vectors stored
        'dimension': self.dimension,            # Vector dimension
        'index_type': self.index_type          # Index type used
    }
```

Simple utility to get information about the store.

---

### 4. guardrails.py

**Purpose:** Safety checks to prevent hallucinations and ensure quality

#### Class Initialization (Lines 11-43)

```python
class Guardrails:
    """
    Implements guardrails for RAG system:
    - Confidence thresholding
    - Context-only answer enforcement
    - Output filtering
    - Hallucination detection
    """
    
    def __init__(self, min_confidence: float = 0.5, require_sources: bool = True):
        """
        Initialize guardrails.
        """
        self.min_confidence = min_confidence
        self.require_sources = require_sources
        
        # Keywords that might indicate hallucination or refusal
        self.refusal_keywords = [
            "i don't know",
            "i cannot",
            "i'm not sure",
            # ... more keywords
        ]
```

**Refusal keywords:** Phrases that indicate the model doesn't know something. We track these to detect if the model is appropriately refusing (good) vs. hallucinating (bad).

#### Confidence Checking (Lines 45-64)

```python
def check_confidence(self, similarity_scores: List[float]) -> Tuple[bool, float]:
    """
    Check if retrieved contexts meet confidence threshold.
    """
    if not similarity_scores:                  # If no scores
        return False, 0.0                       # Failed, no average
    
    avg_score = sum(similarity_scores) / len(similarity_scores)  # Calculate average
    max_score = max(similarity_scores)         # Get highest score
    
    # Check if max score meets threshold
    passed = max_score >= self.min_confidence  # True if best match is good enough
    
    return passed, avg_score                   # Return (passed?, average_score)
```

**Logic:** If the BEST match is below threshold, all matches are probably bad. Reject the whole retrieval.

#### Low Confidence Filtering (Lines 66-79)

```python
def filter_low_confidence(self, results: List[Tuple[Dict, float]], 
                          min_score: Optional[float] = None) -> List[Tuple[Dict, float]]:
    """
    Filter out results below confidence threshold.
    """
    threshold = min_score if min_score is not None else self.min_confidence
    return [(meta, score) for meta, score in results if score >= threshold]
```

**List comprehension:** Python's concise way to filter lists.
- Equivalent to:
```python
filtered = []
for meta, score in results:
    if score >= threshold:
        filtered.append((meta, score))
return filtered
```

#### Hallucination Detection (Lines 81-98)

```python
def detect_hallucination_indicators(self, response: str) -> List[str]:
    """
    Detect potential hallucination indicators in the response.
    """
    detected = []
    response_lower = response.lower()          # Case-insensitive search
    
    for keyword in self.refusal_keywords:
        if keyword in response_lower:          # If keyword found
            detected.append(keyword)           # Add to list
    
    return detected                            # Return all found keywords
```

**Purpose:** Find phrases that suggest uncertainty. This is actually GOOD if the answer isn't in the documents (model is being honest).

#### Context Enforcement (Lines 100-140)

```python
def enforce_context_only(self, response: str, contexts: List[str]) -> Tuple[str, bool]:
    """
    Check if response is based on provided contexts.
    """
    response_lower = response.lower()
    
    # If response explicitly says it doesn't know, that's actually good
    if any(phrase in response_lower for phrase in [
        "based on the provided",
        "according to the documents",
        # ... more phrases
    ]):
        return response, True                  # Good - acknowledges sources
    
    # Check if response contains refusal without context reference
    refusal_detected = self.detect_hallucination_indicators(response)
    
    if refusal_detected:
        if contexts:                           # If we have contexts
            return response, True              # Refusal is valid (info not in docs)
        else:
            return response, False             # No contexts but refusing = suspicious
    
    return response, True                      # Default: assume valid
```

**Logic:**
1. If response cites sources → Good!
2. If response refuses AND we have contexts → Good (honest refusal)
3. If response refuses BUT no contexts → Suspicious (might be hallucinating refusal)

#### Comprehensive Validation (Lines 142-175)

```python
def validate_response(self, response: str, contexts: List[str], 
                     similarity_scores: List[float]) -> Dict:
    """
    Comprehensive validation of response.
    """
    # Check confidence
    confidence_passed, avg_confidence = self.check_confidence(similarity_scores)
    
    # Check context enforcement
    response_valid, is_context_based = self.enforce_context_only(response, contexts)
    
    # Detect hallucination indicators
    hallucination_indicators = self.detect_hallucination_indicators(response)
    
    # Overall validation
    is_valid = confidence_passed and is_context_based
    
    return {
        'is_valid': is_valid,
        'confidence_passed': confidence_passed,
        'average_confidence': avg_confidence,
        'max_confidence': max(similarity_scores) if similarity_scores else 0.0,
        'is_context_based': is_context_based,
        'hallucination_indicators': hallucination_indicators,
        'response': response_valid
    }
```

**Combines all checks** into one validation result. Used by the pipeline to decide if response is trustworthy.

#### Warning Formatter (Lines 177-210)

```python
def format_response_with_warning(self, response: str, validation: Dict) -> str:
    """
    Format response with warnings if validation fails.
    """
    warnings = []
    
    if not validation['confidence_passed']:
        warnings.append(f"⚠️ Low confidence: ...")
    
    if not validation['is_context_based']:
        warnings.append("⚠️ Warning: Response may not be fully based on provided documents")
    
    if validation['hallucination_indicators']:
        warnings.append("⚠️ Note: Response indicates uncertainty about the query")
    
    if warnings:
        warning_text = "\n\n".join(warnings)
        return f"{response}\n\n---\n{warning_text}"
    
    return response
```

**Adds warnings** to response if validation found issues. User sees the answer but also knows about potential problems.

---

### 5. rag_pipeline.py

**Purpose:** Main RAG pipeline - orchestrates everything

#### Imports (Lines 1-10)

```python
from typing import List, Dict, Optional
import ollama                              # Ollama LLM client
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from guardrails import Guardrails
```

**Why import our modules?** Pipeline uses all the components we built!

#### Class Initialization (Lines 13-45)

```python
class RAGPipeline:
    """
    Complete RAG pipeline implementation.
    """
    
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 llm_model: str = "llama3",
                 vector_store: Optional[VectorStore] = None,
                 min_confidence: float = 0.5):
        """
        Initialize the RAG pipeline.
        """
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.llm_model = llm_model
        self.guardrails = Guardrails(min_confidence=min_confidence)
        
        if vector_store is None:
            dimension = self.embedding_generator.embedding_dimension
            self.vector_store = VectorStore(dimension=dimension, index_type="cosine")
        else:
            self.vector_store = vector_store
```

**Initialization:**
- Creates embedding generator
- Sets LLM model name
- Creates guardrails
- Creates new vector store OR uses provided one (for loading saved stores)

#### Prompt Building (Lines 47-103)

```python
def build_prompt(self, query: str, contexts: List[Dict], max_context_length: int = 2000) -> str:
    """
    Build augmented prompt with retrieved contexts.
    """
    # Format contexts
    context_texts = []
    total_length = 0
    
    for ctx in contexts:
        text = ctx['text']
        source = ctx.get('source', 'Unknown')
        
        if total_length + len(text) > max_context_length:
            # Truncate last context if needed
            remaining = max_context_length - total_length
            if remaining > 100:  # Only add if meaningful
                text = text[:remaining] + "..."
                context_texts.append(f"[Source: {source}]\n{text}")
            break
        
        context_texts.append(f"[Source: {source}]\n{text}")
        total_length += len(text)
    
    contexts_str = "\n\n---\n\n".join(context_texts)
```

**Context formatting:**
- Adds source labels to each context
- Limits total length (LLMs have token limits)
- Separates contexts with dividers

```python
    # Build system prompt with strict instructions
    system_instruction = """You are a helpful knowledge assistant that answers questions STRICTLY based on the provided documents. 

CRITICAL RULES:
1. Answer ONLY using information from the provided context documents
2. If the answer is not in the provided documents, explicitly state "Based on the provided documents, I cannot find information about [topic]"
3. Do NOT make up information or use external knowledge
4. Cite the source document when referencing specific information
5. If multiple sources contain relevant information, synthesize them clearly
6. Be concise but complete in your answers

Context Documents:
{contexts}

User Question: {query}

Answer (based ONLY on the provided documents):"""

    prompt = system_instruction.format(
        contexts=contexts_str,
        query=query
    )
    
    return prompt
```

**Prompt engineering:** Very explicit instructions to prevent hallucinations. The model sees this every time and is "trained" by the prompt to follow rules.

#### Context Retrieval (Lines 105-122)

```python
def retrieve_contexts(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
    """
    Retrieve relevant contexts for a query.
    """
    # Generate query embedding
    query_embedding = self.embedding_generator.generate_query_embedding(query)
    
    # Search vector store
    results = self.vector_store.search(query_embedding, k=top_k)
    
    return results
```

**Simple but powerful:** Embed query → Search → Return top matches.

#### Main Generation Function (Lines 124-221)

This is the heart of RAG! Let's break it down:

```python
def generate_response(self, query: str, top_k: int = 5, 
                     use_guardrails: bool = True) -> Dict:
    """
    Complete RAG flow: retrieve contexts and generate response.
    """
    # Step 1: Retrieve contexts
    retrieval_results = self.retrieve_contexts(query, top_k=top_k)
    
    if not retrieval_results:
        return {
            'response': "I couldn't find any relevant documents...",
            'sources': [],
            'confidence': 0.0,
            'validation': {'is_valid': False, 'confidence_passed': False}
        }
```

**Early exit:** If no results, return immediately (nothing to work with).

```python
    # Extract contexts and scores
    contexts = [meta for meta, score in retrieval_results]
    scores = [score for meta, score in retrieval_results]
    
    # Step 2: Apply confidence filtering
    if use_guardrails:
        filtered_results = self.guardrails.filter_low_confidence(retrieval_results)
        if not filtered_results:
            return {
                'response': f"I couldn't find documents with sufficient relevance...",
                'sources': [ctx.get('source', 'Unknown') for ctx in contexts],
                'confidence': max(scores),
                # ... validation info
            }
        contexts = [meta for meta, score in filtered_results]
        scores = [score for meta, score in filtered_results]
```

**Filtering:** Remove low-confidence results. If nothing passes, return error message.

```python
    # Step 3: Build augmented prompt
    prompt = self.build_prompt(query, contexts)
    
    # Step 4: Generate response using Ollama
    try:
        # Try chat API first (preferred), fallback to generate
        try:
            response = ollama.chat(
                model=self.llm_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'temperature': 0.1,  # Low temperature for factual responses
                    'top_p': 0.9,
                }
            )
            response_text = response['message']['content']
        except (AttributeError, KeyError):
            # Fallback to generate API
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={'temperature': 0.1, 'top_p': 0.9}
            )
            response_text = response.get('response', str(response))
    except Exception as e:
        response_text = f"Error generating response: {str(e)}..."
```

**LLM generation:**
- Try `chat()` API first (newer, better)
- Fallback to `generate()` API (older, still works)
- Low temperature (0.1) = less creative, more factual
- Error handling if Ollama not running

**Why two APIs?** Different Ollama versions use different APIs. We support both.

```python
    # Step 5: Validate response with guardrails
    validation = None
    if use_guardrails:
        validation = self.guardrails.validate_response(response_text, contexts, scores)
        response_text = self.guardrails.format_response_with_warning(response_text, validation)
    
    # Extract unique sources
    sources = list(set([ctx.get('source', 'Unknown') for ctx in contexts]))
    
    return {
        'response': response_text,
        'sources': sources,
        'contexts': contexts,
        'confidence': max(scores) if scores else 0.0,
        'average_confidence': sum(scores) / len(scores) if scores else 0.0,
        'validation': validation or {}
    }
```

**Final steps:**
- Validate response
- Add warnings if needed
- Extract unique sources (remove duplicates)
- Return everything in a dictionary

#### Add Documents (Lines 223-238)

```python
def add_documents(self, chunks: List[Dict]):
    """
    Add document chunks to the vector store.
    """
    texts = [chunk['text'] for chunk in chunks]  # Extract just the text
    
    # Generate embeddings
    embeddings = self.embedding_generator.generate_embeddings(texts)
    
    # Add to vector store
    self.vector_store.add_vectors(embeddings, chunks)
    
    print(f"Added {len(chunks)} chunks to vector store")
```

**Simple workflow:** Extract text → Embed → Store. Called when loading documents.

---

### 6. app.py (Streamlit UI)

**Purpose:** Web interface for the RAG system

#### Imports and Setup (Lines 1-26)

```python
import streamlit as st
import os
from pathlib import Path
from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline
from vector_store import VectorStore

# Page configuration
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False
if 'vector_store_path' not in st.session_state:
    st.session_state.vector_store_path = "vector_store"
```

**Session state:** Streamlit's way to store data between interactions. Like global variables but scoped to the user's session.

**Why session state?** Without it, variables reset on every interaction. Session state persists.

#### Initialization Function (Lines 28-46)

```python
def initialize_rag_pipeline():
    """Initialize or load the RAG pipeline."""
    vector_store_path = st.session_state.vector_store_path
    
    if os.path.exists(vector_store_path) and os.path.exists(os.path.join(vector_store_path, "faiss.index")):
        # Load existing vector store
        try:
            vector_store = VectorStore.load(vector_store_path)
            st.session_state.rag_pipeline = RAGPipeline(vector_store=vector_store)
            st.session_state.documents_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
            return False
    else:
        # Create new pipeline (will create vector store when documents are added)
        st.session_state.rag_pipeline = RAGPipeline()
        st.session_state.documents_loaded = False
        return True
```

**Logic:**
- Check if vector store exists
- If yes: Load it (documents already processed)
- If no: Create empty pipeline (will process documents later)

#### Document Loading Function (Lines 48-76)

```python
def load_documents():
    """Load and process documents."""
    documents_dir = "documents"
    
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        st.warning(f"Created '{documents_dir}' directory...")
        return False
    
    # Process documents
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    with st.spinner("Processing documents..."):
        chunks = processor.process_directory(documents_dir)
    
    if not chunks:
        st.error("No documents found...")
        return False
    
    # Add to vector store
    with st.spinner("Generating embeddings and building vector index..."):
        st.session_state.rag_pipeline.add_documents(chunks)
    
    # Save vector store
    st.session_state.rag_pipeline.vector_store.save(st.session_state.vector_store_path)
    
    st.success(f"Successfully loaded {len(chunks)} document chunks...")
    st.session_state.documents_loaded = True
    return True
```

**Process:**
1. Check/create documents directory
2. Process documents (chunking)
3. Generate embeddings and add to vector store
4. Save vector store to disk
5. Update session state

**`st.spinner()`:** Shows loading spinner while processing (better UX).

#### UI Layout (Lines 78-226)

```python
# Main UI
st.title("📚 RAG Knowledge Assistant")
st.markdown("**Answer questions based strictly on your provided documents**")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    embedding_model = st.selectbox(
        "Embedding Model",
        ["BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"],
        index=0
    )
    
    llm_model = st.text_input("LLM Model (Ollama)", value="llama3")
    
    min_confidence = st.slider(
        "Minimum Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    top_k = st.slider(
        "Number of Contexts (Top-K)",
        min_value=1,
        max_value=10,
        value=5
    )
```

**Streamlit widgets:**
- `st.selectbox()`: Dropdown menu
- `st.text_input()`: Text field
- `st.slider()`: Slider for numbers
- `st.sidebar`: Puts content in sidebar

```python
    # Document management
    st.header("📄 Document Management")
    
    if st.button("🔄 Load/Reload Documents", type="primary"):
        if initialize_rag_pipeline():
            if load_documents():
                st.rerun()  # Refresh the page
```

**Button:** When clicked, processes documents. `st.rerun()` refreshes the page to show updated state.

```python
# Main content area
if not st.session_state.documents_loaded:
    st.info("👆 Please load documents using the sidebar before asking questions.")
    # ... show available documents
else:
    # Query interface
    st.subheader("💬 Ask a Question")
    
    query = st.text_input("Enter your question:", ...)
    
    if st.button("🔍 Search", type="primary") or query:
        if query:
            with st.spinner("Searching documents and generating response..."):
                # Update pipeline settings
                if st.session_state.rag_pipeline.llm_model != llm_model:
                    st.session_state.rag_pipeline.llm_model = llm_model
                # ... more updates
                
                # Generate response
                result = st.session_state.rag_pipeline.generate_response(
                    query=query,
                    top_k=top_k,
                    use_guardrails=True
                )
            
            # Display response
            st.subheader("📝 Answer")
            st.markdown(result['response'])
            
            # Display sources
            if result['sources']:
                st.subheader("📚 Sources")
                for source in result['sources']:
                    st.caption(f"• {source}")
```

**Query flow:**
1. Check if documents loaded
2. If not: Show instructions
3. If yes: Show query input
4. On search: Generate response
5. Display answer and sources

**`or query`:** Also triggers on Enter key (Streamlit feature).

---

### 7. main.py (CLI)

**Purpose:** Command-line interface for the RAG system

#### Setup Function (Lines 14-68)

```python
def setup_documents(documents_dir: str = "documents", 
                   vector_store_dir: str = "vector_store",
                   chunk_size: int = 512,
                   chunk_overlap: int = 50):
    """
    Process documents and build vector store.
    """
    print("=" * 60)
    print("RAG System Setup")
    print("=" * 60)
    
    # Check if documents directory exists
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        print(f"Created '{documents_dir}' directory.")
        return False
    
    # Process documents
    print(f"\n1. Processing documents from '{documents_dir}'...")
    processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = processor.process_directory(documents_dir)
    
    if not chunks:
        print(f"❌ No documents found in '{documents_dir}'")
        return False
    
    print(f"✅ Processed {len(chunks)} chunks from documents")
    
    # Initialize RAG pipeline
    print("\n2. Initializing RAG pipeline...")
    pipeline = RAGPipeline()
    
    # Add documents to vector store
    print("3. Generating embeddings and building vector index...")
    pipeline.add_documents(chunks)
    
    # Save vector store
    print(f"4. Saving vector store to '{vector_store_dir}'...")
    pipeline.vector_store.save(vector_store_dir)
    
    stats = pipeline.vector_store.get_stats()
    print(f"\n✅ Setup complete!")
    print(f"   - Total chunks: {stats['num_vectors']}")
    print(f"   - Embedding dimension: {stats['dimension']}")
    print(f"   - Vector store saved to: {vector_store_dir}")
    
    return True
```

**CLI setup:** Same process as UI, but prints to console instead of showing in browser.

#### Query Function (Lines 71-129)

```python
def query_cli(query: str, 
             vector_store_dir: str = "vector_store",
             top_k: int = 5,
             llm_model: str = "llama3"):
    """
    Query the RAG system from command line.
    """
    print("=" * 60)
    print("RAG Query")
    print("=" * 60)
    
    # Load vector store
    if not os.path.exists(vector_store_dir):
        print(f"❌ Vector store not found at '{vector_store_dir}'")
        print("Please run setup first: python main.py setup")
        return
    
    print(f"Loading vector store from '{vector_store_dir}'...")
    vector_store = VectorStore.load(vector_store_dir)
    
    # Initialize pipeline
    pipeline = RAGPipeline(vector_store=vector_store, llm_model=llm_model)
    
    # Query
    print(f"\nQuery: {query}")
    print("\nSearching and generating response...\n")
    
    result = pipeline.generate_response(query, top_k=top_k)
    
    # Display results
    print("=" * 60)
    print("ANSWER")
    print("=" * 60)
    print(result['response'])
    print()
    
    if result['sources']:
        print("=" * 60)
        print("SOURCES")
        print("=" * 60)
        for source in result['sources']:
            print(f"• {source}")
        print()
    
    print("=" * 60)
    print("METADATA")
    print("=" * 60)
    print(f"Max Confidence: {result['confidence']:.3f}")
    print(f"Avg Confidence: {result['average_confidence']:.3f}")
    # ... more metadata
```

**CLI query:** Loads store, queries, prints formatted results.

#### Main Function (Lines 132-179)

```python
def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="RAG Knowledge Assistant")
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Process documents and build vector store')
    setup_parser.add_argument('--documents-dir', default='documents', ...)
    setup_parser.add_argument('--vector-store-dir', default='vector_store', ...)
    setup_parser.add_argument('--chunk-size', type=int, default=512, ...)
    setup_parser.add_argument('--chunk-overlap', type=int, default=50, ...)
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('query', help='Query text')
    query_parser.add_argument('--vector-store-dir', default='vector_store', ...)
    query_parser.add_argument('--top-k', type=int, default=5, ...)
    query_parser.add_argument('--llm-model', default='llama3', ...)
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_documents(...)
    elif args.command == 'query':
        query_cli(...)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

**Argument parsing:** Uses Python's `argparse` to handle command-line arguments.

**Usage:**
```bash
python main.py setup
python main.py query "What is machine learning?"
```

---

## How Everything Works Together

### Complete Flow Example

**User asks:** "What is machine learning?"

1. **app.py** receives query
2. **rag_pipeline.py** `generate_response()` called
3. **embeddings.py** converts query to vector
4. **vector_store.py** searches for similar vectors
5. **rag_pipeline.py** retrieves top 5 chunks
6. **guardrails.py** filters low-confidence results
7. **rag_pipeline.py** builds prompt with contexts
8. **ollama** generates response
9. **guardrails.py** validates response
10. **app.py** displays answer + sources

### Data Flow

```
Documents (PDF/TXT/MD)
    ↓
DocumentProcessor.chunk_text()
    ↓
Chunks [{text, source, ...}, ...]
    ↓
EmbeddingGenerator.generate_embeddings()
    ↓
Vectors (numpy array: shape (N, 768))
    ↓
VectorStore.add_vectors()
    ↓
FAISS Index + Metadata
    ↓
[SAVED TO DISK]
    ↓
[LOADED WHEN NEEDED]
    ↓
Query → Embedding → Search → Retrieve → Prompt → LLM → Answer
```

---

## Common Questions

### Q: Why chunk documents?
**A:** LLMs have token limits (~2000-4000 tokens). Long documents won't fit. Also, smaller chunks = more precise retrieval (find exact relevant section).

### Q: Why overlap chunks?
**A:** Prevents losing context at boundaries. Example: Sentence split between chunks. Overlap ensures both chunks have the sentence.

### Q: Why normalize embeddings?
**A:** Cosine similarity works best with normalized vectors. Also ensures all vectors have same "length" (magnitude).

### Q: What if no documents match?
**A:** System returns "I couldn't find relevant documents" instead of making something up. This is correct behavior!

### Q: Can I use different models?
**A:** Yes! Change `embedding_model` and `llm_model` parameters. Make sure Ollama model is installed: `ollama pull model_name`

### Q: How do I add more documents?
**A:** Just add files to `documents/` folder and click "Load/Reload Documents" in UI or run `python main.py setup` in CLI.

### Q: What's the difference between UI and CLI?
**A:** Same functionality, different interface. UI is easier for users, CLI is better for automation/scripts.

### Q: Why FAISS instead of database?
**A:** FAISS is optimized for vector similarity search. Regular databases (SQL) are slow for this. FAISS can search millions of vectors in milliseconds.

### Q: What if Ollama isn't running?
**A:** You'll get an error. Start it: `ollama serve` in a terminal.

### Q: Can I use OpenAI instead of Ollama?
**A:** Yes, but you'd need to modify `rag_pipeline.py` to use OpenAI API instead of Ollama. It's not open-source though (costs money, sends data to cloud).

---

## Summary

You now understand:
- ✅ Every line of code in every file
- ✅ How the folder structure works
- ✅ Why each technology was chosen
- ✅ All key terms and concepts
- ✅ How everything connects together
- ✅ How to modify and extend the project

**You're now a RAG pro!** 🎉

This project demonstrates:
- Complete RAG pipeline
- Open-source tool integration
- Safety guardrails
- Source attribution
- User-friendly interfaces

You can now:
- Modify the code for your needs
- Add new features
- Explain the project to others
- Build similar systems

**Next steps:**
1. Run the project and experiment
2. Try different models
3. Add your own documents
4. Modify chunk sizes
5. Add new features (e.g., conversation memory, reranking)

Good luck! 🚀
