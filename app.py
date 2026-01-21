"""
Streamlit UI for RAG Application
Provides a user-friendly web interface for the Knowledge Assistant.
"""

import streamlit as st
import os
from pathlib import Path
from document_processor import DocumentProcessor
from rag_pipeline import RAGPipeline
from vector_store import VectorStore

# Fixed configuration values
MIN_CONFIDENCE_DEFAULT = 0.5  # Fixed minimum confidence threshold
TOP_K_DEFAULT = 5             # Fixed number of contexts (Top-K)

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

def load_documents():
    """Load and process documents."""
    documents_dir = "documents"
    
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
        st.warning(f"Created '{documents_dir}' directory. Please add your documents (PDF, TXT, or Markdown) there.")
        return False
    
    # Process documents
    processor = DocumentProcessor(chunk_size=512, chunk_overlap=50)
    
    with st.spinner("Processing documents..."):
        chunks = processor.process_directory(documents_dir)
    
    if not chunks:
        st.error("No documents found or processed. Please add PDF, TXT, or Markdown files to the 'documents' folder.")
        return False
    
    # Add to vector store
    with st.spinner("Generating embeddings and building vector index..."):
        st.session_state.rag_pipeline.add_documents(chunks)
    
    # Save vector store
    st.session_state.rag_pipeline.vector_store.save(st.session_state.vector_store_path)
    
    st.success(f"Successfully loaded {len(chunks)} document chunks from {documents_dir}!")
    st.session_state.documents_loaded = True
    return True

# Main UI
st.title("RAG Knowledge Assistant")
st.markdown("**Answer questions based strictly on your provided documents**")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    # Model selection
    embedding_model = st.selectbox(
        "Embedding Model",
        ["BAAI/bge-base-en-v1.5", "BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"],
        index=0
    )

    llm_model = st.text_input("LLM Model (Ollama)", value="llama3", 
                              help="Make sure Ollama is running and the model is installed")

    # Use fixed values instead of sliders for these settings
    min_confidence = MIN_CONFIDENCE_DEFAULT
    top_k = TOP_K_DEFAULT

    st.markdown(f"**Minimum Confidence Threshold (fixed):** {min_confidence:.2f}")
    st.markdown(f"**Number of Contexts (Top-K, fixed):** {top_k}")
    
    st.divider()
    
    # Document management
    st.header("Document Management")
    
    if st.button("Load/Reload Documents", type="primary"):
        if initialize_rag_pipeline():
            if load_documents():
                st.rerun()
    
    if st.session_state.documents_loaded:
        stats = st.session_state.rag_pipeline.vector_store.get_stats()
        st.success(f"{stats['num_vectors']} chunks loaded")
    else:
        st.info("Click 'Load/Reload Documents' to process documents")
    
    st.divider()
    
    # System info
    st.header("ℹSystem Info")
    st.caption("**Architecture:**")
    st.caption("Query → Embedding → Vector Search → Context Retrieval → LLM Generation")
    
    st.caption("**Components:**")
    st.caption("• Embeddings: Sentence Transformers")
    st.caption("• Vector DB: FAISS")
    st.caption("• LLM: Ollama (Local)")

# Initialize pipeline
if st.session_state.rag_pipeline is None:
    initialize_rag_pipeline()

# Main content area
if not st.session_state.documents_loaded:
    st.info("Please load documents using the sidebar before asking questions.")
    
    # Show document directory info
    documents_dir = "documents"
    if os.path.exists(documents_dir):
        files = list(Path(documents_dir).glob("*"))
        supported_files = [f for f in files if f.suffix.lower() in ['.pdf', '.txt', '.md', '.markdown']]
        
        if supported_files:
            st.subheader("Available Documents")
            for file in supported_files:
                st.text(f"• {file.name}")
        else:
            st.warning(f"No supported documents found in '{documents_dir}' folder.")
            st.caption("Supported formats: PDF (.pdf), Text (.txt), Markdown (.md, .markdown)")
else:
    # Query interface
    st.subheader("Ask a Question")
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., What is the main topic discussed in the documents?",
        label_visibility="collapsed"
    )
    
    if st.button("Search", type="primary") or query:
        if query:
            with st.spinner("Searching documents and generating response..."):
                # Update pipeline settings if changed
                if st.session_state.rag_pipeline.llm_model != llm_model:
                    st.session_state.rag_pipeline.llm_model = llm_model
                if st.session_state.rag_pipeline.guardrails.min_confidence != min_confidence:
                    st.session_state.rag_pipeline.guardrails.min_confidence = min_confidence
                
                # Generate response
                result = st.session_state.rag_pipeline.generate_response(
                    query=query,
                    top_k=top_k,
                    use_guardrails=True
                )
            
            # Display response
            st.subheader("Answer")
            st.markdown(result['response'])
            
            # Display sources
            if result['sources']:
                st.subheader("Sources")
                for source in result['sources']:
                    st.caption(f"• {source}")
            
            # Display metadata
            with st.expander("Retrieval Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Max Confidence", f"{result['confidence']:.3f}")
                with col2:
                    st.metric("Avg Confidence", f"{result['average_confidence']:.3f}")
                
                if result.get('validation'):
                    validation = result['validation']
                    if validation.get('is_valid'):
                        st.success("Response passed validation checks")
                    else:
                        st.warning("Response validation raised concerns")
                    
                    if validation.get('hallucination_indicators'):
                        st.caption(f"Indicators: {', '.join(validation['hallucination_indicators'])}")
                
                # Show retrieved contexts
                if result.get('contexts'):
                    st.subheader("Retrieved Contexts")
                    for i, ctx in enumerate(result['contexts'][:3], 1):  # Show top 3
                        with st.expander(f"Context {i} (from {ctx.get('source', 'Unknown')})"):
                            st.text(ctx['text'][:500] + "..." if len(ctx['text']) > 500 else ctx['text'])

# Footer
st.divider()
st.caption("Built with open-source tools: Sentence Transformers, FAISS, Ollama, Streamlit")
