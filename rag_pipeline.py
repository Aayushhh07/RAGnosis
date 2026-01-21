"""
RAG Pipeline Module
Implements the complete Retrieval-Augmented Generation pipeline.
"""

from typing import List, Dict, Optional
import ollama
from embeddings import EmbeddingGenerator
from vector_store import VectorStore
from guardrails import Guardrails


class RAGPipeline:
    """
    Complete RAG pipeline implementation:
    1. Query embedding
    2. Vector similarity search
    3. Context retrieval
    4. Prompt augmentation
    5. Response generation with guardrails
    """
    
    def __init__(self, 
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 llm_model: str = "llama3",
                 vector_store: Optional[VectorStore] = None,
                 min_confidence: float = 0.5):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model: Sentence Transformers model name
            llm_model: Ollama model name (e.g., "llama3", "mistral")
            vector_store: Pre-initialized vector store (optional)
            min_confidence: Minimum confidence threshold for retrieval
        """
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.llm_model = llm_model
        self.guardrails = Guardrails(min_confidence=min_confidence)
        
        if vector_store is None:
            dimension = self.embedding_generator.embedding_dimension
            self.vector_store = VectorStore(dimension=dimension, index_type="cosine")
        else:
            self.vector_store = vector_store
    
    def build_prompt(self, query: str, contexts: List[Dict], max_context_length: int = 2000) -> str:
        """
        Build augmented prompt with retrieved contexts.
        
        Args:
            query: User query
            contexts: List of context dictionaries with 'text' and metadata
            max_context_length: Maximum characters of context to include
            
        Returns:
            Formatted prompt string
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
    
    def retrieve_contexts(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Retrieve relevant contexts for a query.
        
        Args:
            query: User query
            top_k: Number of contexts to retrieve
            
        Returns:
            List of (metadata, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=top_k)
        
        return results
    
    def generate_response(self, query: str, top_k: int = 5, 
                         use_guardrails: bool = True) -> Dict:
        """
        Complete RAG flow: retrieve contexts and generate response.
        
        Args:
            query: User query
            top_k: Number of contexts to retrieve
            use_guardrails: Whether to apply guardrails
            
        Returns:
            Dictionary with response, sources, and metadata
        """
        # Step 1: Retrieve contexts
        retrieval_results = self.retrieve_contexts(query, top_k=top_k)
        
        if not retrieval_results:
            return {
                'response': "I couldn't find any relevant documents to answer your question. Please ensure documents have been loaded.",
                'sources': [],
                'confidence': 0.0,
                'validation': {
                    'is_valid': False,
                    'confidence_passed': False
                }
            }
        
        # Extract contexts and scores
        contexts = [meta for meta, score in retrieval_results]
        scores = [score for meta, score in retrieval_results]
        
        # Step 2: Apply confidence filtering
        if use_guardrails:
            filtered_results = self.guardrails.filter_low_confidence(retrieval_results)
            if not filtered_results:
                return {
                    'response': f"I couldn't find documents with sufficient relevance to answer your question. The best match had a confidence score of {max(scores):.2f}, which is below the threshold of {self.guardrails.min_confidence:.2f}.",
                    'sources': [ctx.get('source', 'Unknown') for ctx in contexts],
                    'confidence': max(scores),
                    'validation': {
                        'is_valid': False,
                        'confidence_passed': False,
                        'average_confidence': sum(scores) / len(scores),
                        'max_confidence': max(scores)
                    }
                }
            contexts = [meta for meta, score in filtered_results]
            scores = [score for meta, score in filtered_results]
        
        # Step 3: Build augmented prompt
        prompt = self.build_prompt(query, contexts)
        
        # Step 4: Generate response using Ollama
        try:
            # Try chat API first (preferred), fallback to generate
            try:
                response = ollama.chat(
                    model=self.llm_model,
                    messages=[
                        {'role': 'user', 'content': prompt}
                    ],
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
                    options={
                        'temperature': 0.1,
                        'top_p': 0.9,
                    }
                )
                response_text = response.get('response', str(response))
        except Exception as e:
            response_text = f"Error generating response: {str(e)}. Please ensure Ollama is running and the model '{self.llm_model}' is installed. Run: ollama pull {self.llm_model}"
        
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
    
    def add_documents(self, chunks: List[Dict]):
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and metadata
        """
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embedding_generator.generate_embeddings(texts)
        
        # Add to vector store
        self.vector_store.add_vectors(embeddings, chunks)
        
        print(f"Added {len(chunks)} chunks to vector store")
