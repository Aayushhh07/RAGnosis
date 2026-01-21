"""
Vector Store Module
Manages FAISS vector database for efficient similarity search.
"""

import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Tuple


class VectorStore:
    """
    Manages vector storage and similarity search using FAISS.
    FAISS (Facebook AI Similarity Search) is an efficient open-source
    library for similarity search and clustering of dense vectors.
    """
    
    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the embeddings
            index_type: Type of FAISS index
                       - "flat": Exact search (slower but accurate)
                       - "l2": L2 distance index
                       - "cosine": Cosine similarity index (recommended)
        """
        self.dimension = dimension
        self.index_type = index_type
        
        # Create FAISS index
        if index_type == "cosine" or index_type == "flat":
            # Use inner product for cosine similarity (with normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "l2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Store metadata for each vector
        self.metadata = []
    
    def add_vectors(self, vectors: np.ndarray, metadata_list: List[Dict]):
        """
        Add vectors and their metadata to the index.
        
        Args:
            vectors: numpy array of embeddings with shape (num_vectors, dimension)
            metadata_list: List of metadata dictionaries, one per vector
        """
        if len(vectors) != len(metadata_list):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dimension {vectors.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Ensure vectors are normalized for cosine similarity
        if self.index_type == "cosine" or self.index_type == "flat":
            faiss.normalize_L2(vectors)
        
        # Add to FAISS index
        self.index.add(vectors.astype('float32'))
        
        # Store metadata
        self.metadata.extend(metadata_list)
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for the k most similar vectors.
        
        Args:
            query_vector: Query embedding with shape (dimension,)
            k: Number of results to return
            
        Returns:
            List of tuples (metadata, similarity_score) sorted by similarity
        """
        if self.index.ntotal == 0:
            return []
        
        # Reshape query vector for FAISS
        query_vector = query_vector.reshape(1, -1).astype('float32')
        
        # Normalize query vector for cosine similarity
        if self.index_type == "cosine" or self.index_type == "flat":
            faiss.normalize_L2(query_vector)
        
        # Search
        k = min(k, self.index.ntotal)  # Don't request more than available
        distances, indices = self.index.search(query_vector, k)
        
        # Format results
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.metadata):
                # For cosine similarity, distance is inner product (higher is better)
                # Convert to similarity score between 0 and 1
                if self.index_type == "cosine" or self.index_type == "flat":
                    similarity = float((distance + 1) / 2)  # Normalize to [0, 1]
                else:
                    # For L2, lower distance is better, convert to similarity
                    similarity = float(1 / (1 + distance))
                
                results.append((self.metadata[idx], similarity))
        
        return results
    
    def save(self, directory: str):
        """
        Save the index and metadata to disk.
        
        Args:
            directory: Directory path to save the index
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, "faiss.index")
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
        # Save configuration
        config_path = os.path.join(directory, "config.pkl")
        with open(config_path, 'wb') as f:
            pickle.dump({
                'dimension': self.dimension,
                'index_type': self.index_type,
                'num_vectors': self.index.ntotal
            }, f)
    
    @classmethod
    def load(cls, directory: str) -> 'VectorStore':
        """
        Load the index and metadata from disk.
        
        Args:
            directory: Directory path containing the saved index
            
        Returns:
            Loaded VectorStore instance
        """
        # Load configuration
        config_path = os.path.join(directory, "config.pkl")
        with open(config_path, 'rb') as f:
            config = pickle.load(f)
        
        # Create instance
        store = cls(config['dimension'], config['index_type'])
        
        # Load FAISS index
        index_path = os.path.join(directory, "faiss.index")
        store.index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = os.path.join(directory, "metadata.pkl")
        with open(metadata_path, 'rb') as f:
            store.metadata = pickle.load(f)
        
        return store
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        return {
            'num_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type
        }
