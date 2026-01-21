"""
Document Processor Module
Handles ingestion and chunking of PDF, TXT, and Markdown documents.
"""

import os
import re
from typing import List, Dict
from pathlib import Path
import PyPDF2
import markdown
from docx import Document


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
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, file_path: str) -> str:
        """Extract text from a PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise Exception(f"Error reading PDF {file_path}: {str(e)}")
        return text
    
    def load_txt(self, file_path: str) -> str:
        """Load text from a TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            raise Exception(f"Error reading TXT file {file_path}: {str(e)}")
    
    def load_markdown(self, file_path: str) -> str:
        """Load and convert Markdown to plain text."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                # Convert markdown to HTML then extract text
                html = markdown.markdown(md_content)
                # Simple HTML tag removal
                text = re.sub(r'<[^>]+>', '', html)
                return text
        except Exception as e:
            raise Exception(f"Error reading Markdown file {file_path}: {str(e)}")
    
    def load_document(self, file_path: str) -> str:
        """
        Load a document based on its file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return self.load_pdf(str(file_path))
        elif extension == '.txt':
            return self.load_txt(str(file_path))
        elif extension in ['.md', '.markdown']:
            return self.load_markdown(str(file_path))
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        # Use sentence boundaries when possible for better chunking
        sentence_endings = re.compile(r'[.!?]\s+')
        
        while start < text_length:
            end = start + self.chunk_size
            
            if end >= text_length:
                # Last chunk
                chunk_text = text[start:].strip()
            else:
                # Try to break at sentence boundary
                chunk_text = text[start:end]
                last_sentence = sentence_endings.search(chunk_text[::-1])
                
                if last_sentence:
                    # Adjust end to sentence boundary
                    end = start + len(chunk_text) - last_sentence.start()
                    chunk_text = text[start:end].strip()
                else:
                    # Break at word boundary
                    last_space = chunk_text.rfind(' ')
                    if last_space > self.chunk_size * 0.5:  # Only if we're not too far
                        end = start + last_space
                        chunk_text = text[start:end].strip()
                    else:
                        chunk_text = chunk_text.strip()
            
            if chunk_text:
                chunk_metadata = {
                    'text': chunk_text,
                    'chunk_index': len(chunks),
                    'start_char': start,
                    'end_char': end,
                }
                if metadata:
                    chunk_metadata.update(metadata)
                
                chunks.append(chunk_metadata)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_directory(self, directory_path: str) -> List[Dict]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of all chunks from all documents with metadata
        """
        directory = Path(directory_path)
        all_chunks = []
        
        supported_extensions = ['.pdf', '.txt', '.md', '.markdown']
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    print(f"Processing: {file_path.name}")
                    text = self.load_document(str(file_path))
                    
                    metadata = {
                        'source': file_path.name,
                        'file_path': str(file_path),
                    }
                    
                    chunks = self.chunk_text(text, metadata)
                    all_chunks.extend(chunks)
                    print(f"  Created {len(chunks)} chunks from {file_path.name}")
                    
                except Exception as e:
                    print(f"Error processing {file_path.name}: {str(e)}")
                    continue
        
        return all_chunks
    
    def process_files(self, file_paths: List[str]) -> List[Dict]:
        """
        Process a list of specific files.
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            List of all chunks from all files with metadata
        """
        all_chunks = []
        
        for file_path in file_paths:
            try:
                print(f"Processing: {file_path}")
                text = self.load_document(file_path)
                
                metadata = {
                    'source': Path(file_path).name,
                    'file_path': file_path,
                }
                
                chunks = self.chunk_text(text, metadata)
                all_chunks.extend(chunks)
                print(f"  Created {len(chunks)} chunks from {Path(file_path).name}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        return all_chunks
