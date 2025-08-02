from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import faiss
import re
import re

class TextIndex:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.qa_model = pipeline('question-answering', model='deepset/roberta-base-squad2')
        self.index = None
        self.text_chunks = []
        self.page_mapping = []  # Track which page each chunk came from

    def chunk_text(self, text, max_chunk_size=400, overlap=50):
        """Improved chunking with overlap and better sentence handling"""
        # Split by sentences first
        sentences = re.split(r'(?<=[.?!])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed max size
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap (last few words)
                words = current_chunk.split()
                overlap_words = words[-overlap//10:] if len(words) > overlap//10 else []
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk += " " + sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        return [chunk for chunk in chunks if len(chunk.strip()) > 20]

    def create_index(self, text_dict):
        """Create searchable index from PDF text"""
        self.text_chunks = []
        self.page_mapping = []
        
        # Process each page
        for page_num, text in text_dict.items():
            if text.strip():  # Only process non-empty pages
                page_chunks = self.chunk_text(text)
                self.text_chunks.extend(page_chunks)
                # Track which page each chunk came from
                self.page_mapping.extend([page_num] * len(page_chunks))
        
        if not self.text_chunks:
            raise ValueError("No valid text chunks created from PDF")
        
        # Create embeddings
        embeddings = self.embedder.encode(self.text_chunks).astype('float32')
        
        # Create FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        
        print(f"Created index with {len(self.text_chunks)} chunks from {len(text_dict)} pages")

    def query(self, question):
        """Query the document with improved context retrieval"""
        if self.index is None or not self.text_chunks:
            return "No document indexed yet."
        
        try:
            # Get question embedding
            q_emb = self.embedder.encode([question]).astype('float32')
            
            # Search for relevant chunks
            D, I = self.index.search(q_emb, k=3)  # Get top 3 most relevant chunks
            
            # Get the most relevant chunks
            relevant_chunks = []
            for i, idx in enumerate(I[0]):
                if D[0][i] < 1.5:  # Only include chunks with good similarity
                    chunk_text = self.text_chunks[idx]
                    page_num = self.page_mapping[idx]
                    relevant_chunks.append(f"[Page {page_num}] {chunk_text}")
            
            if not relevant_chunks:
                return "I couldn't find relevant information in the document to answer your question."
            
            # Combine chunks as context
            context = "\n\n".join(relevant_chunks)
            
            # Use QA model to generate answer
            result = self.qa_model(question=question, context=context)
            
            answer = result.get('answer', '').strip()
            confidence = result.get('score', 0)
            
            # If confidence is too low, provide a more helpful response
            if confidence < 0.1 or not answer:
                return f"Based on the document, I found these relevant sections but couldn't generate a specific answer:\n\n{context[:500]}..."
            
            return answer
            
        except Exception as e:
            print(f"Error in query: {str(e)}")
            return "Sorry, I encountered an error while processing your question. Please try again."

    def get_stats(self):
        """Get index statistics"""
        return {
            "total_chunks": len(self.text_chunks),
            "indexed": self.index is not None,
            "pages": len(set(self.page_mapping)) if self.page_mapping else 0
        }