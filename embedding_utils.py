from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import faiss

class TextIndex:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.qa_model = pipeline('question-answering', model='deepset/roberta-base-squad2')
        self.index = None
        self.text_chunks = []

    def chunk_text(self, text, max_chunk_size=500):
        import re
        sentences = re.split(r'(?<=[.?!])\s+', text)
        chunks, current_chunk = [], ""
        for sent in sentences:
            if len(current_chunk) + len(sent) < max_chunk_size:
                current_chunk += " " + sent
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sent
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def create_index(self, text_dict):
        # Concatenate all text pages into one string
        full_text = "\n\n".join(text_dict.values())
        # Chunk into paragraphs or smaller chunks
        self.text_chunks = self.chunk_text(full_text)

        # Embed chunks
        embeddings = self.embedder.encode(self.text_chunks).astype('float32')

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def query(self, question):
        if self.index is None or not self.text_chunks:
            return "No document indexed yet."

        q_emb = self.embedder.encode([question]).astype('float32')
        D, I = self.index.search(q_emb, k=5)  # retrieve top 5 chunks

        # Concatenate top chunks as context
        context = "\n\n".join([self.text_chunks[i] for i in I[0]])

        # Use QA model
        result = self.qa_model(question=question, context=context)
        return result.get('answer', 'Sorry, I could not find an answer.')
