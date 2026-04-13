import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticRetriever:
    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.embeddings = None

    def build_index(self, chunks):
        """
        Build FAISS index from text chunks.
        """
        if not chunks:
            raise ValueError("No chunks provided to build the index.")

        self.chunks = chunks

        embeddings = self.model.encode(chunks, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")

        # Normalize for cosine similarity using inner product
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        self.index = index
        self.embeddings = embeddings

    def retrieve(self, query: str, top_k: int = 3):
        """
        Retrieve top-k most similar chunks for a query.
        Returns list of dicts with chunk text, score, and index.
        """
        if self.index is None:
            raise ValueError("FAISS index is not built yet.")

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append({
                    "chunk_id": int(idx),
                    "score": float(score),
                    "text": self.chunks[idx]
                })

        return results