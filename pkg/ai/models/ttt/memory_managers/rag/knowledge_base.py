import faiss  # TODO: add support for GPU based
from sentence_transformers import SentenceTransformer


class KnowledgeBase:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.docs = []  # store original texts
        self.index = None  # FAISS index

    def add_documents(self, docs: list[str]):
        """Add documents to the knowledge base."""
        embeddings = self.embedder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
        self.docs.extend(docs)

        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dim)  # cosine similarity (since normalized)
        self.index.add(embeddings)

    def query(self, question: str, top_k: int = 3) -> list[str]:
        """Retrieve top-k relevant docs for the query."""
        q_emb = self.embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
        scores, idx = self.index.search(q_emb, top_k)
        return [self.docs[i] for i in idx[0] if i < len(self.docs)]
