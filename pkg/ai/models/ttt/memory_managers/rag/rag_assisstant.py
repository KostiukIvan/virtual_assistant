from pkg.ai.models.ttt.memory_managers.rag.knowledge_base import KnowledgeBase


class RAGAssistant:
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def retrieve(self, user_message: str, top_k: int = 3) -> str:
        docs = self.kb.query(user_message, top_k=top_k)
        if not docs:
            return ""
        context_block = "\n".join([f"- {doc}" for doc in docs])
        return f"[Relevant company information:]\n{context_block}"
