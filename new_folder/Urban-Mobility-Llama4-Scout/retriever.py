from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document

from config import settings
from dataset_loader import TaxiDatasetLoader
from vector_store import FaissVectorStore


@dataclass
class UrbanMobilityRetriever:
    top_k: int = settings.top_k

    def __post_init__(self) -> None:
        loader = TaxiDatasetLoader()
        store = FaissVectorStore()
        documents = None
        if not settings.faiss_index_path.exists():
            documents = loader.build_documents()
        self.vector_store = store.load_or_build(documents)

    def search(self, query: str) -> list[Document]:
        return self.vector_store.similarity_search(query, k=self.top_k)

    def context(self, query: str) -> str:
        docs = self.search(query)
        return "\n\n".join(
            f"[Source row {doc.metadata.get('row_id', 'unknown')}] {doc.page_content}" for doc in docs
        )
