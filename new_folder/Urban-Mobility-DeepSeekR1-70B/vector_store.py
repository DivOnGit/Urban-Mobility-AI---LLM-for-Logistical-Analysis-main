from __future__ import annotations

import logging
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from config import settings
from embeddings import build_embeddings

LOGGER = logging.getLogger(__name__)


class FaissVectorStore:
    def __init__(self, index_path: Path = settings.faiss_index_path) -> None:
        self.index_path = index_path
        self.embeddings = build_embeddings()

    def build(self, documents: list[Document]) -> FAISS:
        if not documents:
            raise ValueError("Cannot build a FAISS index without documents.")
        LOGGER.info("Building FAISS index with %d chunks", len(documents))
        store = FAISS.from_documents(documents, self.embeddings)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        store.save_local(str(self.index_path))
        return store

    def load(self) -> FAISS:
        if not self.index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}. Run ingestion first.")
        return FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def load_or_build(self, documents: list[Document] | None = None) -> FAISS:
        if self.index_path.exists():
            return self.load()
        if documents is None:
            raise FileNotFoundError("No FAISS index exists and no documents were supplied to build one.")
        return self.build(documents)
