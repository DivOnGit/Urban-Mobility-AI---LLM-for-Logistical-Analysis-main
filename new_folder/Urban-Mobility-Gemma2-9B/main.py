from __future__ import annotations

import argparse

from dataset_loader import TaxiDatasetLoader
from rag import UrbanMobilityRAG
from utils import configure_logging, ensure_directories
from vector_store import FaissVectorStore


def build_index() -> None:
    loader = TaxiDatasetLoader()
    documents = loader.build_documents()
    FaissVectorStore().build(documents)
    print(f"Built FAISS index with {len(documents)} chunks.")


def chat() -> None:
    rag = UrbanMobilityRAG()
    print("Urban Mobility AI ready. Type 'exit' to quit.")
    while True:
        question = input("Query: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        print(rag.answer(question))


def main() -> None:
    configure_logging()
    ensure_directories()
    parser = argparse.ArgumentParser(description="Urban Mobility AI pipeline for Gemma-2-9B-IT.")
    parser.add_argument("--build-index", action="store_true", help="Build the FAISS index from the configured CSV.")
    parser.add_argument("--chat", action="store_true", help="Start an interactive CLI chat.")
    args = parser.parse_args()
    if args.build_index:
        build_index()
    if args.chat or not args.build_index:
        chat()


if __name__ == "__main__":
    main()
