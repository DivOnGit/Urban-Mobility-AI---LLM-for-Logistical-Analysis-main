from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag import UrbanMobilityRAG
from sample_queries import SAMPLE_QUERIES
from utils import configure_logging, ensure_directories, write_json


@dataclass
class Evaluator:
    output_path: Path = Path("results/evaluation_responses.json")

    def run(self) -> None:
        configure_logging()
        ensure_directories()
        rag = UrbanMobilityRAG()
        results = []
        for item in SAMPLE_QUERIES:
            answer = rag.answer(item["query"])
            results.append({"category": item["category"], "query": item["query"], "response": answer})
        write_json(self.output_path, {"model": "LLaMA-4-Maverick-17B", "results": results})


if __name__ == "__main__":
    Evaluator().run()
