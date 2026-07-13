from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

LOGGER = logging.getLogger(__name__)


@dataclass
class TaxiDatasetLoader:
    csv_path: Path = settings.data_path
    chunk_size: int = settings.chunk_size
    chunk_overlap: int = settings.chunk_overlap

    def load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"NYC Taxi CSV not found at {self.csv_path}. Place a TLC CSV there or set DATA_PATH."
            )
        LOGGER.info("Loading taxi dataset from %s", self.csv_path)
        return pd.read_csv(self.csv_path, low_memory=False)

    def preprocess(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        df.columns = [column.strip().lower() for column in df.columns]
        datetime_columns = [
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "lpep_pickup_datetime",
            "lpep_dropoff_datetime",
            "pickup_datetime",
            "dropoff_datetime",
        ]
        for column in datetime_columns:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors="coerce")

        pickup_column = next((c for c in datetime_columns if "pickup" in c and c in df.columns), None)
        if pickup_column:
            df["pickup_hour"] = df[pickup_column].dt.hour
            df["pickup_day"] = df[pickup_column].dt.day_name()
            df["is_weekend"] = df["pickup_day"].isin(["Saturday", "Sunday"])

        if "trip_distance" in df.columns:
            df["trip_distance"] = pd.to_numeric(df["trip_distance"], errors="coerce").clip(lower=0)
        if "passenger_count" in df.columns:
            df["passenger_count"] = pd.to_numeric(df["passenger_count"], errors="coerce").fillna(0)
        if "total_amount" in df.columns:
            df["total_amount"] = pd.to_numeric(df["total_amount"], errors="coerce")

        df = df.dropna(axis=0, how="all").reset_index(drop=True)
        LOGGER.info("Preprocessed %d taxi records", len(df))
        return df

    def to_documents(self, frame: pd.DataFrame) -> list[Document]:
        documents: list[Document] = []
        for index, row in frame.iterrows():
            metadata = {
                "row_id": int(index),
                "pickup_hour": int(row["pickup_hour"]) if "pickup_hour" in row and pd.notna(row["pickup_hour"]) else None,
                "pickup_day": str(row["pickup_day"]) if "pickup_day" in row and pd.notna(row["pickup_day"]) else None,
                "trip_distance": float(row["trip_distance"]) if "trip_distance" in row and pd.notna(row["trip_distance"]) else None,
            }
            text = self._row_to_text(row)
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", ", ", " "],
        )
        return splitter.split_documents(documents)

    def build_documents(self) -> list[Document]:
        frame = self.preprocess(self.load())
        return self.chunk_documents(self.to_documents(frame))

    @staticmethod
    def _row_to_text(row: pd.Series) -> str:
        fields = []
        for key, value in row.items():
            if pd.notna(value):
                fields.append(f"{key}: {value}")
        return "NYC taxi trip record. " + "; ".join(fields)
