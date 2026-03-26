from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import chromadb
import google.generativeai as genai

from app.config import ensure_runtime_dirs, get_settings
from app.utils import chunk_text_by_words, configure_logging


LOGGER = logging.getLogger(__name__)


def _normalize_embeddings(response: dict[str, Any]) -> list[list[float]]:
    payload = response.get("embedding")
    if payload is None:
        raise ValueError("Embedding response missing `embedding` field.")
    if not payload:
        return []

    first = payload[0]
    if isinstance(first, (int, float)):
        return [payload]  # one embedding vector
    return payload  # batch embeddings


def embed_chunks(model_name: str, chunks: list[str], task_type: str) -> list[list[float]]:
    response = genai.embed_content(model=model_name, content=chunks, task_type=task_type)
    return _normalize_embeddings(response)


def _load_raw_documents(raw_dir: Path) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for json_file in sorted(raw_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as file_handle:
            documents.append(json.load(file_handle))
    return documents


def run_indexing(
    collection_name: str,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    sleep_seconds: float,
) -> None:
    settings = get_settings()
    ensure_runtime_dirs(settings)

    genai.configure(api_key=settings.gemini_api_key)
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_db_path))
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    raw_documents = _load_raw_documents(settings.raw_data_dir)
    if not raw_documents:
        LOGGER.warning("No raw transcript files found in %s", settings.raw_data_dir)
        return

    LOGGER.info("Indexing %d transcript files into collection `%s`", len(raw_documents), collection_name)

    for document in raw_documents:
        video_name = document["name"]
        full_text = document.get("text", "")
        chunks = chunk_text_by_words(full_text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            LOGGER.warning("Skipping %s because text is empty after chunking.", video_name)
            continue

        LOGGER.info("Preparing %d chunks for %s", len(chunks), video_name)
        for start_idx in range(0, len(chunks), batch_size):
            batch = chunks[start_idx : start_idx + batch_size]
            embeddings = embed_chunks(
                model_name=settings.embedding_model,
                chunks=batch,
                task_type="RETRIEVAL_DOCUMENT",
            )
            ids = [f"{video_name}_{start_idx + idx}" for idx in range(len(batch))]
            metadatas = [
                {
                    "video": video_name,
                    "video_id": document.get("video_id", ""),
                    "chunk_idx": start_idx + idx,
                }
                for idx in range(len(batch))
            ]
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=batch,
                metadatas=metadatas,
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        LOGGER.info("Indexed %d chunks from %s", len(chunks), video_name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Chroma index using Gemini embeddings.")
    parser.add_argument("--collection", default="nn_rag", help="Chroma collection name.")
    parser.add_argument("--chunk-size", type=int, default=512, help="Words per chunk.")
    parser.add_argument("--overlap", type=int, default=50, help="Word overlap between chunks.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding batch size. Defaults to EMBEDDING_BATCH_SIZE from .env (100).",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=None,
        help="Pause between embedding API calls. Defaults to EMBEDDING_SLEEP_SECONDS from .env.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    settings = get_settings()
    batch_size = args.batch_size if args.batch_size is not None else settings.embedding_batch_size
    sleep_seconds = (
        args.sleep_seconds if args.sleep_seconds is not None else settings.embedding_sleep_seconds
    )

    run_indexing(
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=batch_size,
        sleep_seconds=sleep_seconds,
    )


if __name__ == "__main__":
    main()
