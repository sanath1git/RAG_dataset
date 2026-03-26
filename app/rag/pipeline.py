from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
import sqlite3
from typing import Any

import chromadb
import google.generativeai as genai

from app.config import ensure_runtime_dirs, get_settings
from app.utils import MinIntervalLimiter, configure_logging


LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert on neural networks and deep learning.
Answer the question using ONLY the provided context chunks.
If the answer is not in the context, say so clearly.
Always mention which source chunk(s) and video(s) you used."""


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict[str, Any]
    distance: float


class GeminiRAGPipeline:
    def __init__(
        self,
        collection_name: str = "nn_rag",
        top_k: int = 10,
        rerank_top_n: int = 3,
    ) -> None:
        self.settings = get_settings()
        ensure_runtime_dirs(self.settings)

        genai.configure(api_key=self.settings.gemini_api_key)
        self.model = genai.GenerativeModel(self.settings.generation_model)

        self.chroma_client = chromadb.PersistentClient(path=str(self.settings.chroma_db_path))
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

        self.top_k = top_k
        self.rerank_top_n = rerank_top_n
        self.generation_limiter = MinIntervalLimiter(self.settings.rag_generation_sleep_seconds)

        self.cache_conn = sqlite3.connect(str(self.settings.cache_db_path))
        self.cache_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                hash TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.cache_conn.commit()

    def close(self) -> None:
        self.cache_conn.close()

    def _get_query_embedding(self, query: str) -> list[float]:
        result = genai.embed_content(
            model=self.settings.embedding_model,
            content=query,
            task_type="RETRIEVAL_QUERY",
        )
        embedding = result.get("embedding")
        if not embedding:
            raise RuntimeError("Query embedding call returned an empty embedding.")
        return embedding

    def retrieve_chunks(self, query: str) -> list[RetrievedChunk]:
        embedding = self._get_query_embedding(query)
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=self.top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for doc, metadata, distance in zip(documents, metadatas, distances):
            retrieved.append(RetrievedChunk(text=doc, metadata=metadata, distance=float(distance)))
        return retrieved

    def _rerank_with_gemini(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []

        chunk_texts = [f"[{idx + 1}] {chunk.text[:500]}" for idx, chunk in enumerate(chunks)]
        prompt = f"""Query: {query}

Rate each chunk's relevance to the query on a scale of 1 to 10.
Return ONLY a JSON array of numbers.

Chunks:
{chr(10).join(chunk_texts)}"""

        self.generation_limiter.wait()
        response = self.model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(response_mime_type="application/json"),
        )

        try:
            parsed_scores = json.loads((response.text or "").strip())
            if not isinstance(parsed_scores, list):
                raise ValueError("JSON response is not a list.")
        except Exception:
            LOGGER.warning("Reranker returned invalid JSON. Falling back to retrieval ordering.")
            return chunks[: self.rerank_top_n]

        scored_chunks: list[tuple[RetrievedChunk, float]] = []
        for idx, chunk in enumerate(chunks):
            score = parsed_scores[idx] if idx < len(parsed_scores) else 0
            try:
                score = float(score)
            except Exception:
                score = 0.0
            scored_chunks.append((chunk, score))

        ranked = sorted(scored_chunks, key=lambda item: item[1], reverse=True)
        return [item[0] for item in ranked[: self.rerank_top_n]]

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {chunk.metadata.get('video', 'unknown')}, chunk {chunk.metadata.get('chunk_idx', 'n/a')}]\n"
            f"{chunk.text}"
            for chunk in chunks
        )

    def _generate_answer(self, question: str, context: str) -> str:
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {question}"
        self.generation_limiter.wait()
        response = self.model.generate_content(prompt)
        return (response.text or "").strip()

    def query(self, question: str, use_cache: bool = True) -> dict[str, Any]:
        cache_key = hashlib.md5(question.encode("utf-8")).hexdigest()

        if use_cache:
            row = self.cache_conn.execute("SELECT response FROM cache WHERE hash=?", (cache_key,)).fetchone()
            if row:
                payload = json.loads(row[0])
                payload["cached"] = True
                return payload

        retrieved = self.retrieve_chunks(question)
        reranked = self._rerank_with_gemini(question, retrieved)
        context = self._build_context(reranked)
        answer = self._generate_answer(question, context)

        result = {
            "question": question,
            "answer": answer,
            "cached": False,
            "source_chunks": [
                {
                    "text": chunk.text,
                    "meta": chunk.metadata,
                    "distance": chunk.distance,
                }
                for chunk in reranked
            ],
        }

        self.cache_conn.execute(
            "INSERT OR REPLACE INTO cache(hash, response) VALUES (?, ?)",
            (cache_key, json.dumps(result)),
        )
        self.cache_conn.commit()
        return result


def query_rag(question: str, collection_name: str = "nn_rag") -> dict[str, Any]:
    pipeline = GeminiRAGPipeline(collection_name=collection_name)
    try:
        return pipeline.query(question)
    finally:
        pipeline.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run query-time Gemini RAG pipeline.")
    parser.add_argument("question", help="Natural-language question for the RAG system.")
    parser.add_argument("--collection", default="nn_rag", help="Chroma collection name.")
    parser.add_argument("--top-k", type=int, default=10, help="Retrieved chunks before reranking.")
    parser.add_argument("--rerank-top-n", type=int, default=3, help="Chunks kept after reranking.")
    parser.add_argument("--no-cache", action="store_true", help="Disable SQLite cache for this query.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    pipeline = GeminiRAGPipeline(
        collection_name=args.collection,
        top_k=args.top_k,
        rerank_top_n=args.rerank_top_n,
    )
    try:
        result = pipeline.query(args.question, use_cache=not args.no_cache)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
