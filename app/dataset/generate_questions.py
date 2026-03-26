from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
from typing import Any, Literal

import chromadb
import google.generativeai as genai
from pydantic import BaseModel, Field, ValidationError

from app.config import ensure_runtime_dirs, get_settings
from app.utils import MinIntervalLimiter, configure_logging


LOGGER = logging.getLogger(__name__)

GENERATOR_PROMPT = """You are building a golden evaluation dataset for a RAG system
about neural networks and deep learning.

For the provided transcript chunk, generate exactly 3 question-answer pairs:
1) factual
2) conceptual
3) applied

Rules:
- Return valid JSON only.
- Keep answers self-contained and clear without requiring the chunk to be visible.
- Use only information from the chunk.
- difficulty must be one of: easy, medium, hard.
- requires_cross_video should be true only if answer needs another video/chunk.
"""


class QAPair(BaseModel):
    question: str = Field(min_length=5)
    ground_truth_answer: str = Field(min_length=5)
    question_type: Literal["factual", "conceptual", "applied", "comparative"]
    difficulty: Literal["easy", "medium", "hard"]
    requires_cross_video: bool


class QABatch(BaseModel):
    pairs: list[QAPair] = Field(min_length=3, max_length=3)


def _build_generation_model(model_name: str) -> genai.GenerativeModel:
    try:
        config = genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=QABatch.model_json_schema(),
        )
        return genai.GenerativeModel(model_name=model_name, generation_config=config)
    except Exception as error:
        LOGGER.warning(
            "Structured response schema unavailable in current SDK (%s). Falling back to JSON mode only.",
            error,
        )
        config = genai.GenerationConfig(response_mime_type="application/json")
        return genai.GenerativeModel(model_name=model_name, generation_config=config)


def _parse_batch(raw_json: str) -> QABatch:
    try:
        batch = QABatch.model_validate_json(raw_json)
    except ValidationError as error:
        raise RuntimeError(f"Could not parse generated QA JSON: {error}") from error
    seen_types = {pair.question_type for pair in batch.pairs}
    required_types = {"factual", "conceptual", "applied"}
    if required_types - seen_types:
        missing = ", ".join(sorted(required_types - seen_types))
        raise RuntimeError(f"Generated batch is missing required question types: {missing}")
    return batch


def generate_qa_for_chunk(
    model: genai.GenerativeModel,
    limiter: MinIntervalLimiter,
    chunk: str,
    video_name: str,
    chunk_id: str,
    retries: int = 2,
) -> list[dict[str, Any]]:
    prompt = f"{GENERATOR_PROMPT}\n\nVideo: {video_name}\nChunk id: {chunk_id}\nChunk:\n{chunk}"

    last_error: Exception | None = None
    for _ in range(retries + 1):
        try:
            limiter.wait()
            response = model.generate_content(prompt)
            batch = _parse_batch(response.text or "")
            break
        except Exception as error:
            last_error = error
    else:
        raise RuntimeError(f"Failed to generate valid QA JSON for {video_name}:{chunk_id}") from last_error

    records: list[dict[str, Any]] = []
    for pair in batch.pairs:
        digest = hashlib.md5(pair.question.encode("utf-8")).hexdigest()[:8]
        records.append(
            {
                "id": f"q_{video_name}_{chunk_id}_{pair.question_type}_{digest}",
                "question": pair.question,
                "ground_truth_answer": pair.ground_truth_answer,
                "source_chunk": chunk,
                "video_id": video_name,
                "question_type": pair.question_type,
                "difficulty": pair.difficulty,
                "requires_cross_video": pair.requires_cross_video,
                "verified_by_human": False,
            }
        )
    return records


def run_question_generation(
    collection_name: str,
    output_file: str,
    min_words: int,
    max_chunks: int | None,
    shuffle: bool,
) -> int:
    settings = get_settings()
    ensure_runtime_dirs(settings)
    genai.configure(api_key=settings.gemini_api_key)

    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_db_path))
    try:
        collection = chroma_client.get_collection(collection_name)
    except Exception as error:
        raise RuntimeError(
            f"Collection `{collection_name}` was not found. Build the index first."
        ) from error

    payload = collection.get(include=["documents", "metadatas"])
    pairs = list(zip(payload.get("documents", []), payload.get("metadatas", [])))
    if shuffle:
        random.shuffle(pairs)
    if max_chunks is not None:
        pairs = pairs[:max_chunks]

    model = _build_generation_model(settings.generation_model)
    limiter = MinIntervalLimiter(settings.question_generation_sleep_seconds)

    generated: list[dict[str, Any]] = []
    for doc, metadata in pairs:
        if len(doc.split()) < min_words:
            continue
        video_name = metadata.get("video", "unknown")
        chunk_id = str(metadata.get("chunk_idx", "0"))
        try:
            records = generate_qa_for_chunk(
                model=model,
                limiter=limiter,
                chunk=doc,
                video_name=video_name,
                chunk_id=chunk_id,
            )
            generated.extend(records)
        except Exception:
            LOGGER.exception("Skipping chunk %s:%s due to generation error.", video_name, chunk_id)

    output_path = settings.project_root / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file_handle:
        for record in generated:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOGGER.info("Generated %d QA records -> %s", len(generated), output_path)
    return len(generated)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate golden QA dataset with Gemini structured output.")
    parser.add_argument("--collection", default="nn_rag", help="Chroma collection name.")
    parser.add_argument(
        "--output-file",
        default="data/golden_unverified.jsonl",
        help="Relative path for generated JSONL output.",
    )
    parser.add_argument("--min-words", type=int, default=80, help="Skip chunks shorter than this word count.")
    parser.add_argument("--max-chunks", type=int, default=None, help="Optional limit on processed chunks.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle chunks before processing.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    run_question_generation(
        collection_name=args.collection,
        output_file=args.output_file,
        min_words=args.min_words,
        max_chunks=args.max_chunks,
        shuffle=args.shuffle,
    )


if __name__ == "__main__":
    main()
