from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_recall, faithfulness

from app.config import get_settings
from app.rag import GeminiRAGPipeline
from app.utils import configure_logging


LOGGER = logging.getLogger(__name__)


def _load_verified_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("verified_by_human"):
                records.append(item)
    return records


def _build_eval_dataset(records: list[dict[str, Any]], rag_pipeline: GeminiRAGPipeline) -> Dataset:
    eval_rows: list[dict[str, Any]] = []
    for item in records:
        rag_result = rag_pipeline.query(item["question"])
        eval_rows.append(
            {
                "question": item["question"],
                "answer": rag_result["answer"],
                "contexts": [chunk["text"] for chunk in rag_result["source_chunks"]],
                "ground_truth": item["ground_truth_answer"],
            }
        )
    return Dataset.from_list(eval_rows)


def run_evaluation(
    golden_path: str,
    collection_name: str,
    limit: int | None,
    output_csv: str | None,
) -> None:
    settings = get_settings()
    verified_records = _load_verified_records(Path(golden_path))
    if not verified_records:
        raise RuntimeError(
            "No verified records found. Mark some `verified_by_human=true` in golden dataset first."
        )
    if limit is not None:
        verified_records = verified_records[:limit]

    rag_pipeline = GeminiRAGPipeline(collection_name=collection_name)
    try:
        dataset = _build_eval_dataset(verified_records, rag_pipeline)
    finally:
        rag_pipeline.close()

    gemini_llm = ChatGoogleGenerativeAI(
        model=settings.generation_model,
        google_api_key=settings.gemini_api_key,
        temperature=0,
    )
    ragas_llm = LangchainLLMWrapper(gemini_llm)

    for metric in (faithfulness, answer_relevancy, context_recall):
        metric.llm = ragas_llm

    scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_recall])
    scores_df = scores.to_pandas()
    print(scores_df)

    if output_csv:
        csv_path = Path(output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        scores_df.to_csv(csv_path, index=False)
        LOGGER.info("Saved evaluation results to %s", csv_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate RAG performance with RAGAS + Gemini judge.")
    parser.add_argument("--golden-path", default="data/golden.jsonl", help="Path to verified golden JSONL.")
    parser.add_argument("--collection", default="nn_rag", help="Chroma collection name.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N verified records.")
    parser.add_argument("--output-csv", default=None, help="Optional path to save score table as CSV.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    run_evaluation(
        golden_path=args.golden_path,
        collection_name=args.collection,
        limit=args.limit,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
