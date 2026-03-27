# Gemini RAG Dataset Pipeline

Production-style, end-to-end implementation of your Gemini-first architecture:

1. Transcript ingestion and normalization (with Hindi fallback translation)
2. Embedding + local Chroma indexing (`models/gemini-embedding-001`)
3. Query-time RAG with Gemini reranking + SQLite cache
4. Golden dataset generation with structured JSON output
5. RAGAS evaluation with Gemini as judge

## Project Structure

```text
app/
  config/
    settings.py
  utils/
    logging_utils.py
    rate_limiter.py
    text_chunking.py
  ingestion/
    fetch_transcripts.py
  embedding/
    build_index.py
  rag/
    pipeline.py
  dataset/
    generate_questions.py
  eval/
    run_ragas.py
```

## 1) Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set:

```bash
GEMINI_API_KEY=your_key_from_aistudio.google.com
GENERATION_MODEL=models/gemini-flash-lite-latest
EMBEDDING_MODEL=models/gemini-embedding-001
```

## 2) Ingest transcripts

Default curated list (3Blue1Brown + Hindi channels with fallback translation):

```powershell
python -m app.ingestion.fetch_transcripts
```

Custom list:

```powershell
python -m app.ingestion.fetch_transcripts --video my_video=YOUTUBE_ID --video another=YOUTUBE_ID
```

## 3) Build embedding index

```powershell
python -m app.embedding.build_index --collection nn_rag --chunk-size 512 --overlap 50
```

This uses:

- `task_type=RETRIEVAL_DOCUMENT` at indexing time
- `models/gemini-embedding-001`
- idempotent `upsert` into local Chroma

## 4) Query the RAG pipeline

```powershell
python -m app.rag.pipeline "What is backpropagation and why is it useful?"
```

This pipeline does:

1. `RETRIEVAL_QUERY` embedding
2. Chroma retrieval
3. Gemini reranking (JSON score array)
4. final answer generation from top chunks
5. SQLite caching (`cache.db`)

## 5) Generate unverified golden dataset

```powershell
python -m app.dataset.generate_questions --collection nn_rag --output-file data/golden_unverified.jsonl --shuffle
```

Each chunk generates exactly 3 QA pairs (factual, conceptual, applied).

## 6) Human verification

Move reviewed records into `data/golden.jsonl` and set:

```json
{"verified_by_human": true}
```

Only verified rows are used in evaluation.

## 7) Run RAGAS evaluation

```powershell
python -m app.eval.run_ragas --golden-path data/golden.jsonl --collection nn_rag --output-csv data/eval_scores.csv
```

Metrics:

- `faithfulness`
- `answer_relevancy`
- `context_recall`

All metrics are wired to Gemini judge via `langchain-google-genai`.

## Operational Notes

- Keep `.env` out of source control.
- Rate pacing is configurable in `.env` to match your free-tier budget.
- This code is designed to be rerunnable safely:
  - transcript files can be overwritten with `--overwrite`
  - embedding index uses `upsert`
  - query layer caches answers by question hash
