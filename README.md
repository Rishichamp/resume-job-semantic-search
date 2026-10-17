# Resume ↔ Job Semantic Search

[![tests](https://github.com/Rishichamp/resume-job-semantic-search/actions/workflows/tests.yml/badge.svg)](https://github.com/Rishichamp/resume-job-semantic-search/actions/workflows/tests.yml)

A two-way semantic search engine that matches **resumes to job postings and job postings to resumes** by meaning rather than keywords. Personal information is redacted **before** anything is embedded, so names, emails and phone numbers never end up in the vector index.

## Features

- **Two collections, one embedding space:** resumes and job postings are indexed in separate ChromaDB collections with the same embedding model, so either side can be the query.
- **Swappable embeddings:** OpenAI (`text-embedding-3-small`) or a local sentence-transformers model, chosen with one setting (`EMBEDDING_PROVIDER`). No other code changes.
- **PII redaction layer:** regex patterns for emails, phone numbers, addresses, IDs and dates of birth, plus optional spaCy name detection, applied before embedding.
- **Similarity threshold and top-k** filtering, with match scores returned.
- **Query logging** (JSONL) for later analysis, which can be switched off.
- **REST API** with FastAPI and automatic OpenAPI docs.

## Tech stack

Python 3.11 · FastAPI · ChromaDB · OpenAI embeddings / sentence-transformers · Pydantic · pytest

## Project structure

```
app/
  main.py            FastAPI app and endpoints
  embedder.py        single entry point for all embedding calls (OpenAI or local)
  pii_redaction.py   regex + optional spaCy redaction
  vector_store.py    ChromaDB collections for resumes and jobs
  models.py          request/response schemas
  config.py          settings from .env
sample_data/load_samples.py   loads realistic demo resumes and jobs
tests/                        17 tests
run.py                        starts the server
```

## Getting started

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env         # macOS/Linux: cp .env.example .env, then add your OPENAI_API_KEY
python run.py                  # http://127.0.0.1:8000/docs
python sample_data/load_samples.py   # optional: load demo data
```

To run without an API key, set `EMBEDDING_PROVIDER=local` in `.env`.

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/resumes/index` | Index a resume |
| POST | `/jobs/index` | Index a job posting |
| POST | `/search/resumes` | Find matching resumes for a job |
| POST | `/search/jobs` | Find matching jobs for a resume |
| DELETE | `/resumes/{resume_id}` | Remove a resume |
| DELETE | `/jobs/{job_id}` | Remove a job posting |
| GET | `/stats` | Index statistics |
| GET | `/health` | Health check |

## Tests

```bash
pytest tests/ -q
```

17 tests covering redaction, indexing, search and the API. External embedding calls are mocked, so no API key is needed.

## Notes

- Changing the embedding model after indexing makes old vectors incompatible; re-index after switching.
- Built as part of Module 2 of a Generative AI engineering programme.

## Author

**Rishi Singh** — [GitHub](https://github.com/Rishichamp)
