"""
Configuration for the Resume-to-Job Semantic Search Engine.
"""
import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    # ── LLM / Embedding provider ──────────────────────────────────
    LLM_PROVIDER: str        = os.getenv("LLM_PROVIDER", "openai")
    OPENAI_API_KEY: str      = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str      = os.getenv("GEMINI_API_KEY", "")

    # Embedding model — MUST stay consistent once you index data
    # openai: text-embedding-3-small | text-embedding-3-large
    # local:  all-MiniLM-L6-v2       | all-mpnet-base-v2
    EMBEDDING_PROVIDER: str  = os.getenv("EMBEDDING_PROVIDER", "openai")
    EMBEDDING_MODEL: str     = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM: int       = int(os.getenv("EMBEDDING_DIM", "1536"))

    # ── Vector store ──────────────────────────────────────────────
    CHROMA_PATH: str         = os.getenv("CHROMA_PATH", "./chroma_db")
    RESUME_COLLECTION: str   = "resumes"
    JOB_COLLECTION: str      = "job_postings"

    # ── Search settings ───────────────────────────────────────────
    TOP_K: int               = int(os.getenv("TOP_K", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.65"))

    # ── Privacy / MLOps ───────────────────────────────────────────
    # When True, PII fields are redacted before storing in vector DB
    REDACT_PII: bool         = os.getenv("REDACT_PII", "true").lower() == "true"
    # Log all queries for monitoring (disable for strict privacy)
    ENABLE_QUERY_LOGGING: bool = os.getenv("ENABLE_QUERY_LOGGING", "true").lower() == "true"
    QUERY_LOG_PATH: str      = os.getenv("QUERY_LOG_PATH", "./logs/queries.jsonl")

    # ── App ───────────────────────────────────────────────────────
    APP_HOST: str            = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int            = int(os.getenv("APP_PORT", "8000"))
    DEBUG: bool              = os.getenv("DEBUG", "false").lower() == "true"

settings = Settings()
