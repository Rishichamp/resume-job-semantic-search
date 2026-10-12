"""
VectorStoreManager — manages Chroma collections for resumes and job postings.

Two separate collections:
  - resumes:      indexed resume texts (PII-redacted embedding text + metadata)
  - job_postings: indexed job description texts + metadata

Both collections use the same embedding model (enforced by the Embedder singleton).
"""

import json
import logging
import os
from typing import Optional
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from app.embedder import get_embedder
from app.models import ResumeMatch, JobMatch

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """
    Manages ChromaDB collections for semantic search.

    Responsibilities:
      - Create/load collections on startup
      - Add resume and job embeddings
      - Perform similarity search with optional metadata filtering
      - Return typed result models
    """

    def __init__(self):
        Path(settings.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.embedder = get_embedder()
        self._resume_col  = None
        self._job_col     = None
        logger.info(f"VectorStoreManager initialised at {settings.CHROMA_PATH}")

    # ── collection accessors ──────────────────────────────────────────────────

    @property
    def resume_collection(self):
        if self._resume_col is None:
            self._resume_col = self._client.get_or_create_collection(
                name=settings.RESUME_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        return self._resume_col

    @property
    def job_collection(self):
        if self._job_col is None:
            self._job_col = self._client.get_or_create_collection(
                name=settings.JOB_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        return self._job_col

    # ── indexing ──────────────────────────────────────────────────────────────

    def index_resume(
        self,
        resume_id: str,
        embedding_text: str,
        metadata: dict,
    ) -> None:
        """
        Embed and store a resume.

        embedding_text: PII-redacted text (skills + experience + education)
        metadata:       includes original PII fields (access-controlled separately)
        """
        vector = self.embedder.embed_single(embedding_text).tolist()

        # Chroma metadata values must be str/int/float/bool — serialise lists
        safe_metadata = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, list):
                safe_metadata[k] = json.dumps(v)
            else:
                safe_metadata[k] = v

        # Store a short snippet of the embedding text for display
        safe_metadata["snippet"] = embedding_text[:300]

        self.resume_collection.upsert(
            ids=[resume_id],
            embeddings=[vector],
            documents=[embedding_text],
            metadatas=[safe_metadata],
        )
        logger.info(f"Indexed resume: {resume_id}")

    def index_job(
        self,
        job_id: str,
        embedding_text: str,
        metadata: dict,
    ) -> None:
        """Embed and store a job posting."""
        vector = self.embedder.embed_single(embedding_text).tolist()
        safe_metadata = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, list):
                safe_metadata[k] = json.dumps(v)
            elif isinstance(v, bool):
                safe_metadata[k] = v
            else:
                safe_metadata[k] = v
        safe_metadata["snippet"] = embedding_text[:300]

        self.job_collection.upsert(
            ids=[job_id],
            embeddings=[vector],
            documents=[embedding_text],
            metadatas=[safe_metadata],
        )
        logger.info(f"Indexed job: {job_id}")

    # ── search ────────────────────────────────────────────────────────────────

    def search_resumes(
        self,
        query_text: str,
        top_k: int = 5,
        level_filter: Optional[str] = None,
        min_years: Optional[float] = None,
        max_years: Optional[float] = None,
        include_pii: bool = False,
    ) -> list[ResumeMatch]:
        """
        Find resumes semantically similar to a job description.

        Args:
            query_text:   job description or skills list to match against
            top_k:        number of results
            level_filter: restrict to a specific experience level
            min/max_years: filter by years of experience
            include_pii:  whether to include name/email in results
        """
        query_vector = self.embedder.embed_single(query_text).tolist()

        # Build metadata filter
        where_clause = self._build_where(level_filter, min_years, max_years)

        query_kwargs = dict(
            query_embeddings=[query_vector],
            n_results=min(top_k, max(1, self.resume_collection.count())),
            include=["documents", "distances", "metadatas"],
        )
        if where_clause:
            query_kwargs["where"] = where_clause

        results = self.resume_collection.query(**query_kwargs)

        matches = []
        for i, rid in enumerate(results["ids"][0]):
            meta     = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # Chroma cosine distance = 1 - cosine_similarity
            score    = round(1.0 - distance, 4)

            if score < settings.SIMILARITY_THRESHOLD:
                continue

            # Deserialise skills list
            skills_raw = meta.get("skills", "[]")
            try:
                skills = json.loads(skills_raw) if isinstance(skills_raw, str) else skills_raw
            except Exception:
                skills = []

            # Compute skill overlap with the query (simple keyword match)
            query_lower = query_text.lower()
            matched = [s for s in skills if s.lower() in query_lower]

            match = ResumeMatch(
                resume_id        = rid,
                similarity_score = score,
                level            = meta.get("level", "unknown"),
                years_experience = float(meta.get("years_experience", 0)),
                matched_skills   = matched,
                snippet          = meta.get("snippet", "")[:200],
                candidate_name   = meta.get("candidate_name") if include_pii else None,
                location         = meta.get("location") if include_pii else None,
            )
            matches.append(match)

        return sorted(matches, key=lambda m: m.similarity_score, reverse=True)

    def search_jobs(
        self,
        query_text: str,
        top_k: int = 5,
        level_filter: Optional[str] = None,
        remote_only: bool = False,
    ) -> list[JobMatch]:
        """Find jobs semantically similar to a resume."""
        query_vector = self.embedder.embed_single(query_text).tolist()

        where_clause = {}
        if level_filter:
            where_clause["level"] = level_filter
        if remote_only:
            where_clause["remote"] = True

        query_kwargs = dict(
            query_embeddings=[query_vector],
            n_results=min(top_k, max(1, self.job_collection.count())),
            include=["documents", "distances", "metadatas"],
        )
        if where_clause:
            query_kwargs["where"] = where_clause

        results = self.job_collection.query(**query_kwargs)

        matches = []
        for i, jid in enumerate(results["ids"][0]):
            meta     = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score    = round(1.0 - distance, 4)

            if score < settings.SIMILARITY_THRESHOLD:
                continue

            req_skills_raw = meta.get("required_skills", "[]")
            try:
                req_skills = json.loads(req_skills_raw) if isinstance(req_skills_raw, str) else req_skills_raw
            except Exception:
                req_skills = []

            query_lower = query_text.lower()
            matched = [s for s in req_skills if s.lower() in query_lower]

            match = JobMatch(
                job_id           = jid,
                title            = meta.get("title", "Unknown"),
                company          = meta.get("company", "Unknown"),
                similarity_score = score,
                level            = meta.get("level", "unknown"),
                location         = meta.get("location"),
                remote           = bool(meta.get("remote", False)),
                salary_range     = meta.get("salary_range"),
                matched_skills   = matched,
                snippet          = meta.get("snippet", "")[:200],
            )
            matches.append(match)

        return sorted(matches, key=lambda m: m.similarity_score, reverse=True)

    # ── utilities ─────────────────────────────────────────────────────────────

    def delete_resume(self, resume_id: str) -> bool:
        try:
            self.resume_collection.delete(ids=[resume_id])
            logger.info(f"Deleted resume: {resume_id}")
            return True
        except Exception as e:
            logger.error(f"Delete failed for {resume_id}: {e}")
            return False

    def delete_job(self, job_id: str) -> bool:
        try:
            self.job_collection.delete(ids=[job_id])
            return True
        except Exception as e:
            logger.error(f"Delete failed for {job_id}: {e}")
            return False

    def stats(self) -> dict:
        return {
            "total_resumes": self.resume_collection.count(),
            "total_jobs":    self.job_collection.count(),
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "collection_names": [settings.RESUME_COLLECTION, settings.JOB_COLLECTION],
        }

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_where(
        level_filter: Optional[str],
        min_years: Optional[float],
        max_years: Optional[float],
    ) -> dict:
        """Build Chroma where clause for metadata filtering."""
        conditions = []
        if level_filter:
            conditions.append({"level": {"$eq": level_filter}})
        if min_years is not None:
            conditions.append({"years_experience": {"$gte": min_years}})
        if max_years is not None:
            conditions.append({"years_experience": {"$lte": max_years}})

        if len(conditions) == 0:
            return {}
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


# ── query logger (MLOps) ──────────────────────────────────────────────────────

def log_query(query: str, results_count: int, search_type: str) -> None:
    """
    Log queries for monitoring and analytics.

    In production this would feed into:
      - Query distribution dashboards (what are users searching for?)
      - Retrieval quality monitoring (how often are results clicked?)
      - Drift detection (are query distributions changing over time?)
    """
    if not settings.ENABLE_QUERY_LOGGING:
        return

    import time
    Path(settings.QUERY_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "timestamp":     time.time(),
        "search_type":   search_type,
        "query_length":  len(query),
        "results_count": results_count,
        # Note: we log query LENGTH, not query TEXT, to preserve privacy
        # In production, log the query text only after explicit user consent
    }

    with open(settings.QUERY_LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
