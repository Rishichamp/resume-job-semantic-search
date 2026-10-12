"""
Resume-to-Job Semantic Search Engine — FastAPI Application
===========================================================
Endpoints:
  POST /resumes/index          — index a resume
  POST /jobs/index             — index a job posting
  POST /search/resumes         — find matching resumes for a job
  POST /search/jobs            — find matching jobs for a resume
  DELETE /resumes/{id}         — remove a resume from the index
  DELETE /jobs/{id}            — remove a job from the index
  GET  /stats                  — index statistics
  GET  /health                 — health check
"""

import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.config import settings
from app.models import (
    ResumeInput, JobPostingInput,
    ResumeSearchRequest, JobSearchRequest,
    SearchResponse, IndexResponse, StatsResponse,
)
from app.pii_redaction import (
    redact_pii,
    build_resume_embedding_text,
    build_job_embedding_text,
)
from app.vector_store import VectorStoreManager, log_query
from app.embedder import get_embedder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Resume-to-Job Semantic Search Engine",
    description=(
        "Semantic matching between resumes and job postings using dense embeddings. "
        "Supports PII redaction, metadata filtering, and MLOps query logging."
    ),
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

store = VectorStoreManager()


@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.time()
    resp = await call_next(request)
    resp.headers["X-Process-Time"] = f"{(time.time()-t)*1000:.1f}ms"
    return resp


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return """<html><body style="font-family:sans-serif;max-width:640px;margin:40px auto">
    <h1>🔍 Resume-to-Job Semantic Search</h1>
    <p><a href="/docs">📖 Swagger UI</a> &nbsp;|&nbsp; <a href="/redoc">📄 ReDoc</a></p>
    </body></html>"""


@app.get("/health")
async def health():
    s = store.stats()
    return {"status": "ok", **s}


# ── Index endpoints ───────────────────────────────────────────────────────────
@app.post("/resumes/index", response_model=IndexResponse)
async def index_resume(resume: ResumeInput):
    """
    Index a resume for semantic search.

    PII handling:
    - Skills, experience, education, certifications, projects → embedded (after PII redaction)
    - Name, email, location → stored in metadata ONLY (never embedded)
    """
    try:
        # Build text to embed (with optional PII redaction)
        embedding_text = build_resume_embedding_text(
            resume.model_dump(),
            redact=settings.REDACT_PII,
        )

        # Metadata — includes PII fields for retrieval/display
        metadata = {
            "level":            resume.level.value,
            "years_experience": resume.years_experience,
            "skills":           resume.skills,  # stored as JSON list
            "location":         resume.location,
        }
        if not settings.REDACT_PII:
            metadata["candidate_name"] = resume.candidate_name
            metadata["email"]          = resume.email
        else:
            # Store a placeholder so we know PII was redacted
            metadata["candidate_name"] = "[REDACTED]"

        store.index_resume(
            resume_id=resume.resume_id,
            embedding_text=embedding_text,
            metadata=metadata,
        )

        embedder = get_embedder()
        return IndexResponse(
            id=resume.resume_id,
            status="indexed",
            message=f"Resume indexed successfully. PII redaction: {settings.REDACT_PII}",
            embedding_dim=embedder.dimension,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jobs/index", response_model=IndexResponse)
async def index_job(job: JobPostingInput):
    """Index a job posting for semantic search."""
    try:
        embedding_text = build_job_embedding_text(job.model_dump())
        metadata = {
            "title":            job.title,
            "company":          job.company,
            "level":            job.level.value,
            "min_years":        job.min_years_experience,
            "location":         job.location,
            "remote":           job.remote,
            "salary_range":     job.salary_range,
            "required_skills":  job.required_skills,
        }
        store.index_job(job_id=job.job_id, embedding_text=embedding_text, metadata=metadata)
        embedder = get_embedder()
        return IndexResponse(
            id=job.job_id,
            status="indexed",
            message="Job posting indexed successfully.",
            embedding_dim=embedder.dimension,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Search endpoints ──────────────────────────────────────────────────────────
@app.post("/search/resumes", response_model=SearchResponse)
async def search_resumes(req: ResumeSearchRequest):
    """
    Find resumes matching a job posting.

    Provide either:
    - A job description text (job_description)
    - Required skills list (required_skills)
    - Or both (they will be concatenated)
    """
    if not req.job_description and not req.required_skills:
        raise HTTPException(400, "Provide job_description or required_skills.")

    start = time.time()
    query_parts = []
    if req.required_skills:
        query_parts.append("Required skills: " + ", ".join(req.required_skills))
    if req.job_description:
        query_parts.append(req.job_description)
    query_text = "\n".join(query_parts)

    try:
        matches = store.search_resumes(
            query_text=query_text,
            top_k=req.top_k,
            level_filter=req.level_filter.value if req.level_filter else None,
            min_years=req.min_years,
            max_years=req.max_years,
            include_pii=not settings.REDACT_PII,
        )
        elapsed = (time.time() - start) * 1000
        log_query(query_text, len(matches), "resume_search")

        return SearchResponse(
            query_summary=f"Resume search: {len(query_text)} chars, top_k={req.top_k}",
            total_results=len(matches),
            results=[m.model_dump() for m in matches],
            search_time_ms=round(elapsed, 2),
        )
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/search/jobs", response_model=SearchResponse)
async def search_jobs(req: JobSearchRequest):
    """Find jobs matching a resume profile."""
    if not req.skills and not req.experience_summary:
        raise HTTPException(400, "Provide skills or experience_summary.")

    start = time.time()
    query_parts = []
    if req.skills:
        query_parts.append("Skills: " + ", ".join(req.skills))
    if req.experience_summary:
        query_parts.append(req.experience_summary)
    query_text = "\n".join(query_parts)

    try:
        matches = store.search_jobs(
            query_text=query_text,
            top_k=req.top_k,
            level_filter=req.level_filter.value if req.level_filter else None,
            remote_only=req.remote_only,
        )
        elapsed = (time.time() - start) * 1000
        log_query(query_text, len(matches), "job_search")

        return SearchResponse(
            query_summary=f"Job search: {len(query_text)} chars, top_k={req.top_k}",
            total_results=len(matches),
            results=[m.model_dump() for m in matches],
            search_time_ms=round(elapsed, 2),
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Delete endpoints ──────────────────────────────────────────────────────────
@app.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: str):
    """Remove a resume from the index (GDPR right to erasure)."""
    success = store.delete_resume(resume_id)
    if not success:
        raise HTTPException(404, f"Resume '{resume_id}' not found.")
    return {"status": "deleted", "id": resume_id}


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Remove a job posting from the index."""
    success = store.delete_job(job_id)
    if not success:
        raise HTTPException(404, f"Job '{job_id}' not found.")
    return {"status": "deleted", "id": job_id}


# ── Stats ─────────────────────────────────────────────────────────────────────
@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Return index statistics."""
    return StatsResponse(**store.stats())
