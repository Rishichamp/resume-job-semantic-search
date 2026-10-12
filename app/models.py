"""
Pydantic models for the Resume-to-Job Semantic Search Engine.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class ExperienceLevel(str, Enum):
    INTERN     = "intern"
    JUNIOR     = "junior"
    MID        = "mid"
    SENIOR     = "senior"
    LEAD       = "lead"
    PRINCIPAL  = "principal"


# ── Resume models ─────────────────────────────────────────────────────────────

class ResumeInput(BaseModel):
    """Input schema for indexing a resume."""
    resume_id: str              = Field(..., description="Unique identifier for this resume")
    # Core text fields — will be embedded
    skills: list[str]           = Field(..., description="List of technical and soft skills")
    experience_summary: str     = Field(..., description="Work experience summary (roles, companies, achievements)")
    education: str              = Field(..., description="Education background")
    certifications: list[str]   = Field(default_factory=list)
    projects: str               = Field("", description="Notable projects or portfolio description")
    # Metadata — stored but not embedded (for filtering)
    years_experience: float     = Field(..., ge=0, le=50)
    level: ExperienceLevel      = Field(...)
    # PII — stored in metadata ONLY if REDACT_PII=false, never embedded
    candidate_name: Optional[str] = Field(None, description="Stored in metadata, never embedded")
    email: Optional[str]          = Field(None, description="Stored in metadata, never embedded")
    location: Optional[str]       = Field(None)

    @field_validator("skills")
    @classmethod
    def normalise_skills(cls, v):
        return [s.strip().lower() for s in v if s.strip()]


class ResumeMatch(BaseModel):
    """A single resume match result."""
    resume_id: str
    similarity_score: float     = Field(..., ge=0.0, le=1.0)
    level: str
    years_experience: float
    matched_skills: list[str]   = Field(default_factory=list)
    # PII only returned if REDACT_PII=false
    candidate_name: Optional[str] = None
    location: Optional[str]       = None
    snippet: str                  = Field("", description="Relevant excerpt from the resume text")


# ── Job posting models ────────────────────────────────────────────────────────

class JobPostingInput(BaseModel):
    """Input schema for indexing a job posting."""
    job_id: str
    title: str
    company: str
    description: str            = Field(..., description="Full job description text")
    required_skills: list[str]
    nice_to_have_skills: list[str] = Field(default_factory=list)
    level: ExperienceLevel
    min_years_experience: float = Field(0, ge=0)
    location: Optional[str]     = None
    remote: bool                = False
    salary_range: Optional[str] = None

    @field_validator("required_skills", "nice_to_have_skills")
    @classmethod
    def normalise_skills(cls, v):
        return [s.strip().lower() for s in v if s.strip()]


class JobMatch(BaseModel):
    """A single job match result."""
    job_id: str
    title: str
    company: str
    similarity_score: float
    level: str
    location: Optional[str]
    remote: bool
    salary_range: Optional[str]
    matched_skills: list[str]
    snippet: str


# ── Search request/response models ───────────────────────────────────────────

class ResumeSearchRequest(BaseModel):
    """Search for resumes matching a job posting."""
    job_id: Optional[str]       = Field(None, description="Search by existing indexed job ID")
    job_description: Optional[str] = Field(None, description="Or provide raw job description text")
    required_skills: list[str]  = Field(default_factory=list)
    level_filter: Optional[ExperienceLevel] = None
    min_years: Optional[float]  = None
    max_years: Optional[float]  = None
    top_k: int                  = Field(5, ge=1, le=20)

    @field_validator("job_description", "job_id")
    @classmethod
    def at_least_one(cls, v):
        return v


class JobSearchRequest(BaseModel):
    """Search for jobs matching a resume."""
    resume_id: Optional[str]    = Field(None, description="Search by existing indexed resume ID")
    skills: Optional[list[str]] = Field(None, description="Or provide skills list directly")
    experience_summary: Optional[str] = None
    level_filter: Optional[ExperienceLevel] = None
    remote_only: bool           = False
    top_k: int                  = Field(5, ge=1, le=20)


class SearchResponse(BaseModel):
    """Generic search response wrapper."""
    query_summary: str
    total_results: int
    results: list               # list[ResumeMatch] or list[JobMatch]
    search_time_ms: float


class IndexResponse(BaseModel):
    """Response after indexing a document."""
    id: str
    status: str
    message: str
    embedding_dim: int


class StatsResponse(BaseModel):
    """Index statistics."""
    total_resumes: int
    total_jobs: int
    embedding_model: str
    embedding_provider: str
    collection_names: list[str]
