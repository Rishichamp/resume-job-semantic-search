"""
Tests for the Resume-to-Job Semantic Search Engine.
Run: pytest tests/ -v
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.pii_redaction import redact_with_regex, redact_pii, build_resume_embedding_text
from app.embedder import Embedder

client = TestClient(app)

# ── PII redaction tests ───────────────────────────────────────────────────────

def test_redact_email():
    result = redact_with_regex("Contact me at john.doe@gmail.com for details.")
    assert "john.doe@gmail.com" not in result.redacted_text
    assert "[EMAIL_REDACTED]" in result.redacted_text
    assert result.redaction_count >= 1

def test_redact_phone_us():
    result = redact_with_regex("Call me at 555-867-5309.")
    assert "555-867-5309" not in result.redacted_text

def test_redact_indian_phone():
    result = redact_with_regex("My number is 9876543210.")
    assert "9876543210" not in result.redacted_text

def test_redact_linkedin():
    result = redact_with_regex("Profile: https://linkedin.com/in/johndoe")
    assert "johndoe" not in result.redacted_text
    assert "[LINKEDIN_REDACTED]" in result.redacted_text

def test_no_pii_unchanged():
    text = "Experienced Python developer with 5 years in ML engineering."
    result = redact_with_regex(text)
    assert result.redacted_text == text
    assert result.redaction_count == 0

def test_multiple_pii_types():
    text = "John can be reached at john@test.com or 555-123-4567."
    result = redact_with_regex(text)
    assert "john@test.com" not in result.redacted_text
    assert "555-123-4567" not in result.redacted_text
    assert result.redaction_count >= 2

def test_build_resume_embedding_text_excludes_name():
    resume = {
        "skills": ["python", "ml"],
        "experience_summary": "Senior engineer at Google, email: test@google.com",
        "education": "B.Tech CS",
        "certifications": [],
        "projects": "",
        "candidate_name": "John Smith",
        "email": "john@smith.com",
    }
    text = build_resume_embedding_text(resume, redact=True)
    # PII should not appear in embedding text
    assert "John Smith" not in text
    assert "john@smith.com" not in text
    # Skills and experience should be present
    assert "python" in text.lower()

# ── Embedder tests ────────────────────────────────────────────────────────────

def test_embedder_cosine_similarity_identical():
    embedder = Embedder.__new__(Embedder)
    embedder.provider = "local"
    embedder.model = "all-MiniLM-L6-v2"
    embedder._client = None
    embedder._st_model = None
    # Normalised identical vectors → similarity = 1.0
    v = np.array([0.6, 0.8], dtype=np.float32)
    assert abs(embedder.cosine_similarity(v, v) - 1.0) < 1e-5

def test_embedder_batch_similarity_shape():
    embedder = Embedder.__new__(Embedder)
    query = np.array([0.6, 0.8], dtype=np.float32)
    corpus = np.array([[0.6, 0.8], [0.8, 0.6], [1.0, 0.0]], dtype=np.float32)
    sims = embedder.batch_cosine_similarity(query, corpus)
    assert sims.shape == (3,)
    # First element should be ~1.0 (same vector)
    assert sims[0] > 0.99

# ── API endpoint tests ────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_stats():
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_resumes" in data
    assert "embedding_model" in data

MOCK_RESUME_MATCH = {
    "resume_id": "r001",
    "similarity_score": 0.87,
    "level": "senior",
    "years_experience": 5.0,
    "matched_skills": ["python", "pytorch"],
    "candidate_name": None,
    "location": None,
    "snippet": "Python, PyTorch, transformers...",
}

@patch("app.main.store.search_resumes", return_value=[])
def test_search_resumes_no_query(mock_search):
    r = client.post("/search/resumes", json={"top_k": 3})
    assert r.status_code == 400

@patch("app.main.store.search_resumes", return_value=[])
def test_search_resumes_empty_result(mock_search):
    r = client.post("/search/resumes", json={
        "job_description": "Senior Python ML engineer with PyTorch experience",
        "top_k": 5
    })
    assert r.status_code == 200
    data = r.json()
    assert data["total_results"] == 0
    assert isinstance(data["results"], list)

@patch("app.main.store.search_jobs", return_value=[])
def test_search_jobs_no_query(mock_search):
    r = client.post("/search/jobs", json={"top_k": 3})
    assert r.status_code == 400

def test_index_resume_valid():
    payload = {
        "resume_id": "test_r999",
        "skills": ["python", "fastapi"],
        "experience_summary": "Software engineer with 3 years experience.",
        "education": "B.Tech CS",
        "years_experience": 3.0,
        "level": "mid",
    }
    with patch("app.main.store.index_resume") as mock_idx:
        r = client.post("/resumes/index", json=payload)
        assert r.status_code == 200
        assert r.json()["status"] == "indexed"

def test_index_job_valid():
    payload = {
        "job_id": "test_j999",
        "title": "ML Engineer",
        "company": "TestCo",
        "description": "Build ML systems.",
        "required_skills": ["python", "sklearn"],
        "level": "mid",
        "min_years_experience": 2.0,
    }
    with patch("app.main.store.index_job") as mock_idx:
        r = client.post("/jobs/index", json=payload)
        assert r.status_code == 200

def test_delete_resume_not_found():
    with patch("app.main.store.delete_resume", return_value=False):
        r = client.delete("/resumes/nonexistent_id")
        assert r.status_code == 404
