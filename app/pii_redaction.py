"""
PII Redaction — privacy layer for the resume search engine.

WHAT IS PII (Personally Identifiable Information)?
  - Names, email addresses, phone numbers
  - Physical addresses, postal codes
  - National ID numbers, passport numbers
  - Dates of birth

WHY REDACT BEFORE EMBEDDING?
  Embeddings encode semantic meaning — including names and contact details.
  If candidate names are embedded, similarity scores could be influenced by
  name-based patterns (gender, ethnicity signals in names). More critically,
  embeddings can potentially be inverted to recover approximate source text.
  Redacting PII before embedding ensures:
    1. Privacy-preserving semantic search (match on skills, not identity)
    2. GDPR compliance — no personal data in the ML model pipeline
    3. Fairer matching — scores reflect skills/experience, not demographics

HOW IT WORKS:
  1. Regex patterns catch common PII formats (emails, phones, IDs)
  2. spaCy NER catches names and locations (PERSON, GPE, LOC entities)
  3. The redacted text is what gets embedded and stored in the vector DB
  4. The original PII is stored SEPARATELY in metadata (access-controlled)
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RedactionResult:
    """Output of the PII redaction process."""
    original_text: str
    redacted_text: str
    pii_found: list[dict] = field(default_factory=list)  # [{type, original, replacement}]
    redaction_count: int = 0

    @property
    def has_pii(self) -> bool:
        return self.redaction_count > 0


# ── Regex-based redaction (fast, no ML needed) ───────────────────────────────

# Patterns: (compiled_regex, replacement_token, pii_type)
PII_PATTERNS = [
    # Email addresses
    (re.compile(
        r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b'
    ), "[EMAIL_REDACTED]", "email"),

    # Phone numbers (various formats)
    (re.compile(
        r'(\+?\d{1,3}[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}'
    ), "[PHONE_REDACTED]", "phone"),

    # Indian mobile numbers
    (re.compile(
        r'\b[6-9]\d{9}\b'
    ), "[PHONE_REDACTED]", "phone_in"),

    # Aadhaar numbers (12 digits, possibly spaced as 4-4-4)
    (re.compile(
        r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'
    ), "[ID_REDACTED]", "aadhaar"),

    # PAN card (India)
    (re.compile(
        r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b'
    ), "[ID_REDACTED]", "pan"),

    # Social Security Number (US)
    (re.compile(
        r'\b\d{3}[\s\-]\d{2}[\s\-]\d{4}\b'
    ), "[SSN_REDACTED]", "ssn"),

    # URLs / LinkedIn profiles / GitHub links
    (re.compile(
        r'https?://(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_/]+'
    ), "[LINKEDIN_REDACTED]", "linkedin"),
    (re.compile(
        r'https?://github\.com/[A-Za-z0-9\-_]+'
    ), "[GITHUB_REDACTED]", "github"),

    # Physical addresses (simplified: number + street pattern)
    (re.compile(
        r'\b\d{1,5}\s+(?:[A-Z][a-z]+\s){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Drive|Dr|Court|Ct|Boulevard|Blvd)\b'
    ), "[ADDRESS_REDACTED]", "address"),

    # Postal / ZIP codes
    (re.compile(
        r'\b\d{5}(?:[-\s]\d{4})?\b'          # US ZIP
    ), "[POSTAL_REDACTED]", "zip_us"),
    (re.compile(
        r'\b[A-Z]\d[A-Z]\s?\d[A-Z]\d\b'       # Canadian postal
    ), "[POSTAL_REDACTED]", "postal_ca"),
    (re.compile(
        r'\b\d{6}\b'                            # Indian PIN code (6 digits)
    ), "[POSTAL_REDACTED]", "pin_in"),
]


def redact_with_regex(text: str) -> RedactionResult:
    """Apply regex-based PII redaction patterns."""
    redacted = text
    pii_found = []

    for pattern, replacement, pii_type in PII_PATTERNS:
        matches = pattern.findall(redacted)
        if matches:
            for match in (matches if isinstance(matches[0], str) else [m[0] for m in matches]):
                pii_found.append({
                    "type": pii_type,
                    "original": match,
                    "replacement": replacement,
                })
            redacted = pattern.sub(replacement, redacted)

    return RedactionResult(
        original_text=text,
        redacted_text=redacted,
        pii_found=pii_found,
        redaction_count=len(pii_found),
    )


def redact_with_ner(text: str) -> RedactionResult:
    """
    Use spaCy NER to redact names, locations, and other entities.
    Falls back gracefully if spaCy is not installed.
    """
    try:
        import spacy
        nlp = _get_spacy_model()
        doc = nlp(text)
        redacted = text
        pii_found = []

        # Process entities in reverse order (to preserve character offsets)
        for ent in reversed(doc.ents):
            if ent.label_ in ("PERSON", "GPE", "LOC", "FAC", "ORG"):
                replacement = f"[{ent.label_}_REDACTED]"
                pii_found.append({
                    "type": ent.label_.lower(),
                    "original": ent.text,
                    "replacement": replacement,
                })
                redacted = redacted[:ent.start_char] + replacement + redacted[ent.end_char:]

        return RedactionResult(
            original_text=text,
            redacted_text=redacted,
            pii_found=pii_found,
            redaction_count=len(pii_found),
        )
    except Exception as e:
        logger.warning(f"NER redaction unavailable: {e}. Using regex only.")
        return RedactionResult(
            original_text=text,
            redacted_text=text,
            pii_found=[],
            redaction_count=0,
        )


_spacy_nlp = None
def _get_spacy_model():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            raise
    return _spacy_nlp


def redact_pii(text: str, use_ner: bool = True) -> RedactionResult:
    """
    Full PII redaction pipeline:
      1. Regex patterns (emails, phones, IDs, addresses)
      2. NER-based redaction (names, locations) — optional

    Args:
        text: input text to redact
        use_ner: whether to use spaCy NER (requires model download)
    """
    # Stage 1: regex
    regex_result = redact_with_regex(text)

    if not use_ner:
        return regex_result

    # Stage 2: NER on the regex-redacted text
    ner_result = redact_with_ner(regex_result.redacted_text)

    # Merge results
    return RedactionResult(
        original_text=text,
        redacted_text=ner_result.redacted_text,
        pii_found=regex_result.pii_found + ner_result.pii_found,
        redaction_count=regex_result.redaction_count + ner_result.redaction_count,
    )


def build_resume_embedding_text(resume_data: dict, redact: bool = True) -> str:
    """
    Build the text string that will be embedded for a resume.

    Key principle: embed only the SEMANTIC content (skills, experience, education).
    Never embed PII — it adds noise and violates privacy.
    """
    parts = []

    # Skills — most important for matching
    if resume_data.get("skills"):
        skills_text = ", ".join(resume_data["skills"])
        parts.append(f"Technical skills: {skills_text}")

    # Experience — second most important
    if resume_data.get("experience_summary"):
        exp = resume_data["experience_summary"]
        if redact:
            exp = redact_pii(exp, use_ner=False).redacted_text  # regex only for speed
        parts.append(f"Experience: {exp}")

    # Education
    if resume_data.get("education"):
        parts.append(f"Education: {resume_data['education']}")

    # Certifications
    if resume_data.get("certifications"):
        certs = ", ".join(resume_data["certifications"])
        parts.append(f"Certifications: {certs}")

    # Projects
    if resume_data.get("projects"):
        parts.append(f"Projects: {resume_data['projects']}")

    text = "\n\n".join(parts)
    return text


def build_job_embedding_text(job_data: dict) -> str:
    """Build the text string that will be embedded for a job posting."""
    parts = []
    parts.append(f"Job title: {job_data.get('title', '')}")
    parts.append(f"Company: {job_data.get('company', '')}")

    if job_data.get("description"):
        parts.append(f"Description: {job_data['description']}")

    if job_data.get("required_skills"):
        parts.append(f"Required skills: {', '.join(job_data['required_skills'])}")

    if job_data.get("nice_to_have_skills"):
        parts.append(f"Nice to have: {', '.join(job_data['nice_to_have_skills'])}")

    return "\n\n".join(parts)
