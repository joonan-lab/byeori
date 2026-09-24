"""Read-only original-PDF identity evidence for an AWS metadata review worker.

Callers supply the catalogue's observed PDF key and SHA-256. This module never infers
paths, changes an object or catalogue row, or verifies scientific claims. PyMuPDF is
imported only when parsing a verified PDF in the ECS execution environment.
"""

from __future__ import annotations

import hashlib
import re
import threading
from typing import Any
from urllib.parse import unquote

from .openalex_match import _squash, normalize_doi


MAX_PDF_BYTES = 200 * 1024 * 1024
MAX_EXCERPT_CHARS = 1500
MAX_DOI_CANDIDATES = 20
MAX_DOI_LENGTH = 512
_PARSE_LOCK = threading.Lock()
REFERENCE_HEADING = re.compile(
    r"(?im)^[ \t]*(?:references(?:[ \t]+and[ \t]+notes)?|bibliography|"
    r"literature[ \t]+cited|reference[ \t]+list|参考(?:文献|文獻)?|참고[ \t]*문헌)"
    r"[ \t]*[:：]?[ \t]*$"
)
EXPLICIT_DOI = re.compile(
    r"(?:\bdoi[ \t]*[:：]\s*|https?://(?:dx\.)?doi\.org/)"
    r"(10\.[0-9]{4,9}/[^\s<>\"']+)", re.I
)


def _doi_candidate(value: str) -> str | None:
    value = value.rstrip(".,;:")
    for left, right in (("(", ")"), ("[", "]"), ("{", "}")):
        while value.endswith(right) and value.count(right) > value.count(left):
            value = value[:-1]
    value = normalize_doi(value)
    if value and len(value) <= MAX_DOI_LENGTH and re.fullmatch(r"10\.[0-9]{4,9}/\S+", value):
        return value
    return None


def classify_text(text: str, *, doi: str | None = None, title: str | None = None,
                  candidate_title: str | None = None) -> dict[str, Any]:
    """Classify first-page text; explicit DOI candidates remain unapproved proposals.

    Title presence means the supplied normalized title occurs before a reference-list
    heading. It is not a fuzzy match or a claim that the complete article was checked.
    ``candidate_title`` optionally cross-checks an independently fetched NCBI title.
    """
    content = REFERENCE_HEADING.split(str(text), maxsplit=1)[0]
    candidates: list[str] = []
    errors: list[str] = []
    for match in EXPLICIT_DOI.finditer(content):
        candidate = _doi_candidate(match.group(1))
        if candidate is None:
            if "invalid_explicit_doi" not in errors:
                errors.append("invalid_explicit_doi")
        elif candidate not in candidates:
            if len(candidates) == MAX_DOI_CANDIDATES:
                errors.append("doi_candidate_limit_reached")
                break
            candidates.append(candidate)
    normalized = _squash(content)
    expected = (normalize_doi(doi) or "").strip()
    normalized_title = _squash(title)
    normalized_candidate_title = _squash(candidate_title)
    return {
        "doi_present": bool(expected and expected in candidates),
        "title_present": bool(normalized_title and normalized_title in normalized),
        "candidate_title_present": bool(normalized_candidate_title and normalized_candidate_title in normalized),
        "doi_candidates": candidates,
        "year_candidates": sorted({int(year) for year in re.findall(r"\b(?:18|19|20)[0-9]{2}\b", content)})[:20],
        "firstpage_excerpt": content[:MAX_EXCERPT_CHARS],
        "errors": errors,
    }


def extract_first_page(raw: bytes) -> str:
    """Parse only page one from memory. No local PDF or rendered preview is created."""
    import fitz

    # S3 reads may run concurrently, but PyMuPDF parsing is serialized per process.
    with _PARSE_LOCK, fitz.open(stream=raw, filetype="pdf") as document:
        if document.page_count < 1 or document.needs_pass:
            raise ValueError("PDF has no accessible first page")
        return document.load_page(0).get_text("text", sort=True)


def _valid_pdf_key(value: Any) -> bool:
    if not isinstance(value, str) or len(value) > 1024 or value != value.strip():
        return False
    decoded = unquote(value)
    return (decoded.startswith("papers/") and decoded.lower().endswith(".pdf")
            and "\\" not in decoded and not any(ord(char) < 32 for char in decoded)
            and all(part not in ("", ".", "..") for part in decoded.split("/")))


def inspect_pdf_identity(*, s3: Any, bucket: str, pdf_key: str | None,
                         pdf_sha256: str | None, doi: str | None = None,
                         title: str | None = None, candidate_title: str | None = None) -> dict[str, Any]:
    """Read one stored original with a 200 MiB cap, verify its hash, then inspect page one.

    The caller must obtain ``pdf_key`` and ``pdf_sha256`` from the current stored row.
    A missing key/hash, an oversized PDF or a hash mismatch prevents PDF parsing.
    Only S3 GetObject is used; returned evidence can be saved separately by the caller.
    """
    result = {
        "sha256_matches": False, "pdf_key": str(pdf_key or "")[:1024],
        "sha256": None, "version_id": None, "etag": None,
        **classify_text("", doi=doi, title=title, candidate_title=candidate_title),
    }
    if not _valid_pdf_key(pdf_key):
        result["errors"] = ["invalid_stored_pdf_key"]
        return result
    if not isinstance(pdf_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", pdf_sha256):
        result["errors"] = ["invalid_stored_pdf_sha256"]
        return result
    body = None
    try:
        response = s3.get_object(Bucket=bucket, Key=pdf_key)
        body = response["Body"]
        result.update(version_id=response.get("VersionId"), etag=response.get("ETag"))
        if int(response.get("ContentLength", 0)) > MAX_PDF_BYTES:
            result["errors"] = ["pdf_size_limit_exceeded"]
            return result
        raw = body.read(MAX_PDF_BYTES + 1)
        if len(raw) > MAX_PDF_BYTES:
            result["errors"] = ["pdf_size_limit_exceeded"]
            return result
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)
        result["errors"] = [f"pdf_read_failed:{str(code)[:100]}"]
        return result
    finally:
        if body is not None:
            body.close()
    result["sha256"] = hashlib.sha256(raw).hexdigest()
    result["sha256_matches"] = result["sha256"] == pdf_sha256.lower()
    if not result["sha256_matches"]:
        result["errors"] = ["pdf_sha256_mismatch"]
        return result
    try:
        text = extract_first_page(raw)
    except Exception as exc:
        result["errors"] = [f"pdf_parse_failed:{type(exc).__name__}"]
        return result
    result.update(classify_text(text, doi=doi, title=title, candidate_title=candidate_title))
    if not text.strip():
        result["errors"].append("no_first_page_text")
    return result
