from __future__ import annotations

import hashlib
import io
import sys
from types import SimpleNamespace

import pytest

from byeori import metadata_pdf as pdf


RAW = b"%PDF-1.7\nTest PDF bytes"
KEY = "papers/example/original.pdf"
TITLE = "Nrxn1a and chromatin organization"
TEXT = "Nrxn1\u03b1 and chromatin organi-\nzation\nPublished 2026\nhttps://doi.org/10.1234/example\n"


class ReadOnlyS3:
    def __init__(self, raw=RAW, content_length=None):
        self.body = io.BytesIO(raw)
        self.content_length = len(raw) if content_length is None else content_length
        self.calls = []

    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        return {"Body": self.body, "ContentLength": self.content_length,
                "VersionId": "observed-version", "ETag": '"observed-etag"'}


def inspect(s3, **kwargs):
    params = {"s3": s3, "bucket": "bucket", "pdf_key": KEY,
              "pdf_sha256": hashlib.sha256(RAW).hexdigest(), "doi": "10.1234/example", "title": TITLE}
    return pdf.inspect_pdf_identity(**(params | kwargs))


def test_exact_title_and_explicit_doi_use_whitespace_hyphen_and_greek_normalization():
    result = pdf.classify_text(TEXT, doi="https://doi.org/10.1234/EXAMPLE", title=TITLE)
    assert result["doi_present"] and result["title_present"]
    assert result["doi_candidates"] == ["10.1234/example"]
    assert result["year_candidates"] == [2026] and not result["errors"]
    assert not pdf.classify_text(TEXT, doi="10.1234/different", title="Nrxn2a and chromatin organization")["title_present"]


@pytest.mark.parametrize("heading", ["References", "REFERENCES AND NOTES", "Bibliography", "参考文献", "참고 문헌"])
def test_reference_list_dois_and_titles_are_not_identity_evidence(heading):
    text = TEXT + f"\n{heading}\ndoi:10.5555/reference\nA different article\n1999"
    result = pdf.classify_text(text, doi="10.5555/reference", title="A different article")
    assert not result["doi_present"] and not result["title_present"]
    assert result["doi_candidates"] == ["10.1234/example"]
    assert result["year_candidates"] == [2026]
    assert "reference" not in result["firstpage_excerpt"].lower()


def test_missing_or_invalid_catalogue_doi_can_return_explicit_unapproved_candidates():
    for supplied in (None, "arXiv:2401.01234", "not a DOI"):
        result = pdf.classify_text(TEXT + "arXiv:2401.01234\n10.7777/unlabelled", doi=supplied, title=TITLE)
        assert not result["doi_present"]
        assert result["doi_candidates"] == ["10.1234/example"] and result["title_present"]
    assert pdf.classify_text("arXiv:2401.01234 https://arxiv.org/abs/2401.01234")["doi_candidates"] == []


def test_doi_punctuation_is_removed_without_damaging_balanced_parentheses():
    result = pdf.classify_text("DOI:10.1016/S0168-9525(01)02291-6.\n(https://doi.org/10.1234/other).")
    assert result["doi_candidates"] == ["10.1016/s0168-9525(01)02291-6", "10.1234/other"]


def test_doi_label_can_precede_a_line_break_and_expected_label_whitespace():
    result = pdf.classify_text("DOI:\n10.1234/example", doi="doi: 10.1234/example")
    assert result["doi_present"] and result["doi_candidates"] == ["10.1234/example"]


def test_optional_ncbi_title_is_reported_separately_and_excerpt_is_bounded():
    result = pdf.classify_text(TEXT + "x" * 3000, title="Wrong stored title", candidate_title=TITLE)
    assert not result["title_present"] and result["candidate_title_present"]
    assert len(result["firstpage_excerpt"]) == 1500


def test_doi_candidates_are_bounded_without_truncating_an_identifier_into_a_proposal():
    long_doi = "doi:10.1234/" + "x" * 1000
    result = pdf.classify_text(long_doi + "\n" + "\n".join(f"doi:10.1234/{i}" for i in range(25)))
    assert len(result["doi_candidates"]) == pdf.MAX_DOI_CANDIDATES
    assert "invalid_explicit_doi" in result["errors"]
    assert "doi_candidate_limit_reached" in result["errors"]


def test_verified_stored_pdf_is_read_once_and_has_source_receipt(monkeypatch):
    s3 = ReadOnlyS3()
    monkeypatch.setattr(pdf, "extract_first_page", lambda raw: TEXT if raw == RAW else pytest.fail("Unexpected PDF"))
    result = inspect(s3)
    assert result["sha256_matches"] and result["doi_present"] and result["title_present"]
    assert result["pdf_key"] == KEY and result["sha256"] == hashlib.sha256(RAW).hexdigest()
    assert result["version_id"] == "observed-version" and result["etag"] == '"observed-etag"'
    assert s3.calls == [{"Bucket": "bucket", "Key": KEY}] and s3.body.closed
    assert result["errors"] == []


@pytest.mark.parametrize("key", [None, "", "wiki/example.pdf", "/papers/example.pdf", "papers/../example.pdf",
                                 "papers/x/../../example.pdf", "papers/%2e%2e/example.pdf", "papers/x\\example.pdf",
                                 "papers//example.pdf", "papers/example/clean.md"])
def test_invalid_or_missing_stored_path_never_infers_a_pdf_key_or_reads_s3(key):
    s3 = ReadOnlyS3()
    assert inspect(s3, pdf_key=key)["errors"] == ["invalid_stored_pdf_key"]
    assert not s3.calls


@pytest.mark.parametrize("digest", [None, "", "abc", "g" * 64])
def test_missing_or_invalid_observed_hash_prevents_reading(digest):
    s3 = ReadOnlyS3()
    assert inspect(s3, pdf_sha256=digest)["errors"] == ["invalid_stored_pdf_sha256"]
    assert not s3.calls


def test_hash_mismatch_does_not_parse_pdf(monkeypatch):
    monkeypatch.setattr(pdf, "extract_first_page", lambda raw: pytest.fail("Unverified PDF must not be parsed"))
    result = inspect(ReadOnlyS3(), pdf_sha256="0" * 64)
    assert not result["sha256_matches"] and not result["doi_present"]
    assert result["errors"] == ["pdf_sha256_mismatch"]


def test_declared_size_limit_rejects_before_body_read(monkeypatch):
    monkeypatch.setattr(pdf, "MAX_PDF_BYTES", 10)
    s3 = ReadOnlyS3(content_length=11)
    monkeypatch.setattr(s3.body, "read", lambda *a: pytest.fail("Oversized body must not be read"))
    assert inspect(s3)["errors"] == ["pdf_size_limit_exceeded"]
    assert s3.body.closed


def test_actual_size_limit_rejects_even_if_content_length_is_wrong(monkeypatch):
    monkeypatch.setattr(pdf, "MAX_PDF_BYTES", len(RAW) - 1)
    monkeypatch.setattr(pdf, "extract_first_page", lambda raw: pytest.fail("Oversized PDF must not be parsed"))
    assert inspect(ReadOnlyS3(content_length=1))["errors"] == ["pdf_size_limit_exceeded"]


def test_parse_errors_and_image_only_first_page_are_explicit(monkeypatch):
    def fail(raw):
        raise ValueError("Corrupt document")
    monkeypatch.setattr(pdf, "extract_first_page", fail)
    assert inspect(ReadOnlyS3())["errors"] == ["pdf_parse_failed:ValueError"]
    monkeypatch.setattr(pdf, "extract_first_page", lambda raw: "")
    assert inspect(ReadOnlyS3())["errors"] == ["no_first_page_text"]


def test_extract_first_page_imports_fitz_lazily_and_never_selects_page_two(monkeypatch):
    loaded_pages = []
    class Document:
        page_count = 2
        needs_pass = False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def load_page(self, page):
            loaded_pages.append(page)
            assert page == 0
            return SimpleNamespace(get_text=lambda kind, sort: TEXT)
    def open_pdf(**kwargs):
        assert kwargs == {"stream": RAW, "filetype": "pdf"}
        return Document()
    monkeypatch.setitem(sys.modules, "fitz", SimpleNamespace(open=open_pdf))
    assert pdf.extract_first_page(RAW) == TEXT
    assert loaded_pages == [0]


def test_real_tiny_pdf_first_page_when_pymupdf_is_available():
    fitz = pytest.importorskip("fitz")
    with fitz.open() as document:
        document.new_page().insert_text((72, 72), "A tiny identity test\nDOI:10.1234/first\n2026")
        document.new_page().insert_text((72, 72), "DOI:10.1234/second")
        raw = document.tobytes()
    result = inspect(ReadOnlyS3(raw), pdf_sha256=hashlib.sha256(raw).hexdigest(),
                     title="A tiny identity test", doi="10.1234/first")
    assert result["doi_present"] and result["title_present"] and result["sha256_matches"]
    assert result["doi_candidates"] == ["10.1234/first"]
