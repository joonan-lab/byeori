"""A PDF GROBID cannot read is read through a derived copy, and the original is never the one changed."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

WORKER = Path(__file__).parents[1] / "infra/extract_worker.py"


def load_worker(monkeypatch, preprocess):
    monkeypatch.setenv("BUCKET_NAME", "bucket")
    monkeypatch.setenv("TABLE_NAME", "table")
    monkeypatch.setenv("JOB_KEY", "jobs/x.json")
    monkeypatch.setenv("PREPROCESS", preprocess)
    fake_boto3 = types.SimpleNamespace(client=lambda *a, **k: None,
                                       resource=lambda *a, **k: types.SimpleNamespace(Table=lambda name: None))
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    # The container installs requests; the project environment does not need it.
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(RequestException=Exception))
    spec = importlib.util.spec_from_file_location("extract_worker", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ocr_rasterises_every_page_because_a_scan_can_carry_a_stamp_as_its_only_text(monkeypatch):
    worker = load_worker(monkeypatch, "ocr")
    command = worker.preprocess_command("/tmp/in.pdf", "/tmp/out.pdf")
    assert command[0] == "ocrmypdf" and "--force-ocr" in command and command[-2:] == ["/tmp/in.pdf", "/tmp/out.pdf"]
    assert worker.PREPROCESS_PACKAGES["ocr"] == ["ghostscript", "ocrmypdf", "tesseract-ocr-eng"]


def test_shrink_rewrites_with_ghostscript_and_keeps_the_text_layer(monkeypatch):
    worker = load_worker(monkeypatch, "shrink")
    command = worker.preprocess_command("/tmp/in.pdf", "/tmp/out.pdf")
    assert command[0] == "gs" and "-sDEVICE=pdfwrite" in command and "-dPDFSETTINGS=/ebook" in command
    assert "-sOutputFile=/tmp/out.pdf" in command and command[-1] == "/tmp/in.pdf"


def test_an_unknown_preprocess_stops_the_worker_before_any_paper(monkeypatch):
    with pytest.raises(SystemExit):
        load_worker(monkeypatch, "sharpen")


def test_no_preprocess_is_the_default_and_names_no_derived_copy(monkeypatch):
    worker = load_worker(monkeypatch, "")
    assert worker.PREPROCESS == "" and worker.derived_key("paper-one") == "papers/paper-one/derived.pdf"


def test_the_launcher_passes_the_preprocess_to_the_worker_and_refuses_others():
    from byeori import extract
    assert extract.worker_environment("jobs/r/0.json", force=True, preprocess="ocr") == [
        {"name": "JOB_KEY", "value": "jobs/r/0.json"}, {"name": "FORCE", "value": "1"}, {"name": "PREPROCESS", "value": "ocr"}]
    assert extract.worker_environment("jobs/r/0.json") == [{"name": "JOB_KEY", "value": "jobs/r/0.json"}]
    with pytest.raises(ValueError):
        extract.worker_environment("jobs/r/0.json", preprocess="sharpen")


def test_the_status_follows_the_verdict_the_row_carries_when_the_text_is_stored(monkeypatch):
    """Storing the text starts the identity resolve, so the verdict can land while GROBID runs.

    Reading it from the copy taken before extraction would park a paper that has just been cleared
    (2026-09-23, when the intake moved to AWS).
    """
    worker = load_worker(monkeypatch, "")
    source = WORKER.read_text()
    body = source[source.index("def extract("):]
    read_again = body.index('current = table.get_item(Key={"work_id": stem}).get("Item") or item')
    assert read_again > body.index('Key=md_key'), "the row is read after the text is stored"
    assert body.index("verdict = ((current.get") > read_again
    assert "verdict = ((item.get(" not in body, "the stale copy no longer decides the status"
    assert worker.UNCLASSIFIED == "fulltext_ready_unclassified"
