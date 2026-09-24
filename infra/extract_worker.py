"""Fargate extraction worker: PDF in S3 -> GROBID TEI XML + clean Markdown, one paper folder at a time.

Runs beside a GROBID container (localhost:8070) in one ECS task. Reads the job manifest named by
JOB_KEY (JSON list of stems), and for each stem reads papers/{stem}/original.pdf, writes
papers/{stem}/grobid.tei.xml and papers/{stem}/clean.md, and marks the DynamoDB item
fulltext_ready (or extract_failed with the reason). Idempotent: a stem whose clean.md already
matches the PDF hash is skipped unless FORCE=1.

PREPROCESS reads a PDF GROBID cannot through a derived copy, stored as papers/{stem}/derived.pdf
beside the untouched original (2026-09-23). `ocr` rasterises and OCRs every page with ocrmypdf, for
scans and for image-only pages such as a Science first-release PDF whose only text is its running
header. `shrink` rewrites the PDF with ghostscript at /ebook resolution, for a born-digital paper
whose images make it too large for GROBID (a 201 MB accepted manuscript failed with error 134). The
tools are installed into this container only when a job asks for them.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from xml.etree import ElementTree

import boto3
import requests

BUCKET = os.environ["BUCKET_NAME"]
TABLE = os.environ["TABLE_NAME"]
JOB_KEY = os.environ["JOB_KEY"]
GROBID = os.environ.get("GROBID_URL", "http://localhost:8070")
FORCE = os.environ.get("FORCE") == "1"
EXTRACTOR = os.environ.get("TEXT_EXTRACTOR", "grobid-0.8.2")
# Extracted, but its journal has not been checked against the allowlist yet, so note generation
# leaves it alone: `aws-pipeline-stems` selects on the statuses in EXTRACTED minus this one, and the
# Lambda's `source_note` action refuses anything that is not `fulltext_ready`.
UNCLASSIFIED = "fulltext_ready_unclassified"
EXTRACTED = ("fulltext_ready", UNCLASSIFIED, "model_draft", "draft_failed")
PREPROCESS = os.environ.get("PREPROCESS", "")
PREPROCESS_PACKAGES = {"ocr": ["ghostscript", "ocrmypdf", "tesseract-ocr-eng"], "shrink": ["ghostscript"]}
PREPROCESS_TOOL = {"ocr": "ocrmypdf", "shrink": "ghostscript"}
if PREPROCESS and PREPROCESS not in PREPROCESS_PACKAGES:
    raise SystemExit(f"PREPROCESS must be one of {sorted(PREPROCESS_PACKAGES)}, not {PREPROCESS!r}")

s3 = boto3.client("s3")
table = boto3.resource("dynamodb").Table(TABLE)


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc).replace(microsecond=0).isoformat()} {message}", flush=True)


def wait_for_grobid(timeout: int = 600) -> None:
    started = time.time()
    while time.time() - started < timeout:
        try:
            if requests.get(f"{GROBID}/api/isalive", timeout=5).text.strip() == "true":
                log("grobid is alive")
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise RuntimeError("GROBID did not come up")


# A figure drawn in LaTeX and exported through Inkscape carries its source in the PDF as
# `<latexit sha1_base64="...">`, and GROBID reads that blob as the figure's caption, one character
# per token. It is unreadable to a person and, being high-entropy nonsense, it is what the model's
# safety filter stops on: klein-2023-genot was 38.8% blob and every note attempt ended
# `content_filtered` after a handful of tokens (2026-09-23). Real prose never runs this many
# one-character tokens together; the longest genuine run measured in this corpus is math notation
# like "τ = 1.0 τ = 0.8", well under the threshold.
SPACED_BLOB = re.compile(r"(?:(?<=\s)|^)(?:\S ){%d,}\S(?=\s|$)" % 40)
# What is left of a caption that was nothing but blob is still blob, in pieces the run above is too
# short to catch. Measured on GENOT's 63 captions the two are far apart and nothing falls between:
# every blob remnant averages 1.07-1.16 characters per token, every real caption 3.4 or more, the
# shortest real one being "Proportion originating from Ngn3 High late" at 5.71.
BLOB_TOKEN_CHARS = 2.0
BLOB_MIN_TOKENS = 15


def drop_spaced_blobs(text: str) -> tuple[str, int]:
    """Remove the character-by-character blobs of a LaTeX figure, and say how many characters went."""
    cleaned = " ".join(SPACED_BLOB.sub(" ", text).split())
    tokens = cleaned.split()
    if len(tokens) >= BLOB_MIN_TOKENS and sum(map(len, tokens)) / len(tokens) < BLOB_TOKEN_CHARS:
        cleaned = ""
    return cleaned, len(text) - len(cleaned)


def node_text(node) -> str:
    return "" if node is None else " ".join("".join(node.itertext()).split())


def tei_to_markdown(xml_bytes: bytes, fallback_title: str) -> str:
    root = ElementTree.fromstring(xml_bytes)
    title = node_text(root.find(".//{*}titleStmt/{*}title")) or fallback_title
    lines = [f"# {title}", ""]
    abstract = node_text(root.find(".//{*}profileDesc/{*}abstract"))
    if abstract:
        lines.extend(["## Abstract", "", abstract, ""])
    body = root.find(".//{*}text/{*}body")
    if body is None:
        raise ValueError("TEI has no body")
    paragraphs = 0
    for node in body.iter():
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "head":
            text = node_text(node)
            if text:
                lines.extend([f"## {text}", ""])
        elif tag == "p":
            text = node_text(node)
            if text:
                lines.extend([text, ""])
                paragraphs += 1
        elif tag == "figure":
            head, _ = drop_spaced_blobs(node_text(node.find("{*}head")))
            desc, _ = drop_spaced_blobs(node_text(node.find("{*}figDesc")))
            if head or desc:
                lines.extend([f"*{head}: {desc}*".strip(), ""])
    if paragraphs == 0:
        raise ValueError("TEI has no body paragraphs")
    refs = root.findall(".//{*}listBibl/{*}biblStruct")
    if refs:
        lines.extend(["## References", ""])
        for ref in refs:
            ref_title = node_text(ref.find(".//{*}title"))
            year = node_text(ref.find(".//{*}date"))
            doi = node_text(ref.find(".//{*}idno[@type='DOI']"))
            if ref_title:
                lines.append(f"- {ref_title} ({year}) {doi}".rstrip())
        lines.append("")
    return "\n".join(lines)


def derived_key(stem: str) -> str:
    return f"papers/{stem}/derived.pdf"


def preprocess_command(source: str, target: str) -> list[str]:
    if PREPROCESS == "ocr":
        # --force-ocr because a scan often carries a publisher's stamp as its only text, and a page
        # with any text is otherwise skipped. One page at a time: the worker container has 2 GB, and
        # four pages of Ebert et al. 2021 at once were killed (-9) on 2026-09-23.
        return ["ocrmypdf", "--force-ocr", "-l", "eng", "--output-type", "pdf", "--jobs", "1", "--quiet",
                source, target]
    return ["gs", "-q", "-dNOPAUSE", "-dBATCH", "-dSAFER", "-sDEVICE=pdfwrite", "-dPDFSETTINGS=/ebook",
            f"-sOutputFile={target}", source]


def install_preprocess_tools() -> dict:
    if not PREPROCESS:
        return {}
    env = {**os.environ, "DEBIAN_FRONTEND": "noninteractive"}
    subprocess.run(["apt-get", "update", "-qq"], check=True, env=env)
    subprocess.run(["apt-get", "install", "-y", "-qq", "--no-install-recommends", *PREPROCESS_PACKAGES[PREPROCESS]],
                   check=True, env=env)
    versions = {"gs": subprocess.run(["gs", "--version"], capture_output=True, text=True).stdout.strip()}
    if PREPROCESS == "ocr":
        versions["ocrmypdf"] = subprocess.run(["ocrmypdf", "--version"], capture_output=True, text=True).stdout.strip()
    log(f"preprocess {PREPROCESS}: {versions}")
    return versions


TOOL_VERSIONS: dict = {}


def preprocess(stem: str, pdf: bytes, pdf_sha256: str) -> bytes:
    """Write the derived copy GROBID will read, and return its bytes. The original is not touched."""
    with tempfile.TemporaryDirectory() as tmp:
        source, target = f"{tmp}/in.pdf", f"{tmp}/out.pdf"
        with open(source, "wb") as handle:
            handle.write(pdf)
        done = subprocess.run(preprocess_command(source, target), capture_output=True, text=True, timeout=1800)
        if done.returncode != 0 or not os.path.exists(target):
            raise RuntimeError(f"{PREPROCESS} failed ({done.returncode}): {(done.stderr or '')[-300:]}")
        with open(target, "rb") as handle:
            derived = handle.read()
    s3.put_object(Bucket=BUCKET, Key=derived_key(stem), Body=derived, ContentType="application/pdf",
                  Metadata={"derived_from_sha256": pdf_sha256, "method": PREPROCESS,
                            "sha256": hashlib.sha256(derived).hexdigest(),
                            "tools": json.dumps(TOOL_VERSIONS)[:1000]})
    log(f"    {stem}: {PREPROCESS} {len(pdf)} -> {len(derived)} bytes")
    return derived


class GrobidGone(RuntimeError):
    """GROBID stopped answering. Every later paper in this shard would fail the same way."""


def grobid_alive(timeout: int = 5) -> bool:
    try:
        return requests.get(f"{GROBID}/api/isalive", timeout=timeout).text.strip() == "true"
    except requests.RequestException:
        return False


def extract(stem: str) -> dict:
    pdf_key = f"papers/{stem}/original.pdf"
    meta_key = f"papers/{stem}/meta.json"
    meta = json.loads(s3.get_object(Bucket=BUCKET, Key=meta_key)["Body"].read())
    pdf_sha256 = meta.get("pdf_sha256", "")
    item = table.get_item(Key={"work_id": stem}).get("Item") or {}
    if not FORCE and item.get("ingest_status") in EXTRACTED and item.get("extracted_pdf_sha256") == pdf_sha256:
        return {"stem": stem, "state": "skipped"}
    pdf = s3.get_object(Bucket=BUCKET, Key=pdf_key)["Body"].read()
    if pdf[:5] != b"%PDF-":
        raise ValueError("object is not a PDF")
    started = time.time()
    if PREPROCESS:
        pdf = preprocess(stem, pdf, pdf_sha256)
    try:
        response = requests.post(
            f"{GROBID}/api/processFulltextDocument",
            files={"input": (f"{stem}.pdf", pdf, "application/pdf")},
            data={"consolidateHeader": "0", "consolidateCitations": "0", "includeRawCitations": "1"},
            timeout=900,
        )
    except requests.RequestException as exc:
        # A big PDF can take GROBID's process down with it. Give it a moment to come back; if it
        # does not, stop the shard rather than marking every remaining paper as a parse failure.
        log(f"    {stem}: no answer from GROBID ({exc.__class__.__name__}); waiting for it to return")
        for _ in range(12):
            time.sleep(10)
            if grobid_alive():
                raise RuntimeError(f"GROBID dropped this PDF but recovered: {exc}") from None
        raise GrobidGone(f"GROBID stopped answering while processing {stem}") from None
    if response.status_code in (500, 503) and not grobid_alive():
        raise GrobidGone(f"GROBID returned {response.status_code} and is no longer alive")
    if response.status_code != 200:
        raise RuntimeError(f"GROBID HTTP {response.status_code}: {response.text[:200]}")
    tei = response.content
    markdown = tei_to_markdown(tei, meta.get("title") or stem)
    tei_key = f"papers/{stem}/grobid.tei.xml"
    md_key = f"papers/{stem}/clean.md"
    s3.put_object(Bucket=BUCKET, Key=tei_key, Body=tei, ContentType="application/xml")
    s3.put_object(Bucket=BUCKET, Key=md_key, Body=markdown.encode("utf-8"), ContentType="text/markdown; charset=utf-8")
    seconds = round(time.time() - started, 1)
    # The gate. A paper is handed to note generation only once its journal has been checked against
    # the allowlist; until then it waits here. Without this, anything dropped in the shared folder
    # became an evidence note at roughly $0.45 a paper with nobody having decided it belonged in
    # the corpus, which is how 450 papers were written up in September 2026. The OpenAlex route
    # always had this check (`pipeline.require_allowlisted_journal`); the upload route did not.
    # The row is read again here, not taken from `item`: storing the text above is what starts the
    # identity resolve in AWS, so by now a record with a verdict may already be on the row, and the
    # copy read before GROBID ran would park a paper that has just been cleared (2026-09-23).
    current = table.get_item(Key={"work_id": stem}).get("Item") or item
    verdict = ((current.get("record") or {}).get("corpus") or {}).get("journal_verdict")
    status = "fulltext_ready" if verdict == "include" else UNCLASSIFIED
    # `extracted_pdf_sha256` stays the original's, which is what identifies the paper; the copy that
    # was read is named separately, and the extractor says how the text was obtained.
    extractor = f"{EXTRACTOR}+{PREPROCESS_TOOL[PREPROCESS]}" if PREPROCESS else EXTRACTOR
    expression = ("SET ingest_status = :st, source_key = :src, tei_key = :tei, grobid_sha256 = :gh, "
                  "extracted_pdf_sha256 = :ph, text_extractor = :ex, text_extracted_date = :dt, "
                  "extract_seconds = :sec, extract_error = :none")
    values = {":st": status, ":src": md_key, ":tei": tei_key,
              ":gh": hashlib.sha256(tei).hexdigest(), ":ph": pdf_sha256, ":ex": extractor,
              ":dt": datetime.now(timezone.utc).date().isoformat(), ":sec": str(seconds), ":none": ""}
    if PREPROCESS:
        expression += ", text_input_key = :in, text_preprocess = :pre"
        values |= {":in": derived_key(stem), ":pre": PREPROCESS}
    table.update_item(Key={"work_id": stem}, UpdateExpression=expression, ExpressionAttributeValues=values)
    return {"stem": stem, "state": "extracted", "seconds": seconds, "chars": len(markdown),
            "ingest_status": status}


def main() -> None:
    stems = json.loads(s3.get_object(Bucket=BUCKET, Key=JOB_KEY)["Body"].read())
    log(f"job {JOB_KEY}: {len(stems)} papers")
    TOOL_VERSIONS.update(install_preprocess_tools())
    wait_for_grobid()
    summary = {"job": JOB_KEY, "extracted": 0, "skipped": 0, "failed": 0, "abandoned": 0}
    for index, stem in enumerate(stems, 1):
        try:
            result = extract(stem)
            summary[result["state"]] += 1
            log(f"[{index}/{len(stems)}] {stem} {result['state']} {result.get('seconds', '')}")
        except GrobidGone as exc:
            # Leave the rest of the shard as pdf_uploaded so a later run picks it up untouched.
            summary["abandoned"] = len(stems) - index + 1
            log(f"[{index}/{len(stems)}] {exc}; abandoning {summary['abandoned']} papers for a later run")
            break
        except Exception as exc:
            summary["failed"] += 1
            log(f"[{index}/{len(stems)}] {stem} failed: {exc}")
            try:
                table.update_item(Key={"work_id": stem},
                                  UpdateExpression="SET ingest_status = :st, extract_error = :err",
                                  ExpressionAttributeValues={":st": "extract_failed", ":err": str(exc)[:500]})
            except Exception as inner:
                log(f"    could not record failure: {inner}")
    s3.put_object(Bucket=BUCKET, Key=JOB_KEY.replace("jobs/", "jobs/done/"), Body=json.dumps(summary).encode("utf-8"),
                  ContentType="application/json")
    log(f"done {summary}")


if __name__ == "__main__":
    main()
