"""Publisher supplementary files beside each paper's original, and one section in its note.

The lab's supplementary material (``llm-wiki/papers-supplementary/{stem}/``) is where gene lists,
DE and association tables, cohort sheets and the long methods live. A local reading of every
folder (2026-09-23) decided per file whether it is worth keeping and wrote what each kept file
holds. This module publishes that result:

    papers/{stem}/supplementary/{publisher filename}   the kept files, bytes unchanged
    papers/{stem}/supplementary/README.md               the file guide, linking back to the note
    papers/{stem}/supplementary/manifest.json           sha256, size, kind and decision per file

and then gives the evidence note ``wiki/sources/{stem}.md`` one section, ``## Supplementary
Files``, that says what the supplement offers and where the guide is. That section is what the
index reads, so a search for a cell-type DEG table or a cohort sheet finds the paper.

``assets/`` next door holds the figure crops of the article itself and is never touched here;
neither are ``original.pdf``, ``clean.md``, ``grobid.tei.xml`` and ``meta.json``.

**Originals are never overwritten.** A kept file is put with ``IfNoneMatch='*'``. An object already
at the key with the same sha256 counts as already stored, which makes a run resumable; one with a
different sha256 is reported as a conflict and left alone. The guide and the manifest are ours and
may be replaced by a later run.

**Identity is checked, not assumed.** The reading recorded ``identity`` per paper. ``confirmed``
(a publisher filename code, a first-page title or a DOI in the document matched) is published;
``probable`` (only the folder name linked them) is published only when the DOI in the lab's own
manifest equals the DOI Byeori stores for the paper. Anything else is reported and not published.

**The note is edited once.** The section goes in before ``## 랩 질문`` when that section exists,
because the lab-question linker appends inside the last section of the page; it is written back
with the ETag of the read, and a note that already has the heading is left alone.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from collections.abc import Callable, Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from botocore.exceptions import ClientError

from byeori.wiki_question_links import HEADING as LAB_HEADING

__all__ = [
    "NOTE_HEADING", "NOTE_LIMIT", "build_manifest", "check_identity", "check_note_section", "guide_key",
    "insert_section", "iter_manifest_keys", "link_notes", "manifest_key", "member_key", "prefix", "publish_paper",
]

NOTE_HEADING = "## Supplementary Files"
NOTE_LIMIT = 1200
CONFLICT = frozenset({"PreconditionFailed", "ConditionalRequestConflict"})
MISSING = frozenset({"NoSuchKey", "404", "NotFound"})
PUBLISHABLE = frozenset({"confirmed", "probable"})


def prefix(stem: str) -> str:
    return f"papers/{stem}/supplementary/"


def guide_key(stem: str) -> str:
    return prefix(stem) + "README.md"


def manifest_key(stem: str) -> str:
    return prefix(stem) + "manifest.json"


def member_key(stem: str, relative: str) -> str:
    """The key of one kept file; a path that could leave the folder is refused."""
    path = PurePosixPath(relative)
    ours = len(path.parts) == 1 and path.name in {"README.md", "manifest.json"}
    if not relative or "\\" in relative or path.is_absolute() or ".." in path.parts or ours:
        raise ValueError(f"unsafe supplementary path: {relative!r}")
    return prefix(stem) + path.as_posix()


def normal_doi(doi: str | None) -> str:
    text = (doi or "").strip().lower()
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", text)


def check_identity(triage: dict[str, Any], byeori_doi: str | None, lab_manifest_doi: str | None) -> str | None:
    """Why this paper may be published (``None`` when it may not)."""
    identity = triage.get("identity")
    if identity == "confirmed":
        return "confirmed"
    if identity == "probable" and normal_doi(byeori_doi) and normal_doi(byeori_doi) == normal_doi(lab_manifest_doi):
        return "probable_doi_match"
    return None


def check_note_section(text: str, stem: str) -> list[str]:
    """Problems with a note section as written by the reading; an empty list means it can be used."""
    problems = []
    body = text.strip()
    if not body.startswith(NOTE_HEADING + "\n"):
        problems.append("does not start with the heading")
    if body.count("\n## ") or body.count(NOTE_HEADING) != 1:
        problems.append("holds more than one section")
    if len(body) > NOTE_LIMIT:
        problems.append(f"{len(body)} characters, over {NOTE_LIMIT}")
    if guide_key(stem) not in body:
        problems.append("does not name the file guide")
    return problems


def insert_section(text: str, section: str) -> str:
    """The note with the section added once; the lab-question section, if any, stays last."""
    if NOTE_HEADING in text:
        return text
    block = section.strip()
    body = text.rstrip("\n")
    if LAB_HEADING in body:
        head, _, tail = body.rpartition(LAB_HEADING)
        return f"{head.rstrip()}\n\n{block}\n\n{LAB_HEADING}{tail}\n"
    return f"{body}\n\n{block}\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_type(name: str) -> str:
    guessed, _ = mimetypes.guess_type(name)
    return guessed or "application/octet-stream"


def build_manifest(stem: str, triage: dict[str, Any], *, doi: str, identity_basis: str,
                   note_section: str) -> dict[str, Any]:
    """The machine record of one paper's supplement: every held file, kept or not, and why."""
    files = []
    for entry in triage.get("files", []):
        kept = entry.get("decision") == "upload"
        record = {key: entry.get(key) for key in ("file", "sha256", "bytes", "kind", "decision", "reason",
                                                    "label", "uses", "gene_set_threshold", "summary")
                  if entry.get(key) not in (None, "", [])}
        if kept:
            record["key"] = member_key(stem, entry["file"])
        files.append(record)
    return {
        "schema": "byeori-supplementary-v1",
        "stem": stem,
        "doi": doi,
        "identity": triage.get("identity"),
        "identity_basis": identity_basis,
        "source_note": f"wiki/sources/{stem}.md",
        "guide": guide_key(stem),
        "files_kept": sum(1 for f in files if f.get("decision") == "upload"),
        "files_not_kept": sum(1 for f in files if f.get("decision") != "upload"),
        "missing_publisher_files": triage.get("missing_publisher_files"),
        "read_by": "claude-code local agent",
        "note_section": note_section.strip() + "\n",
        "files": files,
    }


def _error_code(exc: ClientError) -> str:
    return exc.response.get("Error", {}).get("Code", "")


def _stored_sha(s3: Any, bucket: str, key: str) -> str | None:
    """The sha256 recorded on an existing object, ``""`` when it has none, ``None`` when absent."""
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if _error_code(exc) in MISSING:
            return None
        raise
    return (head.get("Metadata") or {}).get("sha256", "")


def publish_paper(s3: Any, bucket: str, stem: str, *, triage: dict[str, Any], guide: str, note_section: str,
                  files_root: Path, lab_manifest_doi: str | None, apply: bool,
                  hasher: Callable[[Path], str] = sha256_file) -> dict[str, Any]:
    """Put one paper's kept files, guide and manifest. Without ``apply`` it only reads and reports."""
    kept = [entry for entry in triage.get("files", []) if entry.get("decision") == "upload"]
    if not kept:
        return {"stem": stem, "outcome": "nothing_to_keep"}
    try:
        meta = json.loads(s3.get_object(Bucket=bucket, Key=f"papers/{stem}/meta.json")["Body"].read())
    except ClientError:
        return {"stem": stem, "outcome": "unmatched_paper"}
    doi = meta.get("doi") or ""
    basis = check_identity(triage, doi, lab_manifest_doi)
    if basis is None:
        return {"stem": stem, "outcome": "identity_unconfirmed", "identity": triage.get("identity")}
    problems = check_note_section(note_section, stem)
    if problems:
        return {"stem": stem, "outcome": "note_section_invalid", "problems": problems}

    planned = []
    for entry in kept:
        path = files_root / entry["file"]
        if not path.is_file() or path.stat().st_size != entry.get("bytes") or hasher(path) != entry.get("sha256"):
            return {"stem": stem, "outcome": "changed_since_read", "file": entry["file"]}
        planned.append((entry, path, member_key(stem, entry["file"])))

    report: dict[str, Any] = {"stem": stem, "identity": basis, "files": len(planned),
                              "bytes": sum(entry["bytes"] for entry, _, _ in planned),
                              "stored": 0, "already": 0, "conflicts": []}
    if not apply:
        report["outcome"] = "would_publish"
        return report

    for entry, path, key in planned:
        existing = _stored_sha(s3, bucket, key)
        if existing == entry["sha256"]:
            report["already"] += 1
            continue
        if existing is not None:
            report["conflicts"].append(key)
            continue
        try:
            with path.open("rb") as handle:
                s3.put_object(Bucket=bucket, Key=key, Body=handle, ContentType=content_type(path.name),
                              Metadata={"sha256": entry["sha256"]}, IfNoneMatch="*")
        except ClientError as exc:
            if _error_code(exc) in CONFLICT:
                report["conflicts"].append(key)
                continue
            raise
        report["stored"] += 1

    if report["conflicts"]:
        # A kept file could not be stored as read, so the guide would describe bytes that are not
        # there. Leave the guide and manifest unwritten until the conflict is looked at.
        report["outcome"] = "conflict"
        return report
    manifest = build_manifest(stem, triage, doi=doi, identity_basis=basis, note_section=note_section)
    s3.put_object(Bucket=bucket, Key=guide_key(stem), Body=guide.encode("utf-8"),
                  ContentType="text/markdown; charset=utf-8")
    s3.put_object(Bucket=bucket, Key=manifest_key(stem),
                  Body=(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n").encode("utf-8"),
                  ContentType="application/json")
    report["outcome"] = "published"
    return report


def iter_manifest_keys(s3: Any, bucket: str) -> Iterator[str]:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix="papers/"):
        for item in page.get("Contents", ()):
            if item["Key"].endswith("/supplementary/manifest.json"):
                yield item["Key"]


def link_notes(s3: Any, bucket: str, *, stems: list[str] | None = None, dry_run: bool = True,
               limit: int | None = None) -> dict[str, Any]:
    """Give each published paper's evidence note its section, from the manifest in S3 only."""
    report: dict[str, Any] = {"dry_run": dry_run, "scanned": 0, "already_linked": 0, "linked": 0,
                              "no_note": [], "conflicts": [], "errors": [], "samples": []}
    keys = [manifest_key(stem) for stem in stems] if stems else iter_manifest_keys(s3, bucket)
    for key in keys:
        if limit is not None and report["scanned"] >= limit:
            break
        report["scanned"] += 1
        stem = key[len("papers/"):-len("/supplementary/manifest.json")]
        try:
            manifest = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
            section = manifest["note_section"]
        except (ClientError, KeyError, ValueError) as exc:
            report["errors"].append({"stem": stem, "error": type(exc).__name__})
            continue
        note_key = f"wiki/sources/{stem}.md"
        try:
            response = s3.get_object(Bucket=bucket, Key=note_key)
            text = response["Body"].read().decode("utf-8")
        except ClientError as exc:
            if _error_code(exc) in MISSING:
                report["no_note"].append(stem)
            else:
                report["errors"].append({"stem": stem, "error": _error_code(exc)})
            continue
        if NOTE_HEADING in text:
            report["already_linked"] += 1
            continue
        if not dry_run:
            try:
                s3.put_object(Bucket=bucket, Key=note_key, Body=insert_section(text, section).encode("utf-8"),
                              ContentType="text/markdown; charset=utf-8", IfMatch=response.get("ETag", ""))
            except ClientError as exc:
                code = _error_code(exc)
                (report["conflicts"] if code in CONFLICT else report["errors"]).append({"stem": stem, "error": code})
                continue
        report["linked"] += 1
        if len(report["samples"]) < 3:
            report["samples"].append({"key": note_key, "section": section})
    return report
