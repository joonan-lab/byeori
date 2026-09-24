"""Give each indexed wiki page one standing line down to its lab questions.

An answered student question is kept as Markdown under ``wiki/lab-questions/`` and its hub,
``wiki/lab-questions/by-page/{link}.md``, lists every question that cited a page. That is the way
down from a question to its evidence. This module writes the way back: one line, once, in the
indexed page itself.

Why one line rather than one line per question: a source note is indexed, so every edit to it is
re-indexed and every line lengthens what a search reads. The hub absorbs the growth instead, and
the note stays the same length however many questions cite it. A note is edited at most once in
its life by this tool, and re-running it changes nothing.

Why an administrator tool rather than the answer worker: ``wiki/sources/`` is scientific content
and the answer worker is a path eighteen students can trigger. Its IAM role may put exactly
``wiki/lab-questions/*`` and nothing else under ``wiki/``, and that stays true. This runs with the
administrator's own credentials, like ``aws-openalex-match``.

The edit itself is an appended section at the end of the page, under the heading ``## 랩 질문``,
written back with the ETag of the read, so a page someone changed in between is reported and
skipped rather than overwritten.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from botocore.exceptions import ClientError

from byeori.lab_pages import ANSWER_PAGE_PREFIX, page_link_line

__all__ = ["HEADING", "INDEXED_FOLDERS", "add_line", "iter_pages", "link_pages", "needs_line"]

HEADING = "## 랩 질문"
# The layers the index reads. ``wiki/lab-questions/`` is not one of them and is never edited here.
INDEXED_FOLDERS = ("wiki/sources/", "wiki/concepts/", "wiki/overviews/", "wiki/questions/")
CONFLICT = frozenset({"PreconditionFailed", "ConditionalRequestConflict"})


def iter_pages(s3: Any, bucket: str, folders: tuple[str, ...] = INDEXED_FOLDERS) -> Iterator[str]:
    """Every Markdown page key under the indexed folders, in listing order."""
    paginator = s3.get_paginator("list_objects_v2")
    for folder in folders:
        for page in paginator.paginate(Bucket=bucket, Prefix=folder):
            for item in page.get("Contents", ()):
                key = item["Key"]
                if key.endswith(".md") and not key.startswith(ANSWER_PAGE_PREFIX) and "/failed/" not in key:
                    yield key


def needs_line(text: str, link: str) -> bool:
    """False when this page already carries its line, so a re-run is a no-op."""
    return page_link_line(link) not in text


def add_line(text: str, link: str) -> str:
    """The page with its standing line appended under ``HEADING``; existing content is untouched."""
    body = text.rstrip("\n")
    if HEADING in body:
        # The section exists from an earlier run with a different link; add this one inside it.
        head, _, tail = body.rpartition(HEADING)
        return f"{head}{HEADING}{tail.rstrip()}\n{page_link_line(link)}\n"
    return f"{body}\n\n{HEADING}\n\n{page_link_line(link)}\n"


def link_pages(s3: Any, bucket: str, *, dry_run: bool = True, limit: int | None = None,
               folders: tuple[str, ...] = INDEXED_FOLDERS) -> dict[str, Any]:
    """Add the line to every page that lacks it. ``dry_run`` reads and reports without writing."""
    report: dict[str, Any] = {"dry_run": dry_run, "scanned": 0, "already_linked": 0, "linked": 0,
                              "conflicts": [], "errors": [], "samples": []}
    for key in iter_pages(s3, bucket, folders):
        if limit is not None and report["scanned"] >= limit:
            break
        report["scanned"] += 1
        link = key[len("wiki/"):-len(".md")]
        try:
            response = s3.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            try:
                text = body.read().decode("utf-8")
            finally:
                body.close()
        except ClientError as exc:
            report["errors"].append({"key": key, "error": exc.response.get("Error", {}).get("Code", "")})
            continue
        if not needs_line(text, link):
            report["already_linked"] += 1
            continue
        if dry_run:
            report["linked"] += 1
            if len(report["samples"]) < 3:
                report["samples"].append({"key": key, "line": page_link_line(link)})
            continue
        try:
            s3.put_object(Bucket=bucket, Key=key, Body=add_line(text, link).encode("utf-8"),
                          ContentType="text/markdown; charset=utf-8", IfMatch=response.get("ETag", ""))
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            (report["conflicts"] if code in CONFLICT else report["errors"]).append({"key": key, "error": code})
            continue
        report["linked"] += 1
        if len(report["samples"]) < 3:
            report["samples"].append({"key": key, "line": page_link_line(link)})
    return report
