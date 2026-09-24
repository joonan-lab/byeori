"""AWS-only wiki access. Clients receive outlines and bounded AWS-selected excerpts."""
from __future__ import annotations

import re
from typing import Any

from .aws_store import AwsStore
from .config import Settings

DOC_TYPES = {"note": "wiki/sources", "paper": "wiki/papers", "overview": "wiki/overviews",
             "concept": "wiki/concepts", "question": "wiki/questions"}


def _require_aws(settings: Settings, backend: str = "auto") -> None:
    if backend not in ("auto", "aws"):
        raise ValueError("The local wiki backend was removed; use AWS")
    if not settings.aws_bucket:
        raise RuntimeError("AWS_KIRO_WIKI_BUCKET is not configured; no local fallback is available")


def _check_doc_id(doc_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*", doc_id) or ".." in doc_id.split("/"):
        raise ValueError("doc_id must be a work ID, stem, slug, or category/slug")


def search(settings: Settings, query: str, *, limit: int = 10, doc_type: str | None = None,
           category: str | None = None, backend: str = "auto", store: AwsStore | None = None) -> dict[str, Any]:
    _require_aws(settings, backend)
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be 1 to 100")
    if doc_type is not None and doc_type not in DOC_TYPES:
        raise ValueError(f"doc_type must be one of {', '.join(DOC_TYPES)}")
    if category is not None and not re.fullmatch(r"[a-z0-9-]+", category):
        raise ValueError("category must be a lowercase slug")
    result = (store or AwsStore(settings)).wiki_search(query, limit=limit, doc_type=doc_type, category=category)
    # Older deployed indexes label S3 objects with the retired mirror's paths.
    for hit in result.get("results", []):
        path = hit.get("path", "")
        if path.startswith("data/sources/"):
            hit["path"] = f"s3://{settings.aws_bucket}/wiki/{path[5:]}"
        elif path.startswith("data/wiki/"):
            hit["path"] = f"s3://{settings.aws_bucket}/{path[5:]}"
    return {**result, "backend": "aws"}


def backlinks(settings: Settings, doc_type: str, doc_id: str, *, backend: str = "auto",
              store: AwsStore | None = None) -> dict[str, Any]:
    _require_aws(settings, backend)
    if doc_type not in DOC_TYPES:
        raise ValueError(f"doc_type must be one of {', '.join(DOC_TYPES)}")
    _check_doc_id(doc_id)
    return {**(store or AwsStore(settings)).wiki_backlinks(doc_type, doc_id), "backend": "aws"}


def read_page(settings: Settings, doc_type: str, doc_id: str, *, section: str | None = None,
              start: int = 0, max_chars: int = 4000, backend: str = "auto",
              store: AwsStore | None = None) -> dict[str, Any]:
    """Request an outline or a bounded section; AWS resolves and processes the page."""
    _require_aws(settings, backend)
    if doc_type not in DOC_TYPES:
        raise ValueError(f"doc_type must be one of {', '.join(DOC_TYPES)}")
    _check_doc_id(doc_id)
    return (store or AwsStore(settings)).wiki_read(
        doc_type, doc_id, section=section, start=start, max_chars=max_chars)


def categories(settings: Settings, *, store: AwsStore | None = None) -> dict[str, Any]:
    """Request category counts computed from the index inside AWS."""
    _require_aws(settings)
    return {**(store or AwsStore(settings)).wiki_categories(), "backend": "aws"}


def save_question(settings: Settings, *, title: str, question: str, sharper_followup: str, holdings: str,
                  tentative_answer: str, related: list[str], tags: list[str], author: str | None = None,
                  store: AwsStore | None = None) -> dict[str, Any]:
    """Publish a research question in S3, refusing to overwrite an existing page."""
    if not settings.aws_bucket:
        raise RuntimeError("AWS_KIRO_WIKI_BUCKET is not configured; the wiki lives only in S3")
    store = store or AwsStore(settings)
    from datetime import UTC, datetime
    import json as _json
    title = title.strip()
    if not title.endswith("?"):
        raise ValueError("a question page's title is a question and ends with '?'")
    if not all(x.strip() for x in (question, sharper_followup, holdings, tentative_answer)):
        raise ValueError("question, sharper_followup, holdings, and tentative_answer must not be empty")
    if not related:
        raise ValueError("a question page must link at least one wiki page")
    today = datetime.now(UTC).date().isoformat()
    # Who wrote this comes from the credentials in use, not from something a caller typed.
    if author is None and (settings.aws_bucket or settings.aws_table):
        author = (store or AwsStore(settings)).caller_name()
    author = author or "unknown"
    if not re.fullmatch(r"[A-Za-z0-9._@-]+", author):
        raise ValueError("author must be an IAM user name")
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:80] or "question"
    page = "\n".join([
        "---",
        f"title: {_json.dumps(title, ensure_ascii=False)}",
        'category: "questions"',
        f"tags: [{', '.join(tags)}]",
        f"created: {today}",
        f"updated: {today}",
        f"author: {_json.dumps(author)}",
        "---",
        "",
        "## Question",
        question.strip(),
        "",
        "## Sharper follow-up",
        sharper_followup.strip(),
        "",
        "## What the knowledge base holds",
        holdings.strip(),
        "",
        "## Tentative answer from the knowledge base",
        tentative_answer.strip(),
        "",
        "## Related Pages",
        *(f"- [[{link}]]" for link in related),
        "",
    ])
    key = f"wiki/questions/{slug}.md"
    published = store.put_text(key, page, create_only=True)
    return {"doc_type": "question", "doc_id": slug,
            "path": f"s3://{settings.aws_bucket}/{key}", "s3": published}
