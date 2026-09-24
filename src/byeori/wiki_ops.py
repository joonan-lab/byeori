"""AWS-side wiki reads, validation and review. Clients receive bounded results only.

Called by the existing ingest Lambda with its S3 client, DynamoDB table and cached AWS index.
The complete page and SQLite index never leave the worker through these operations.
"""
from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace

from botocore.exceptions import ClientError

from .promote import promote_draft_in_worker
from .synthesis_manifest import parse_frontmatter
from .validation import page_errors

ACTIONS = {"wiki_read", "read_text", "wiki_categories", "wiki_validate", "wiki_metrics", "promote_draft",
           "pipeline_failures", "corpus_status", "synthesis_coverage", "notes_in_category"}
DOC_TYPES = {"note": "sources", "paper": "papers", "overview": "overviews",
             "concept": "concepts", "question": "questions"}
MAX_CHARS = 8000
PAGE_STEMS = 1000
MAX_RESPONSE_BYTES = 128 * 1024
HEADING = re.compile(r"^## (.+)$", re.M)
NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:\s?%|×)?(?![\w])")
COMPARISON_NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:\s?%|×|x)?(?![\w])")
WIKILINK = re.compile(r"\[\[([^\]|#]+)")
GENE = re.compile(r"\b[A-Z][A-Z0-9]{2,7}\b")
STOP_GENES = {"ASD", "NDD", "ADHD", "WGS", "WES", "DNA", "RNA", "SNV", "CNV", "SNP", "FDR", "MAF", "QC", "PCR",
              "DSM", "ID", "DD", "GI", "IQ", "CI", "OR", "HR", "MRI", "EEG", "USA", "UK", "NIH", "GWAS", "PRS",
              "LOF", "LGD", "VUS", "ACMG", "HPO", "GO", "KEGG", "MSSNG", "SSC", "SFARI", "SPARK", "AGRE", "ACGC"}


def _key(key):
    if (not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9._/-]+", key)
            or any(p in {"", ".", ".."} for p in key.split("/"))
            or not key.startswith(("wiki/", "sources/")) or not key.endswith(".md")):
        raise ValueError("Expected a Markdown key under wiki/ or sources/")
    return key


def _published(key):
    return (key.startswith("wiki/") and len(key.split("/")) >= 3 and key.endswith(".md")
            and not key.startswith("wiki/drafts/") and "/failed/" not in key)


def _read(s3, bucket, key):
    key = _key(key)
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in {"NoSuchKey", "404"}:
            raise FileNotFoundError(f"S3 page not found: s3://{bucket}/{key}; the search index may be stale") from exc
        raise
    body = response["Body"]
    try:
        raw = body.read()
    finally:
        body.close()
    return raw.decode("utf-8"), response.get("ETag", ""), hashlib.sha256(raw).hexdigest()


def _window(event):
    start = event.get("start", 0)
    size = event.get("max_chars", 4000)
    if (type(start) is not int or start < 0 or type(size) is not int or not 1 <= size <= MAX_CHARS):
        raise ValueError(f"start must be a non-negative integer and max_chars must be 1 to {MAX_CHARS}")
    return start, size


def _excerpt(text, event):
    start, size = _window(event)
    if start > len(text):
        raise ValueError("start exceeds the selected section length")
    end = min(start + size, len(text))
    return {"text": text[start:end], "start": start, "next_start": end,
            "has_more": end < len(text), "total_chars": len(text)}


def _page_key(event, table, index, bucket):
    kind, ident = event.get("doc_type"), event.get("doc_id")
    if (kind not in DOC_TYPES or not isinstance(ident, str)
            or not re.fullmatch(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*", ident)
            or any(p in {".", "..", "failed"} for p in ident.split("/"))):
        raise ValueError("Invalid wiki document type or ID")
    key = f"wiki/{DOC_TYPES[kind]}/{ident}.md"
    if kind == "paper":
        item = table.get_item(Key={"work_id": ident}).get("Item") or {}
        if item.get("page_key"):
            key = item["page_key"]
        elif re.fullmatch(r"[a-z0-9-]+", item.get("category") or ""):
            key = f"wiki/{item['category']}/{ident}.md"
        else:
            con, _ = index()
            try:
                row = con.execute("SELECT path FROM docs WHERE doc_type = ? AND doc_id = ?", (kind, ident)).fetchone()
            finally:
                con.close()
            if row:
                path = row[0]
                key = path.removeprefix(f"s3://{bucket}/").removeprefix("data/")
    _key(key)
    if not _published(key):
        raise ValueError("The recorded key is not a published wiki page")
    return key


def _wiki_read(event, s3, table, bucket, index):
    _window(event)
    key = _page_key(event, table, index, bucket)
    text, etag, digest = _read(s3, bucket, key)
    fields, body = parse_frontmatter(text)
    parts = HEADING.split(body)
    sections = [(name.strip(), content.strip()) for name, content in zip(parts[1::2], parts[2::2])]
    opening = re.sub(r"^# .+$", "", parts[0], flags=re.M).strip()
    if opening:
        label = "(opening)"
        while any(name.casefold() == label for name, _ in sections):
            label = "(" + label + ")"
        sections.insert(0, (label, opening))
    result = {"doc_type": event["doc_type"], "doc_id": event["doc_id"], "path": f"s3://{bucket}/{key}",
              "etag": etag, "sha256": digest,
              "metadata": {k: str(fields[k])[:1000] for k in ("title", "doi", "journal", "year", "publication_year", "review_status") if k in fields},
              "sections": [name[:200] for name, _ in sections[:128]]}
    section = event.get("section")
    if section is None:
        # The default response is an outline, never the entire body.
        return {**result, "text": "", "mode": "outline"}
    if not isinstance(section, str) or not section.strip() or len(section) > 200:
        raise ValueError("section must name a heading from the outline")
    for name, content in sections:
        if name.casefold() == section.strip().casefold():
            return {**result, "mode": "section", "section": name, **_excerpt(content, event)}
    raise ValueError(f"Section {section!r} not found; request the page outline first")


def _categories(index):
    con, etag = index()
    try:
        rows = con.execute("SELECT category, doc_type, count(*) FROM docs GROUP BY category, doc_type ORDER BY 3 DESC").fetchall()
    finally:
        con.close()
    counts = {}
    for category, kind, count in rows:
        counts.setdefault(category, {})[kind] = count
    return {"categories": counts, "total": sum(sum(c.values()) for c in counts.values()), "index_etag": etag}


def _synthesis_coverage(index):
    """Per category, how many notes a concept or overview cites, and how many nothing cites.

    llm-wiki keeps this as the number its root index reports for every category, and holds it at
    100% with none or one orphan; Byeori had the whole-corpus count in `runs/synthesis/orphans.json`
    and no way to see where the gap is (user, 2026-09-23). A category with many orphans is where
    synthesis is missing, which is what decides the next question to ask.
    """
    con, etag = index()
    try:
        rows = con.execute(
            "SELECT d.category, count(*), "
            "  sum(CASE WHEN d.doc_id IN (SELECT to_id FROM links WHERE to_type = 'note' "
            "      AND from_type IN ('concept', 'overview')) THEN 1 ELSE 0 END) "
            "FROM docs d WHERE d.doc_type = 'note' GROUP BY d.category ORDER BY 2 DESC").fetchall()
    finally:
        con.close()
    categories = [{"category": category or "other", "notes": notes, "connected": connected or 0,
                   "orphans": notes - (connected or 0),
                   "coverage": round((connected or 0) / notes, 3) if notes else 0.0}
                  for category, notes, connected in rows]
    notes = sum(c["notes"] for c in categories)
    connected = sum(c["connected"] for c in categories)
    return {"categories": categories, "notes": notes, "connected": connected,
            "orphans": notes - connected,
            "coverage": round(connected / notes, 3) if notes else 0.0,
            "index_etag": etag, "execution": "aws"}


def _notes_in_category(event, index):
    """The stems filed under one category, which is how a caller works through `other`."""
    category = str(event.get("category") or "")
    if not re.fullmatch(r"[a-z0-9-]+", category):
        raise ValueError("category must be a lowercase slug such as other or liver")
    limit, offset = event.get("limit") or 0, event.get("offset") or 0
    if type(limit) is not int or limit < 0 or type(offset) is not int or offset < 0:
        raise ValueError("limit and offset must be nonnegative integers")
    # Stems only, and a page at a time: the whole of `other` was 1,336 rows and the titles took the
    # response past the 128 KiB budget (2026-09-23). Whoever needs a title reads the note.
    window = min(limit, PAGE_STEMS) if limit else PAGE_STEMS
    con, etag = index()
    try:
        total = con.execute("SELECT count(*) FROM docs WHERE doc_type = 'note' AND category = ?",
                            (category,)).fetchone()[0]
        rows = con.execute("SELECT doc_id FROM docs WHERE doc_type = 'note' AND category = ? "
                           "ORDER BY doc_id LIMIT ? OFFSET ?", (category, window, offset)).fetchall()
    finally:
        con.close()
    returned = offset + len(rows)
    return {"category": category, "stems": [r[0] for r in rows], "count": len(rows), "total": total,
            "offset": offset, "next_offset": returned if returned < total else None,
            "index_etag": etag, "execution": "aws"}


def _validate(event, s3, bucket):
    request = {"Bucket": bucket, "Prefix": "wiki/", "MaxKeys": 10}
    if event.get("cursor"):
        request["ContinuationToken"] = event["cursor"]
    listing = s3.list_objects_v2(**request)
    checked, errors = 0, []
    for obj in listing.get("Contents", []):
        key = obj["Key"]
        if not _published(key):
            continue
        checked += 1
        try:
            text, _, _ = _read(s3, bucket, key)
            errors.extend(f"s3://{bucket}/{key}: {error}" for error in page_errors(key, text))
        except FileNotFoundError:
            errors.append(f"s3://{bucket}/{key}: page disappeared during validation")
    return {"checked": checked, "errors": errors, "next_cursor": listing.get("NextContinuationToken")}


def _metrics(event, s3, bucket):
    key = _key(event.get("key"))
    if not _published(key):
        raise ValueError("Metrics require a published wiki page")
    text, etag, digest = _read(s3, bucket, key)
    fields, body = parse_frontmatter(text)
    values = {m.group(0) for m in NUMBER.finditer(body)}
    result = {"key": key, "etag": etag, "sha256": digest,
              "metrics": {"chars": len(body), "numbers": len(values), "wikilinks": len(set(WIKILINK.findall(body)))}}
    if event.get("compare"):
        result["numbers"] = sorted({m.group(0).replace(",", "").rstrip("×x ").strip()
                                    for m in COMPARISON_NUMBER.finditer(body)} - {"1", "2", "3", "4", "5"})
        result["genes"] = sorted(set(GENE.findall(body)) - STOP_GENES)
        result["provenance"] = {k: str(v)[:200] for k, v in fields.items() if k.startswith("ingest_")}
    return result


class _ReviewStore:
    """AWS-native adapter for the shared promotion checks; never used by clients."""
    def __init__(self, s3, table, bucket):
        self.s3, self.table, self.bucket = s3, table, bucket

    def get_item(self, work_id):
        return self.table.get_item(Key={"work_id": work_id}).get("Item") or {}

    def get_text(self, key):
        return _read(self.s3, self.bucket, key)[0]

    def put_text(self, key, text, *, create_only=False):
        _key(key)
        options = {"IfNoneMatch": "*"} if create_only else {}
        try:
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=text.encode(),
                               ContentType="text/markdown; charset=utf-8", **options)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in {"PreconditionFailed", "412"}:
                raise FileExistsError(key) from exc
            raise
        return {"bucket": self.bucket, "key": key}

    def mark_reviewed(self, work_id, fields, *, expected_draft_sha256):
        names = {f"#r{i}": name for i, name in enumerate(fields)}
        values = {f":r{i}": value for i, value in enumerate(fields.values())}
        values.update({":expected_draft": expected_draft_sha256, ":model_draft": "model_draft"})
        try:
            self.table.update_item(Key={"work_id": work_id},
                UpdateExpression="SET " + ", ".join(f"{name} = :r{name[2:]}" for name in names),
                ConditionExpression="draft_sha256 = :expected_draft AND ingest_status = :model_draft",
                ExpressionAttributeNames=names, ExpressionAttributeValues=values)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                raise RuntimeError("The reviewed draft changed; re-read it before promotion") from exc
            raise
        return {"work_id": work_id, "fields": sorted(fields)}


def dispatch(event, *, s3, table, bucket, index):
    """Execute a wiki operation in AWS; no model calls or full-body response option."""
    action = event.get("action")
    if action in {"pipeline_failures", "corpus_status"}:
        from .corpus_ops import report
        result = report(event, table=table, s3=s3, bucket=bucket, index=index)
    elif action == "wiki_read":
        result = _wiki_read(event, s3, table, bucket, index)
    elif action == "read_text":
        _window(event)
        text, etag, digest = _read(s3, bucket, event.get("key"))
        result = {"bucket": bucket, "key": event["key"], "etag": etag, "sha256": digest, **_excerpt(text, event)}
    elif action == "wiki_categories":
        result = _categories(index)
    elif action == "synthesis_coverage":
        result = _synthesis_coverage(index)
    elif action == "notes_in_category":
        result = _notes_in_category(event, index)
    elif action == "wiki_validate":
        result = _validate(event, s3, bucket)
    elif action == "wiki_metrics":
        result = _metrics(event, s3, bucket)
    elif action == "promote_draft":
        work_id = event.get("work_id")
        if not isinstance(work_id, str) or not re.fullmatch(r"W[0-9]+", work_id):
            raise ValueError("Promotion requires an OpenAlex work ID")
        result = promote_draft_in_worker(
            SimpleNamespace(aws_bucket=bucket, aws_table="configured"), work_id,
            reviewer=str(event.get("reviewer") or ""), method=event.get("method", "agent-session"),
            note=event.get("note"), edited=event.get("edited", False),
            reviewed_text=event.get("reviewed_text"), dry_run=event.get("dry_run", False),
            store=_ReviewStore(s3, table, bucket))
    else:
        raise ValueError("Unknown wiki operation")
    result["execution"] = "aws"
    if len(json.dumps(result).encode()) > MAX_RESPONSE_BYTES:
        raise ValueError("Response exceeds the 128 KiB budget; narrow the request")
    return result
