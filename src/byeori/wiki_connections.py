"""Publish wiki Markdown and maintain its reciprocal links and list catalogs in S3.

Scientific text is written once against the version the caller read. Only the small
managed link/catalog blocks are merged again after a concurrent write.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from botocore.exceptions import ClientError


FOLDERS = ("sources", "overviews", "concepts", "questions")
MISSING = {"404", "NoSuchKey", "NotFound"}
CONFLICT = {"412", "PreconditionFailed", "409", "ConditionalRequestConflict"}
MAX_ATTEMPTS = 4
BACKLINK_START = "<!-- byeori:backlinks:start -->"
BACKLINK_END = "<!-- byeori:backlinks:end -->"
CATALOG_START = "<!-- byeori:catalog:start -->"
CATALOG_END = "<!-- byeori:catalog:end -->"
BACKLINK_BLOCK = re.compile(re.escape(BACKLINK_START) + r".*?" + re.escape(BACKLINK_END), re.S)
CATALOG_BLOCK = re.compile(re.escape(CATALOG_START) + r".*?" + re.escape(CATALOG_END), re.S)
WIKILINK = re.compile(r"\[\[([^\]\n]+)\]\]")


class PageConflictError(RuntimeError):
    """The caller must read the current scientific text before revising it."""


@dataclass(frozen=True)
class _Page:
    text: str
    etag: str


def _page_key(key):
    if not isinstance(key, str) or not key.startswith("wiki/") or not key.endswith(".md"):
        raise ValueError("Expected a wiki Markdown key")
    parts = key.split("/")
    if (len(parts) < 3 or parts[1] not in FOLDERS
            or any(p in {"", ".", "..", "failed", "drafts"} for p in parts)
            or not all(re.fullmatch(r"[A-Za-z0-9._-]+", p) for p in parts)):
        raise ValueError("Expected a published source, overview, concept, or question key")
    return key


def _link(key):
    return key.removeprefix("wiki/").removesuffix(".md")


def _linked_key(value):
    target = value.split("|", 1)[0].split("#", 1)[0].strip().removeprefix("wiki/")
    if not target.endswith(".md"):
        target += ".md"
    try:
        return _page_key("wiki/" + target)
    except ValueError:
        return None


def _read(s3, bucket, key):
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in MISSING:
            return None
        raise
    body = response["Body"]
    try:
        text = body.read().decode("utf-8")
    finally:
        body.close()
    etag = response.get("ETag")
    if not etag:
        raise RuntimeError(f"S3 returned no ETag for {key}")
    return _Page(text, etag)


def _put(s3, bucket, key, text, previous):
    condition = {"IfMatch": previous.etag} if previous else {"IfNoneMatch": "*"}
    return s3.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"),
                         ContentType="text/markdown; charset=utf-8", **condition)


def _frontmatter(text):
    match = re.match(r"\A---\r?\n.*?\r?\n---(?:\r?\n|\Z)", text, re.S)
    return (match.group(), text[match.end():]) if match else ("", text)


def _title(text, key):
    front, body = _frontmatter(text)
    match = re.search(r"^title:\s*(.+)$", front, re.M)
    if match:
        value = match.group(1).strip()
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            value = value.strip("'")
        return str(value)
    match = re.search(r"^# (.+)$", body, re.M)
    return match.group(1).strip() if match else _link(key)


def _label(value):
    return (re.sub(r"\s+", " ", str(value)).replace("|", " ").replace("[", "(")
            .replace("]", ")").replace("\u00b7", ",").strip())


def _entries(text, pattern, *, catalogs=False):
    entries = {}
    for block in pattern.findall(text):
        for value in WIKILINK.findall(block):
            target, _, label = value.partition("|")
            key = _linked_key(target)
            if catalogs and target.startswith("indexes/"):
                key = "wiki/" + target.removesuffix(".md") + ".md"
            if key:
                entries[key] = _label(label or _link(key))
    return entries


def _outgoing(text):
    # A backlink added by this module is not a new scientific citation to process.
    body = _frontmatter(BACKLINK_BLOCK.sub("", text))[1]
    return {key for value in WIKILINK.findall(body) if (key := _linked_key(value))}


def _with_backlinks(text, key, entries):
    cleaned = BACKLINK_BLOCK.sub("", text)
    if not entries:
        return cleaned
    block = (BACKLINK_START + "\n### Linked pages\n"
             + "\n".join(f"- [[{_link(k)}|{_label(v)}]]" for k, v in sorted(entries.items()))
             + "\n" + BACKLINK_END)
    if key.startswith("wiki/sources/"):
        section = re.search(r"^## (?:\d+[.)]\s*)?Related (?:Work|Papers|Pages)[ \t]*$",
                            cleaned, re.M | re.I)
        if section:
            following = re.search(r"^## ", cleaned[section.end():], re.M)
            offset = section.end() + following.start() if following else len(cleaned)
            return cleaned[:offset].rstrip() + "\n\n" + block + "\n\n" + cleaned[offset:]
    return cleaned.rstrip() + "\n\n" + block + "\n"


def _scientific_text(incoming, current, key):
    entries = _entries(incoming, BACKLINK_BLOCK)
    if current:
        # The current document owns its metadata and incoming links, even when a
        # model returns only its rewritten scientific body.
        entries.update(_entries(current.text, BACKLINK_BLOCK))
        old_front, _ = _frontmatter(current.text)
        if old_front:
            new_front, new_body = _frontmatter(incoming)
            if key.startswith("wiki/questions/") and new_front:
                # A regenerated answer owns fresh model/run metadata; only its
                # first creation date belongs to the previous answer.
                created = re.search(r"^created:[^\r\n]*", old_front, re.M)
                if created:
                    if re.search(r"^created:", new_front, re.M):
                        new_front = re.sub(r"^created:[^\r\n]*", lambda _: created.group(), new_front, flags=re.M)
                    else:
                        closing = new_front.rfind("---")
                        new_front = new_front[:closing] + created.group() + "\n" + new_front[closing:]
                incoming = new_front + new_body
            else:
                incoming = old_front + new_body
    return _with_backlinks(incoming, key, entries)


def _cas_merge(s3, bucket, key, transform, *, require_existing=False, check_remaining=None):
    for attempt in range(MAX_ATTEMPTS):
        if check_remaining:
            check_remaining()
        current = _read(s3, bucket, key)
        if current is None and require_existing:
            return {"key": key, "status": "missing"}
        text = transform(current.text if current else "")
        if current and text == current.text:
            return {"key": key, "status": "unchanged", "etag": current.etag}
        try:
            if check_remaining:
                check_remaining()
            response = _put(s3, bucket, key, text, current)
            return {"key": key, "status": "updated", "etag": response.get("ETag", "")}
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in CONFLICT:
                raise
            if attempt == MAX_ATTEMPTS - 1:
                raise PageConflictError(f"Concurrent managed edits did not settle for {key}") from exc


def _reciprocal(s3, bucket, target, origin, title, *, remove=False, check_remaining=None):
    def merge(text):
        entries = _entries(text, BACKLINK_BLOCK)
        if remove:
            entries.pop(origin, None)
        elif origin in _outgoing(text):
            # A link in the authored body already connects these pages. Do not add
            # a second copy, and never edit the author's existing link.
            entries.pop(origin, None)
        else:
            entries[origin] = title
        return _with_backlinks(text, target, entries)
    return _cas_merge(s3, bucket, target, merge, require_existing=True, check_remaining=check_remaining)


def _connections(s3, bucket, key, text, previous, *, check_remaining=None):
    """Converge the touched edges on the latest origin version before returning.

    A newer publisher may finish while this one is writing another S3 object.
    The origin postcheck detects that race; only links are reconciled again.
    """
    touched = _outgoing(text) | (_outgoing(previous.text) if previous else set())
    reports, errors = {}, {}
    current = None
    converged = False
    attempts = 0
    try:
        if check_remaining:
            check_remaining()
        current = _read(s3, bucket, key)
        for attempts in range(1, MAX_ATTEMPTS + 1):
            if current is None:
                raise PageConflictError(f"Published origin disappeared during link reconciliation: {key}")
            desired = _outgoing(current.text)
            touched.update(desired)
            title = _title(current.text, key)
            for target in sorted(touched - {key}):
                errors.pop(target, None)
                try:
                    result = _reciprocal(s3, bucket, target, key, title, remove=target not in desired,
                                         check_remaining=check_remaining)
                    result["relationship"] = "removed" if target not in desired else "linked"
                    if result["status"] == "missing" and target in desired:
                        result["relationship"] = "unresolved"
                        errors[target] = {"key": target, "error": "Cited wiki page was not found"}
                    reports[target] = result
                except Exception as exc:
                    errors[target] = {"key": target, "error": str(exc)[:300]}
            if check_remaining:
                check_remaining()
            latest = _read(s3, bucket, key)
            if latest and latest.etag == current.etag:
                converged = True
                break
            current = latest
        if not converged:
            raise PageConflictError(f"Origin kept changing during link reconciliation: {key}")
    except Exception as exc:
        errors[key] = {"key": key, "error": str(exc)[:300]}
    return ({"pages": list(reports.values()), "errors": list(errors.values()),
             "converged": converged, "attempts": attempts,
             "origin_etag": current.etag if current else None}, current)


def _catalog_text(current, key, entries):
    block = (CATALOG_START + "\n"
             + "\n".join(f"- [[{_link(k)}|{_label(v)}]]" for k, v in sorted(entries.items()))
             + "\n" + CATALOG_END)
    if CATALOG_BLOCK.search(current):
        # Replace the first block and remove accidental duplicate managed blocks.
        first = True
        def replace(match):
            nonlocal first
            value = block if first else ""
            first = False
            return value
        return CATALOG_BLOCK.sub(replace, current)
    if not current:
        name = "Wiki" if key == "wiki/index.md" else key.rsplit("/", 1)[1][:-3].capitalize()
        current = f"# {name}\n"
    return current.rstrip() + "\n\n" + block + "\n"


def _catalogs(s3, bucket, additions, *, check_remaining=None):
    results, errors = [], []
    for folder in FOLDERS:
        key = f"wiki/indexes/{folder}.md"
        def merge(text, folder=folder, key=key):
            entries = _entries(text, CATALOG_BLOCK)
            entries.update({k: v for k, v in additions.items() if k.startswith(f"wiki/{folder}/")})
            return _catalog_text(text, key, entries)
        try:
            results.append(_cas_merge(s3, bucket, key, merge, check_remaining=check_remaining))
        except Exception as exc:
            errors.append({"key": key, "error": str(exc)[:300]})
    root_entries = {f"wiki/indexes/{folder}.md": folder.capitalize() for folder in FOLDERS}
    # The per-field catalogs are the browse path a reader takes before searching, so the root has to
    # name them; without this line `aws-build-category-catalogs` writes pages nothing reaches
    # (2026-09-23). The page itself is written there, not here.
    root_entries["wiki/indexes/categories.md"] = "Categories (one catalog per field)"
    try:
        results.append(_cas_merge(s3, bucket, "wiki/index.md",
                                   lambda text: _catalog_text(text, "wiki/index.md", root_entries),
                                   check_remaining=check_remaining))
    except Exception as exc:
        errors.append({"key": "wiki/index.md", "error": str(exc)[:300]})
    return {"pages": results, "errors": errors}


def publish_page(s3, bucket, key, text, *, expected_etag=None, create_only=False, check_remaining=None):
    """Publish one body, then merge reciprocal links and catalog entries in AWS.

    Existing bodies require the ETag from the caller's read. A conflict is never
    retried with a newer scientific body. Secondary failures are returned after
    the primary save, so the caller can report and repair incomplete connections.
    """
    _page_key(key)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Page text must be nonempty Markdown")
    if create_only and expected_etag is not None:
        raise ValueError("Use either create_only or expected_etag")
    current = _read(s3, bucket, key)
    if current and (create_only or expected_etag is None
                    or current.etag.strip('"') != expected_etag.strip('"')):
        raise PageConflictError(f"Read the current version before replacing {key}")
    if not current and expected_etag is not None:
        raise PageConflictError(f"The version read for {key} no longer exists")
    rendered = _scientific_text(text, current, key)
    try:
        response = _put(s3, bucket, key, rendered, current)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in CONFLICT:
            raise PageConflictError(f"Page changed while publishing {key}; read it again") from exc
        raise
    connections, latest = _connections(s3, bucket, key, rendered, current, check_remaining=check_remaining)
    title = _title(latest.text if latest else rendered, key)
    catalogs = _catalogs(s3, bucket, {key: title}, check_remaining=check_remaining)
    return {"key": key, "chars": len(rendered), "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "etag": response.get("ETag", ""), "replaced": current is not None,
            "connections": connections, "catalogs": catalogs,
            "errors": connections["errors"] + catalogs["errors"]}


def rebuild_catalogs(s3, bucket, documents):
    """Register the indexer's full list without erasing concurrent publications.

    Publishing does not delete pages. The supplied list is a snapshot, so existing
    entries absent from it may have been published after that snapshot was taken.
    Catalogs themselves and legacy per-paper folders are not canonical documents.
    """
    entries = {}
    for key, title in documents:
        try:
            entries[_page_key(key)] = str(title)
        except ValueError:
            continue
    return {"documents": len(entries), **_catalogs(s3, bucket, entries)}
