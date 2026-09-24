"""Scopes, note metadata, and the rules that decide what a synthesis run writes again.

Stdlib only; shared by the synthesis Lambda, the local CLI and the tests.
"""
from __future__ import annotations

import hashlib
import json
import re

SCOPES: dict[str, tuple[str, ...]] = {
    "autism": ("asd-ndd", "asd-models", "psychiatric-genetics", "gwas", "germline-mutation"),
}
KEEP_SECTIONS = ("## One-line Summary", "## 2. Key Contributions", "## 3. Methodology and Architecture",
                 "## 4. Key Results and Benchmarks", "## 5. Limitations and Future Work")
STALE_MIN_NOTES = 5
STALE_FRACTION = 0.2
SLUG = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
HEADING = re.compile(r"^(## .+)$", re.M)
RESERVED_SLUGS = {"index", "failed"}


def scope_categories(scope: str) -> tuple[str, ...] | None:
    """The categories a scope names; None means every category."""
    if scope == "all":
        return None
    if scope in SCOPES:
        return SCOPES[scope]
    raise ValueError(f"scope must be one of all, {', '.join(SCOPES)}")


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """The flat ``key: "json string"`` frontmatter the Lambdas write, and the body after it."""
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}, text
    fields: dict = {}
    for line in text[4:end].splitlines():
        key, sep, raw = line.partition(":")
        if not sep or line.startswith(" "):
            continue
        raw = raw.strip()
        try:
            fields[key.strip()] = json.loads(raw) if raw else ""
        except json.JSONDecodeError:
            fields[key.strip()] = raw.strip('"')
    return fields, text[end + 5:].lstrip("\n")


AUTHOR_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
AUTHOR_PARTICLES = {"van", "von", "de", "del", "der", "da", "di", "le", "la"}


def first_author(authors: str) -> str:
    head = re.split(r"\s*(?:;|,|\band\b|&|\bet\s+al\.?)\s*", authors or "")[0].strip()
    words = head.replace(".", "").split()
    if words and words[-1].isdigit():
        words = words[:-1]  # drop an affiliation digit, as in 'A Abromeit 1'
    if words and words[-1].lower() in AUTHOR_SUFFIXES:
        words = words[:-1]  # drop a generational suffix, as in 'Smith Jr'
    if not words:
        return "Unknown"
    if len(words) > 1 and re.fullmatch(r"[A-Z]{1,3}", words[-1]):
        return " ".join(words[:-1])  # 'Kong A' or 'van der Meer J': surname first, initials last
    if len(words) > 1 and words[-2].lower() in AUTHOR_PARTICLES:
        return " ".join(words[-2:])  # 'Silvia De Rubeis': particle joins the surname it precedes
    return words[-1]


def sections(body: str) -> list[tuple[str, str]]:
    parts = HEADING.split(body)
    return [(heading.strip(), content.strip()) for heading, content in zip(parts[1::2], parts[2::2])]


def note_metadata(stem: str, text: str) -> dict:
    fields, body = parse_frontmatter(text)
    summary = next((content for heading, content in sections(body) if heading == "## One-line Summary"), "")
    return {"stem": stem, "title": str(fields.get("title") or stem), "first_author": first_author(str(fields.get("authors") or "")),
            "year": str(fields.get("year") or ""), "category": str(fields.get("category") or ""),
            "summary": summary.splitlines()[0].strip() if summary else ""}


def synthesis_input(meta: dict, text: str) -> str:
    """One note as the model reads it: a citation header, the One-line Summary, and sections 2 to 5 (no table,
    related work or glossary)."""
    _fields, body = parse_frontmatter(text)
    kept = [f"{heading}\n{content}" for heading, content in sections(body) if heading.startswith(KEEP_SECTIONS)]
    return (f"=== Note {meta['stem']} | {meta['first_author']} ({meta['year']}). {meta['title']} ===\n"
            + "\n\n".join(kept) + "\n")


def catalog_line(meta: dict, text: str, limit: int = 420) -> str:
    """One member as a catalogue row: title, author, year, and its One-line Summary, clipped.

    This is what llm-wiki's per-category index carries, about 460 bytes a paper, and it is why a
    synthesis page can see every member it was assigned without reading any of them in full.
    """
    _fields, body = parse_frontmatter(text)
    summary = ""
    for heading, content in sections(body):
        if "one-line summary" in heading.lower():
            summary = " ".join(content.split())
            break
    if not summary:
        for heading, content in sections(body):
            if content.strip():
                summary = " ".join(content.split())
                break
    if len(summary) > limit:
        summary = summary[:limit].rsplit(" ", 1)[0] + "..."
    return f"- [[sources/{meta['stem']}]] - {meta['first_author']} ({meta['year']}). {meta['title']} - {summary}"


def member_digest(members: dict[str, str]) -> str:
    return hashlib.sha256("\n".join(f"{s}:{h}" for s, h in sorted(members.items())).encode("utf-8")).hexdigest()


def stale_mode(previous: dict[str, str] | None, current: dict[str, str]) -> str:
    """generate: no page yet, or membership moved by 5 notes or 20%; refresh: a smaller change, code sections
    only; skip: nothing changed. ``previous`` and ``current`` map stem -> note sha256."""
    if previous is None:
        return "generate"
    changed = set(previous) ^ set(current)
    changed |= {s for s in set(previous) & set(current) if previous[s] != current[s]}
    if not changed:
        return "skip"
    if len(changed) >= STALE_MIN_NOTES or len(changed) / max(len(current), 1) >= STALE_FRACTION:
        return "generate"
    return "refresh"


def validate_partition(partition, stems: list[str], *, category: str, min_subtopics: int = 4,
                       max_subtopics: int = 12, min_notes: int = 5) -> list[str]:
    """Every defect in a proposed subtopic partition, in the words the model is asked to fix."""
    subtopics = partition.get("subtopics") if isinstance(partition, dict) else None
    if not isinstance(subtopics, list):
        return ["partition must be an object with a subtopics list"]
    other = f"{category}-other"
    known = set(stems)
    assigned: dict[str, str] = {}
    slugs: set[str] = set()
    problems: list[str] = []
    for st in subtopics:
        if not isinstance(st, dict):
            problems.append("subtopic entries must be objects")
            continue
        slug = str(st.get("slug", ""))
        if not SLUG.fullmatch(slug) or not any(ch.isalpha() for ch in slug):
            problems.append(f"bad slug {slug!r}: use lowercase words joined by hyphens")
        if slug in RESERVED_SLUGS:
            problems.append(f"{slug} is a reserved slug: name the subtopic after what its papers study")
        if slug in slugs:
            problems.append(f"duplicate subtopic {slug}")
        slugs.add(slug)
        if not str(st.get("title", "")).strip():
            problems.append(f"{slug}: missing title")
        if not str(st.get("scope", "")).strip():
            problems.append(f"{slug}: missing scope sentence")
        raw_stems = st.get("stems")
        if raw_stems is None:
            members: list[str] = []
        elif isinstance(raw_stems, list):
            members = [str(s) for s in raw_stems]
        else:
            problems.append(f"{slug}: stems must be a list of stems")
            members = []
        if slug != other and len(members) < min_notes:
            problems.append(f"{slug}: {len(members)} notes, fewer than {min_notes}")
        for s in members:
            if s not in known:
                problems.append(f"{slug}: unknown stem {s}")
            elif s in assigned:
                problems.append(f"{s} assigned to both {assigned[s]} and {slug}")
            else:
                assigned[s] = slug
    missing = sorted(known - set(assigned))
    if missing:
        problems.append(f"{len(missing)} notes unassigned: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")
    real = [s for s in slugs if s != other]
    if not min_subtopics <= len(real) <= max_subtopics:
        problems.append(f"{len(real)} subtopics, need {min_subtopics} to {max_subtopics}")
    return problems


def chunked(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]
