"""Prompts, validators and page assembly for the synthesis layer.

The model writes only the sections that need reading; code writes the lists (Related concepts,
Notes, Subtopics, Key concepts, Coverage) and the frontmatter, as the note's Document Information
table already is. Every validator returns a list of problems and never raises on content.
"""
from __future__ import annotations

import json
import re

CONCEPT_SECTIONS = ("## Definition", "## What the notes show", "## Disagreements and limits")
SUBTOPIC_SECTIONS = ("## Scope", "## Findings", "## Comparison", "## Open questions")
CATEGORY_SECTIONS = ("## Landscape", "## Open questions")
LINKED_SECTIONS = frozenset({"What the notes show", "Findings"})
CATEGORY_MIN_CHARS = 3000
MIN_CHARS_BY_SECTIONS = {CATEGORY_SECTIONS: CATEGORY_MIN_CHARS}
# "As an AI, I ..." is the self-reference; see ingest_lambda.DRAFT_FORBIDDEN for why the bare words are not.
FORBIDDEN = re.compile(r"\[(?:TODO|TBD|placeholder|citation needed)\b[^\]]*\]|\[insert [^\]]*here\]|lorem ipsum|"
                       r"\bas an ai(?: language model| assistant)?,? i\b|"
                       r"^\s*(?:wait|okay|ok)\b\s*[,.:]|^\s*(?:let me|i need to|i'll)\b", re.I | re.M)
ALL_LINKS = re.compile(r"\[\[([^\]]+)\]\]")
INNER_LINK = re.compile(r"^([a-z]+)/([^|#]+?)(?:[|#].*)?$")
BULLET = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
HEADING_LINE = re.compile(r"^\s{0,3}#{1,6}\s")
BULLET_LINK_END = re.compile(r"(?:\[\[sources/[a-z0-9][a-z0-9-]*(?:[|#][^\]]*)?\]\][\s.,;:)\]]*)+$")
MODEL_ID_PATTERN = re.compile(r"claude-([a-z]+)-(\d+)(?:-(\d+))?")

COMMON_RULES = (
    "- Use only what the evidence notes state: no outside knowledge, no background the notes do not give, "
    "no paper the notes do not contain.\n"
    "- Keep every number exactly as the note writes it, with its unit and its cohort.\n"
    "- Attribute: every bullet under {linked} ends with the link of the note it comes from, written exactly "
    "as [[sources/STEM]] with the stem from that note's header; several notes, several links.\n"
    "- A bullet may wrap onto further lines, but nothing may follow its link(s) except a full stop; the alias "
    "form [[sources/STEM|Author Year]] is allowed.\n"
    "- Where notes disagree, say so and attribute each position.\n"
    "- No frontmatter, no level-1 title, no preamble, no placeholders, no notes to the reader, no remarks "
    "about your own process. Start directly with '{first}'.\n"
)
# The four rules this knowledge base runs on, as its owner wrote them. Every page written here is
# bound by them: where the material may come from, what to do when the wiki falls short, and what
# to say when nothing here can answer.
CORE_RULES = (
    "This wiki runs on four rules.\n"
    "1. No web search. Never reach outside this wiki. Every claim is grounded in papers actually "
    "present here.\n"
    "2. The evidence notes and the wiki pages are the only sources of truth.\n"
    "3. When they are not enough, the paper's stored full text is the next source — extracted at "
    "ingest and kept beside the PDF — and what is taken from it belongs on the page, so the next "
    "question does not have to read that text again.\n"
    "4. When no paper here can support a claim, say so plainly and name the paper that is missing. "
    "Do not improvise, and do not write a gap as if it were a finding.\n\n"
)

CONCEPT_SYSTEM = (
    CORE_RULES +
    "You write the concept page of a scientific literature wiki. A concept is one named thing: a gene, a "
    "method, a cohort, a phenomenon. You receive its name, its aliases, a catalogue of every evidence note "
    "whose glossary names it (one line each: link, citation, one-line summary), and the full text of the "
    "pages its name retrieved, each headed '=== Retrieved ... ==='. The catalogue tells you where the "
    "concept appears; the retrieved pages are what you read closely. Some are overviews or concepts written "
    "earlier for this wiki: build on them rather than restating them. Write Markdown in English with "
    "exactly these level-2 sections in this order: Definition, What the notes show, Disagreements and limits.\n\n"
    "Rules:\n"
    "- Definition: two to four sentences saying what the thing is, built only from how the notes define it.\n"
    "- What the notes show: one finding per bullet, each with its numbers, each ending with its "
    "[[sources/STEM]] link(s). When there are more than eight bullets, group them under level-3 headings by aspect.\n"
    "- Disagreements and limits: only conflicts between notes, or limits a note states about this concept. "
    "If there are none, write the single line 'None recorded.'\n"
    + COMMON_RULES.format(linked="What the notes show", first="## Definition")
)
CONCEPT_MERGE_SYSTEM = (
    "You merge partial concept pages into one. Each partial was written from a different batch of evidence "
    "notes about the same concept and has the sections Definition, What the notes show, Disagreements and "
    "limits. Write one page with exactly those three level-2 sections in that order. Keep every finding and "
    "keep each [[sources/STEM]] link attached to the claim it supports; merge duplicates into one bullet "
    "carrying all their links; when partials conflict, keep both positions under Disagreements and limits "
    "with their links. Add nothing the partials do not contain. No frontmatter, no title, no preamble. "
    "Start directly with '## Definition'.\n"
    + COMMON_RULES.format(linked="What the notes show", first="## Definition")
)
SUBTOPIC_SYSTEM = (
    CORE_RULES +
    "You write a subtopic page of a scientific literature wiki: the dense record of what a set of papers "
    "inside one category found. You receive the category, the subtopic's title and scope, a catalogue of "
    "every evidence note assigned to the subtopic (one line each: link, citation, one-line summary), and the "
    "full text of the pages that the subtopic's own words retrieved, each headed '=== Retrieved ... ==='. The "
    "catalogue tells you what the subtopic contains; the retrieved pages are what you read closely. Some "
    "retrieved pages are overviews or concepts written earlier for this wiki: build on them and say where "
    "they already settle something, rather than restating them. Write Markdown in English with exactly "
    "these level-2 sections in this order: Scope, Findings, Comparison, Open questions.\n\n"
    "Rules:\n"
    "- Scope: one paragraph: what the subtopic covers and which papers it rests on, cited by first author and year.\n"
    "- Findings: one finding per bullet with its numbers, ending with its [[sources/STEM]] link(s); level-3 "
    "headings by aspect when there are more than eight bullets.\n"
    "- Comparison: when three or more notes are comparable, a Markdown table with the columns Study, Cohort, "
    "Design, Method, Main result, Note (the last column is the [[sources/STEM]] link). Otherwise the single "
    "line 'Not applicable.'\n"
    "- Open questions: only questions that follow from gaps or disagreements the notes themselves show, each "
    "naming the notes it arises from.\n"
    + COMMON_RULES.format(linked="Findings", first="## Scope")
)
# A synthesis page here grows the way the local wiki's did: by being revised when a question
# touches it, not by being planned and written once. The agent that just answered something
# decides whether the page should carry it, and this prompt does the revision.
UPDATE_SYSTEM = (
    CORE_RULES +
    "You revise one page of a scientific literature wiki. You receive the page as it stands, what has just "
    "been established, any corrections to it, and the evidence notes behind them. Return the whole revised "
    "page in Markdown, with exactly the same level-2 sections it already has, in the same order.\n\n"
    "This is a revision, not a rewrite.\n"
    "- Keep every sentence the update does not touch, word for word. A section the update says nothing about "
    "comes back unchanged.\n"
    "- Put what was established where it belongs, next to the claims it bears on, and attach its "
    "[[sources/STEM]] link. If it refines an existing claim, fold it into that claim rather than adding a "
    "second bullet that says nearly the same thing.\n"
    "- When a correction says a statement is wrong, replace that statement. Do not leave the old claim "
    "standing beside the new one, and do not write that the page previously said something else: the page "
    "records what the evidence shows, and its earlier versions are kept elsewhere.\n"
    "- Add nothing the supplied notes do not support. Where the update leaves a question open, it belongs "
    "under the page's own open-questions section if it has one.\n"
    "- Do not restructure, rename or reorder sections, and do not pad. A page should grow only by what the "
    "update actually adds.\n"
    "No frontmatter, no level-1 title, no preamble, no commentary about the revision itself.\n"
)
SUBTOPIC_MERGE_SYSTEM = (
    "You merge partial subtopic pages into one. Each partial was written from a different batch of evidence "
    "notes of the same subtopic and has the sections Scope, Findings, Comparison, Open questions. Write one "
    "page with exactly those four level-2 sections in that order. Keep every finding and keep each "
    "[[sources/STEM]] link attached to the claim it supports; merge duplicates into one bullet carrying all "
    "their links; join the comparison tables into one when they share columns, otherwise keep them as "
    "consecutive tables; keep open questions that any partial raises. Add nothing the partials do not "
    "contain. No frontmatter, no title, no preamble. Start directly with '## Scope'.\n"
    + COMMON_RULES.format(linked="Findings", first="## Scope")
)
CATEGORY_RULES = (
    "- Use only what the subtopic pages state: no outside knowledge, no background they do not give, no claim "
    "they do not support.\n"
    "- Attribute: every bullet under Open questions ends with the [[overviews/CATEGORY/SLUG]] link(s) of the "
    "subtopic page(s) it arises from, written exactly as in that page's header; several subtopics, several links.\n"
    "- A bullet may wrap onto further lines, but nothing may follow its link(s) except a full stop.\n"
    "- No frontmatter, no level-1 title, no preamble, no placeholders, no notes to the reader, no remarks about "
    "your own process. Start directly with '## Landscape'.\n"
)
CATEGORY_SYSTEM = (
    "You write the landscape page of one category of a scientific literature wiki, for a human reader. You "
    "receive the category's subtopic pages (Scope, Findings, Comparison, Open questions), each headed by its "
    "page link. Write Markdown in English with exactly these two level-2 sections in this order: Landscape, "
    "Open questions.\n\n"
    "Rules:\n"
    "- Landscape: 600 to 1,200 words of connected prose, not bullets: what the field has established, where "
    "it divides, and how the subtopics relate. Every claim carries a link: the subtopic page it comes from, "
    "written exactly as in that page's header ([[overviews/CATEGORY/SLUG]]), or a note link copied from that "
    "page ([[sources/STEM]]).\n"
    "- Open questions: only questions that recur across, or conflict between, the subtopics' own Open "
    "questions, each with the subtopic links it comes from.\n"
    + CATEGORY_RULES
)
PARTITION_SYSTEM = (
    "Partition the papers of one category into subtopics. Each input line is a short paper ID, title and summary. "
    "Return only one JSON object, without prose or fences: "
    '{{"subtopics":[{{"slug":"kebab-case","title":"Noun phrase","scope":"One sentence","ids":["p0001"]}}]}}. '
    "Use {min_subtopics} to {max_subtopics} subtopics, each containing at least {min_notes} papers. "
    "Every given ID must occur exactly once; never invent IDs or output filenames. "
    "Name subtopics after mechanisms, variant classes, methods or cohort designs, not paper types. "
    "Papers that fit none go to '{other}', exempt from the minimum paper count, with a scope explaining why."
)
PARTITION_MERGE_SYSTEM = (
    "Merge the supplied candidate groups from multiple partitions. Each group has a unique ID, title, scope "
    "and paper count. Return only one JSON object: "
    '{{"merge":{{"g0001":"new-slug"}},"subtopics":[{{"slug":"new-slug","title":"Noun phrase","scope":"One sentence"}}]}}. '
    "Map every supplied group ID exactly once to a declared target slug. Use {min_subtopics} to "
    "{max_subtopics} non-other subtopics. Merge semantically equivalent groups and retain distinct ones. "
    "The input includes the permitted other slug, which is exempt from the minimum paper count."
)
ASSIGN_SYSTEM = (
    "Assign every new paper ID to one of the supplied subtopic slugs. Return only one JSON object: "
    '{{"assignments":{{"p0001":"slug"}}}}. Include every supplied ID exactly once, no other IDs. '
    "Use only the given slugs or '{other}' for a paper that fits none. Do not output filenames."
)
CANDIDATE_SYSTEM = (
    "You tidy a list of candidate concepts extracted from the glossaries of a scientific literature wiki. Each "
    "candidate has a slug, a title, aliases, a kind (gene or term) and up to two sample glossary definitions. "
    "Return only JSON, no prose and no code fence: a list with one object per candidate, in the same order, "
    '{"slug": ..., "merge_into": null or the slug of another candidate in this list that names the same thing, '
    '"entity_type": one of "gene", "method", "cohort", "phenomenon", "other"}. Merge only true synonyms '
    "(PRS and polygenic risk score), never a part into a whole or a specific term into a general one."
)


def sections(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    return [(name.strip(), content.strip()) for name, content in zip(parts[1::2], parts[2::2])]


def validate_structure(text: str, expected: tuple[str, ...], *, min_chars: int | None = None) -> list[str]:
    if min_chars is None:
        min_chars = MIN_CHARS_BY_SECTIONS.get(expected, 400)
    problems = []
    names = [name for name, _ in sections(text)]
    want = [s[3:] for s in expected]
    if names != want:
        problems.append(f"sections are {names}, expected {want}")
    if not text.lstrip().startswith(expected[0]):
        problems.append(f"page must start with {expected[0]}")
    if "---" in text.split(expected[0])[0]:
        problems.append("frontmatter must not be model-written")
    if FORBIDDEN.search(text):
        problems.append("placeholder, filler or scratchpad text present")
    if len(text) < min_chars:
        problems.append("page is too short")
    return problems


def _bullet_groups(content: str) -> list[dict]:
    """Each top-level bullet with its wrapped continuation lines and any nested sub-bullets, and each
    level-3+ heading the model uses to group more than eight bullets by aspect, as its own group
    (``heading: True``) that a bullet check skips and that always starts the next bullet fresh."""
    groups: list[dict] = []
    for line in content.splitlines():
        if not line.strip():
            continue
        if HEADING_LINE.match(line):
            groups.append({"indent": 0, "lines": [line], "heading": True})
            continue
        indent = len(line) - len(line.lstrip())
        fresh = not groups or groups[-1].get("heading") or indent <= groups[-1]["indent"]
        if BULLET.match(line) and fresh:
            groups.append({"indent": indent, "lines": [line]})
        elif groups and not groups[-1].get("heading"):
            groups[-1]["lines"].append(line)
        else:
            groups.append({"indent": indent, "lines": [line]})
    return groups


def _bullet_group_ok(group: list[str]) -> bool:
    """A group passes when its last line carries the link, or - for a ``...:`` header followed by nested
    bullets - when every nested bullet carries one (the header itself is not a finding)."""
    head = group[0].strip()
    children = [line for line in group[1:] if BULLET.match(line)]
    if head.endswith(":") and children:
        return all(BULLET_LINK_END.search(child.strip()) for child in children)
    return bool(BULLET_LINK_END.search(group[-1].strip()))


def validate_links(text: str, *, allowed_stems, allowed_pages=(), linked_sections=LINKED_SECTIONS) -> list[str]:
    """Bullets in the attributed sections end with note links; every link points inside the page's members."""
    problems = []
    stems, pages = set(allowed_stems), set(allowed_pages)
    for name, content in sections(text):
        if name in linked_sections:
            for group in _bullet_groups(content):
                if group.get("heading"):
                    continue
                if not _bullet_group_ok(group["lines"]):
                    problems.append(f"{name}: bullet without a [[sources/...]] link: {group['lines'][0].strip()[:80]}")
        if name == "Comparison" and content.strip() != "Not applicable.":
            rows = [line for line in content.splitlines() if line.strip().startswith("|")]
            if not any("[[sources/" in row for row in rows):
                problems.append("Comparison: table rows must end with the note link, or the section must read "
                               "'Not applicable.'")
    for inner in ALL_LINKS.findall(text):
        m = INNER_LINK.match(inner)
        if not m:
            problems.append(f"malformed link: [[{inner}]]")
            continue
        kind, ident = m.group(1), m.group(2).strip()
        if kind == "sources":
            if ident not in stems:
                problems.append(f"link to a note outside the members: {ident}")
        elif kind == "overviews":
            if ident not in pages:
                problems.append(f"link to a page outside this category: {ident}")
        else:
            problems.append(f"link of a kind the model may not write: {kind}/{ident}")
    return problems


def parse_json(text: str) -> tuple[object | None, str | None]:
    """Strict parse first; then a fenced ```json block; then the first JSON value found in the text."""
    try:
        return json.loads(text.strip()), None
    except json.JSONDecodeError as exc:
        error = exc
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1)), None
        except json.JSONDecodeError as exc:
            error = exc
    start = re.search(r"[{\[]", text)
    if start:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text, start.start())
            return obj, None
        except json.JSONDecodeError as exc:
            error = exc
    return None, f"model did not return JSON: {error}"


def parse_plan_json(text: str) -> tuple[object | None, str | None]:
    """Planning needs one complete value and unambiguous keys; never salvage a prefix."""
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj
    try:
        return json.loads(text.strip(), object_pairs_hook=unique), None
    except (ValueError, json.JSONDecodeError) as exc:
        return None, f"model did not return one complete JSON value: {exc}"


def frontmatter(fields: list[tuple[str, object]]) -> str:
    return "---\n" + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in fields) + "\n---\n\n"


def provenance(model_id: str, reasoning: str) -> list[tuple[str, object]]:
    m = MODEL_ID_PATTERN.search(model_id)
    family, version = (m.group(1), m.group(2) + ("." + m.group(3) if m.group(3) else "")) if m else (model_id, "")
    return [("ingest_harness", "aws-bedrock"), ("ingest_agent", "byeori-synthesis"), ("ingest_agent_version", "v1"),
            ("ingest_model", family), ("ingest_model_version", version), ("ingest_reasoning", reasoning),
            ("ingest_model_id", model_id)]


def _require_sha256(m: dict, ident_key: str = "stem") -> str:
    sha = m.get("sha256")
    if not sha:
        raise ValueError(f"member {m.get(ident_key)} has no sha256")
    return sha


def _note_line(m: dict) -> str:
    return f"- [[sources/{m['stem']}]] {m.get('first_author') or 'Unknown'} ({m.get('year') or ''}). {m.get('title') or m['stem']}"


def notes_list(members: list[dict]) -> str:
    return "\n".join(_note_line(m) for m in members) or "- None."


def concept_page(model_text: str, *, concept: dict, members: list[dict], mentions: list[dict],
                 related: list[tuple[str, int]], model_id: str, reasoning: str, generation: str,
                 manifest_ref: str, created: str | None, today: str) -> str:
    fields: list[tuple[str, object]] = [
        ("title", concept["title"]), ("kind", "concept"), ("slug", concept["slug"]),
        ("aliases", list(concept.get("aliases") or [])), ("entity_type", concept.get("entity_type") or "other"),
        ("categories", sorted({m.get("category") for m in members if m.get("category")})), ("note_count", len(members)),
        ("source_notes", [{"stem": m["stem"], "sha256": _require_sha256(m)} for m in members]),
        ("generation", generation), ("manifest", manifest_ref), *provenance(model_id, reasoning),
        ("created", created or today), ("updated", today)]
    body = model_text.strip()
    body += "\n\n## Related concepts\n" + ("\n".join(f"- [[concepts/{s}]] ({n} shared notes)" for s, n in related) or "- None.")
    body += "\n\n## Notes\n" + notes_list(members)
    if mentions:
        body += "\n\nAlso mentioned in the body of:\n" + "\n".join(_note_line(m) for m in mentions)
    return frontmatter(fields) + body + "\n"


def subtopic_page(model_text: str, *, category: str, subtopic: dict, members: list[dict],
                  concepts: list[tuple[str, int]], model_id: str, reasoning: str, generation: str,
                  manifest_ref: str, created: str | None, today: str) -> str:
    fields: list[tuple[str, object]] = [
        ("title", subtopic["title"]), ("kind", "subtopic"), ("category", category), ("slug", subtopic["slug"]),
        ("note_count", len(members)), ("source_notes", [{"stem": m["stem"], "sha256": _require_sha256(m)} for m in members]),
        ("generation", generation), ("manifest", manifest_ref), *provenance(model_id, reasoning),
        ("created", created or today), ("updated", today)]
    body = model_text.strip()
    body += "\n\n## Concepts\n" + ("\n".join(f"- [[concepts/{s}]] ({n} notes)" for s, n in concepts) or "- None.")
    body += "\n\n## Notes\n" + notes_list(members)
    return frontmatter(fields) + body + "\n"


_ABBREVIATIONS = ("e.g.", "i.e.", "al.", "vs.", "cf.", "Fig.")


def _first_sentence(text: str) -> str:
    """The first sentence of ``text``, without splitting at an abbreviation like 'e.g.' or 'et al.'."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    sentence = parts[0] if parts else ""
    i = 1
    while i < len(parts) and any(sentence.endswith(a) for a in _ABBREVIATIONS):
        sentence += " " + parts[i]
        i += 1
    return sentence


def category_page(model_text: str, *, category: str, subtopics: list[dict], key_concepts: list[tuple[str, int]],
                  coverage: dict, model_id: str, reasoning: str, manifest_ref: str, created: str | None, today: str,
                  generation: str = "single") -> str:
    """subtopics: {slug, title, scope, note_count, sha256}; coverage: {note_count, year_min, year_max, generated_at}."""
    parts = dict(sections(model_text))
    if "Landscape" not in parts:
        raise ValueError("model text lacks ## Landscape")
    fields: list[tuple[str, object]] = [
        ("title", f"{category}: landscape"), ("kind", "category"), ("category", category),
        ("note_count", coverage["note_count"]), ("subtopic_count", len(subtopics)),
        ("source_pages", [{"slug": s["slug"], "sha256": _require_sha256(s, "slug")} for s in subtopics]),
        ("generation", generation), ("manifest", manifest_ref), *provenance(model_id, reasoning),
        ("created", created or today), ("updated", today)]
    body = "## Landscape\n" + parts.get("Landscape", "").strip()
    body += "\n\n## Subtopics\n" + ("\n".join(
        f"- [[overviews/{category}/{s['slug']}]] {s.get('title') or s['slug']} ({s.get('note_count', 0)} notes): "
        f"{_first_sentence(s.get('scope') or '')}" for s in subtopics) or "- None.")
    body += "\n\n## Key concepts\n" + ("\n".join(f"- [[concepts/{s}]] ({n} notes)" for s, n in key_concepts) or "- None.")
    body += "\n\n## Open questions\n" + parts.get("Open questions", "").strip()
    body += (f"\n\n## Coverage\nNotes: {coverage['note_count']}. Years: {coverage['year_min']}-{coverage['year_max']}. "
             f"Subtopics: {len(subtopics)}. Generated: {coverage['generated_at']}.")
    return frontmatter(fields) + body + "\n"
