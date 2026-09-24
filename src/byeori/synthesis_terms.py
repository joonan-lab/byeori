"""Concept candidates from the notes' Glossary sections.

Stdlib only: this module runs inside the synthesis Lambda and in the test suite unchanged. A
term becomes a concept when enough notes name it in their Glossary; a gene symbol is folded onto
its HGNC-approved symbol first, so Nav1.2 and SCN2A are one concept rather than two.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

GLOSSARY_HEADING = re.compile(r"^## 7\. Glossary\s*$", re.M)
TERM_LINE = re.compile(r"^\s*[-*]\s+\*\*(.+?)\*\*\s*[:：]\s*(.*)$", re.M)
PAREN_ALIAS = re.compile(r"^(.*?)\s*\(([^()]{1,60})\)\s*$")
LEFT_OF_COMPARISON = re.compile(r"\s+versus\s+|\s+vs\.?\s+")
SLASH_ALIAS = re.compile(r"\s+/\s+")
TOKEN = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*")   # 'SCN2A-related' yields SCN2A; 'Nav1.2' stays whole
ACRONYM_PLURAL = re.compile(r"[A-Z][A-Za-z0-9-]*[A-Z0-9]s")   # 'SNPs', not 'variants' or 'analysis'
GENE_SUFFIX = re.compile(r"\s+(?:gene|protein|channel|receptor|locus)$", re.IGNORECASE)
GREEK = str.maketrans({
    "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "e", "θ": "th", "ι": "i",
    "κ": "k", "λ": "l", "μ": "m", "ν": "n", "ξ": "x", "ο": "o", "π": "p", "ρ": "r", "σ": "s",
    "τ": "t", "υ": "u", "φ": "ph", "χ": "ch", "ψ": "ps", "ω": "w",
    "µ": "m",  # micro sign (U+00B5), distinct from Greek mu above but common in units like 'µM'
    "ς": "s",  # final sigma
    "ϵ": "e",  # lunate epsilon
})
SHORT_UPPER_FORM = re.compile(r"[A-Z0-9]{1,3}")
# Words that name the corpus or a category rather than a thing inside it, plus statistics every
# paper reports. The threshold does most of the filtering; this catches what a count cannot.
# plan_concepts adds every category name (and its hyphen-free form) to this set.
STOP_TERMS = frozenset({
    "autism", "autism spectrum disorder", "asd", "autistic", "neurodevelopmental disorder", "ndd",
    "single-cell", "single cell", "deep learning", "machine learning",
    "p-value", "p value", "odds ratio", "confidence interval", "false discovery rate", "fdr", "effect size",
})
# Upper-case abbreviations this corpus uses that also happen to be HGNC symbols or aliases; a
# glossary line naming one of these is never a gene, whatever the table says.
NOT_GENES = frozenset({
    "CS", "US", "TD", "SD", "GC", "HR", "MS", "CAT", "SET", "MAX", "MIN", "ASD", "ADHD", "ID", "IQ",
    "OR", "CI", "SE", "SNP", "CNV", "SV", "WES", "WGS", "PRS", "GWAS", "MRI", "PCR", "RNA", "DNA",
    "ACC", "PFC",
})


def glossary_section(text: str) -> str:
    match = GLOSSARY_HEADING.search(text)
    if not match:
        return ""
    rest = text[match.end():]
    following = re.search(r"^## ", rest, re.M)
    return rest[: following.start()] if following else rest


def glossary_entries(text: str) -> list[tuple[str, str]]:
    """(raw term, definition) for each '- **Term**: definition' line of the Glossary."""
    return [(m.group(1).strip(), m.group(2).strip()) for m in TERM_LINE.finditer(glossary_section(text))]


def split_aliases(raw: str) -> list[str]:
    """'HbF / F-cells' -> ['HbF', 'F-cells']; 'PRS (polygenic risk score)' -> ['PRS', 'polygenic risk score'].

    A slash only splits when it has whitespace on both sides, so 'mg/kg' and 'MAPK/ERK pathway'
    stay whole; ' versus '/' vs ' keeps only the left-hand side, since the right names what the
    left is being compared against, not another name for it.
    """
    left = LEFT_OF_COMPARISON.split(raw, maxsplit=1)[0]
    parts: list[str] = []
    for piece in SLASH_ALIAS.split(left):
        piece = piece.strip().strip(",;")
        if not piece:
            continue
        m = PAREN_ALIAS.match(piece)
        if m and m.group(1).strip():
            parts.extend([m.group(1).strip(), m.group(2).strip()])
        else:
            parts.append(piece)
    return [p for p in dict.fromkeys(parts) if len(p) >= 2]


def normalise(term: str) -> str:
    original_words = term.split()
    last_original = original_words[-1] if original_words else ""
    is_acronym_plural = ACRONYM_PLURAL.fullmatch(last_original) is not None
    t = term.lower().translate(GREEK)
    t = re.sub(r"[‐‑–—]", "-", t)
    t = re.sub(r"[^a-z0-9+.\- ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip(" .-")
    words = t.split(" ")
    last = words[-1] if words else ""
    if last.endswith("s") and (is_acronym_plural or (len(last) > 4 and not last.endswith(("ss", "us", "is")))):
        words[-1] = last[:-1]
    return " ".join(words)


def slugify(text: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", text.lower().translate(GREEK))).strip("-")[:80]


def _looks_like_a_symbol(term: str) -> bool:
    """A gene symbol is written in digits/caps/hyphens, or contains a digit; an everyday word is not."""
    body = GENE_SUFFIX.sub("", term)
    if len(body) < 2:
        return False
    if any(ch.isdigit() for ch in body):
        return True
    return bool(re.fullmatch(r"[A-Z0-9-]+", body))


class HgncTable:
    """Approved symbols with their aliases and previous symbols, from HGNC's complete set TSV."""

    def __init__(self, lookup: dict[str, str], names: dict[str, str] | None = None):
        self.lookup = lookup  # upper-cased symbol or alias -> approved symbol
        self.names = names or {}
        self.approved = {symbol.upper() for symbol in lookup.values()}

    @classmethod
    def from_tsv(cls, text: str) -> "HgncTable":
        lines = text.splitlines()
        if not lines:
            return cls({})
        header = lines[0].split("\t")
        col = {name: i for i, name in enumerate(header)}
        if "symbol" not in col:
            raise ValueError(f"HGNC TSV has no 'symbol' column; header was {header!r}")
        lookup: dict[str, str] = {}
        names: dict[str, str] = {}
        claims: dict[str, set[str]] = defaultdict(set)
        for line in lines[1:]:
            row = line.split("\t")
            symbol = row[col["symbol"]].strip() if len(row) > col["symbol"] else ""
            if not symbol:
                continue
            lookup[symbol.upper()] = symbol
            if "name" in col and len(row) > col["name"]:
                names[symbol] = row[col["name"]].strip()
            for column in ("alias_symbol", "prev_symbol"):
                if column in col and len(row) > col[column]:
                    for alias in row[col[column]].split("|"):
                        alias = alias.strip().strip('"')
                        if alias:
                            claims[alias.upper()].add(symbol)
        for alias, symbols in claims.items():
            if len(symbols) == 1:
                lookup.setdefault(alias, next(iter(symbols)))  # an approved symbol always wins over an alias
        return cls(lookup, names)

    def resolve(self, term: str) -> str | None:
        term = term.strip().strip("*`")
        if not term or not _looks_like_a_symbol(term):
            return None
        key = term.upper()
        stripped = GENE_SUFFIX.sub("", key)
        if key in NOT_GENES or stripped in NOT_GENES:
            return None
        if key in self.lookup:
            return self.lookup[key]
        return self.lookup.get(stripped)


@dataclass
class NoteTerms:
    stem: str
    category: str
    kinds: dict[str, str] = field(default_factory=dict)          # slug -> "gene" | "term"
    titles: dict[str, str] = field(default_factory=dict)         # slug -> display form as first written
    aliases: dict[str, set[str]] = field(default_factory=dict)   # slug -> surface forms seen in this note
    definitions: dict[str, str] = field(default_factory=dict)    # slug -> this note's glossary definition


def line_key(forms: list[str], hgnc: HgncTable | None, definition: str = "") -> tuple[str, str, str]:
    """Resolve identity conservatively; an explicit acronym expansion outranks an HGNC alias."""
    expansions = [f for f in forms if len(f.split()) > 1 and not GENE_SUFFIX.search(f)]
    for form in forms:
        symbol = hgnc.resolve(form) if hgnc else None
        if symbol:
            name_key = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
            if expansions and not any(name_key(f) == name_key(hgnc.names.get(symbol, "")) for f in expansions):
                continue
            cleaned = GENE_SUFFIX.sub("", form.strip().strip("*`")).upper()
            # Short aliases collide with unrelated scientific abbreviations. Without an explicit
            # approved symbol or its official expansion, preserve the glossary term as written.
            if cleaned not in hgnc.approved and not any(ch.isdigit() for ch in cleaned) and not expansions and not re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", definition):
                continue
            return slugify(symbol), "gene", symbol
    title = expansions[0] if expansions else forms[0]
    return slugify(normalise(title)), "term", title


def note_terms(stem: str, category: str, text: str, hgnc: HgncTable | None) -> NoteTerms:
    out = NoteTerms(stem=stem, category=category)
    for raw, definition in glossary_entries(text):
        # A slash-delimited list can name different genes, assays or cell types. Parse each
        # element independently; only canonical identity can subsequently combine them.
        for part in SLASH_ALIAS.split(LEFT_OF_COMPARISON.split(raw, maxsplit=1)[0]):
            forms = split_aliases(part)
            if not forms:
                continue
            symbols = {hgnc.resolve(f) for f in forms} - {None} if hgnc else set()
            groups = [[f] for f in forms] if len(symbols) > 1 else [forms]
            for group in groups:
                slug, kind, title = line_key(group, hgnc, definition)
                if not slug or normalise(title) in STOP_TERMS:
                    continue
                out.kinds.setdefault(slug, kind)
                out.titles.setdefault(slug, title)
                out.aliases.setdefault(slug, set()).update(group)
                out.definitions.setdefault(slug, definition)
    return out


@dataclass
class Candidate:
    slug: str
    title: str
    kind: str                       # gene | term
    aliases: list[str]
    count_total: int
    count_in_scope: int
    glossary_stems: list[str]
    samples: list[str]              # up to two glossary definitions, for the model's typing pass
    mention_stems: list[str] = field(default_factory=list)
    entity_type: str = "other"      # gene | method | cohort | phenomenon | other, set by the model pass


def safe_model_merges(candidates: list[Candidate], merges: dict[str, str]) -> tuple[dict[str, str], list[dict]]:
    """Accept automatic synonym merges only with a shared non-abbreviation surface form.

    HGNC has already canonicalized gene identities. Distinct gene concepts and gene/term
    pairs cannot be merged by a model. Human overrides remain a separate reviewed input.
    """
    by_slug = {c.slug: c for c in candidates}
    accepted, rejected = {}, []
    for source, target in flatten_merges(merges).items():
        a, b = by_slug.get(source), by_slug.get(target)
        reason = None
        if a is None or b is None:
            reason = "unknown candidate"
        elif a.kind == "gene" or b.kind == "gene":
            reason = "gene identity must be established by HGNC, not a model merge"
        else:
            def names(c):
                return {normalise(f) for f in [c.title, *c.aliases]
                        if len(f.split()) > 1 and not SHORT_UPPER_FORM.fullmatch(f)}
            if not names(a).intersection(names(b)):
                reason = "no shared expanded synonym; related concepts or acronym collisions stay distinct"
        if reason:
            rejected.append({"source": source, "target": target, "reason": reason})
        else:
            accepted[source] = target
    return accepted, rejected


def flatten_merges(merge: dict[str, str]) -> dict[str, str]:
    """Follow each source to its terminal target so a two-hop merge counts under one slug.

    A cycle (source eventually leads back to itself) leaves that source unmapped rather than
    looping forever.
    """
    resolved: dict[str, str] = {}
    for source in merge:
        seen: set[str] = set()
        current = source
        while current in merge and current not in seen:
            seen.add(current)
            current = merge[current]
        if current in seen:
            continue
        resolved[source] = current
    return resolved


def count_candidates(notes: list[NoteTerms], *, scope: set[str] | None, threshold: int,
                     stop_terms: set[str] | frozenset[str], merge: dict[str, str], exclude: list[str]) -> list[Candidate]:
    """Terms named by at least ``threshold`` in-scope notes' glossaries, counted over every note given.

    ``merge`` maps a slug onto the slug it should count as (user overrides and the model's earlier
    synonym decisions), followed transitively to its terminal target; ``exclude`` drops slugs
    entirely; ``stop_terms`` are matched against the normalised title, so 'GWAS' and 'gwas' both
    stop. Notes are visited in stem order so the samples and the majority title below are
    deterministic regardless of the order the caller passes them in.
    """
    merge = flatten_merges(merge)
    exclude = set(exclude)
    stop = {normalise(s) for s in stop_terms} | set(stop_terms)
    by_slug: dict[str, dict] = {}
    for note in sorted(notes, key=lambda n: n.stem):
        in_scope = scope is None or note.category in scope
        for raw_slug, kind in note.kinds.items():
            slug = merge.get(raw_slug, raw_slug)
            if slug in exclude or normalise(note.titles[raw_slug]) in stop or slug in stop:
                continue
            entry = by_slug.setdefault(slug, {"kind": kind, "aliases": set(), "stems": set(), "in_scope": set(),
                                              "samples": [], "title_counts": Counter(), "gene_title": None})
            title = note.titles[raw_slug]
            entry["title_counts"][title] += 1
            if kind == "gene":
                entry["kind"], entry["gene_title"] = "gene", title
            entry["aliases"].update(note.aliases[raw_slug])
            entry["stems"].add(note.stem)
            if in_scope:
                entry["in_scope"].add(note.stem)
            if len(entry["samples"]) < 2:
                entry["samples"].append(note.definitions[raw_slug])
    out = []
    for slug, e in by_slug.items():
        if len(e["in_scope"]) < threshold:
            continue
        # Most-named surface form wins the title; ties go to the longer form, then alphabetically.
        title = e["gene_title"] or min(e["title_counts"].items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))[0]
        out.append(Candidate(slug=slug, title=title, kind=e["kind"], aliases=sorted(e["aliases"], key=str.lower),
                             count_total=len(e["stems"]), count_in_scope=len(e["in_scope"]),
                             glossary_stems=sorted(e["stems"]), samples=e["samples"]))
    out.sort(key=lambda c: (-c.count_in_scope, c.slug))
    return out


def merge_candidates(candidates: list[Candidate], merges: dict[str, str], *,
                     in_scope: Callable[[str], bool]) -> list[Candidate]:
    """Fold each source candidate into its target (the model's synonym decisions, followed
    transitively to a terminal slug so a multi-hop merge lands regardless of input order) and
    recount the scope."""
    merges = flatten_merges(merges)
    by_slug = {c.slug: c for c in candidates}
    for source, target in merges.items():
        if source == target or source not in by_slug or target not in by_slug:
            continue
        s, t = by_slug.pop(source), by_slug[target]
        t.aliases = sorted(set(t.aliases) | set(s.aliases) | {s.title}, key=str.lower)
        t.glossary_stems = sorted(set(t.glossary_stems) | set(s.glossary_stems))
        t.mention_stems = sorted(set(t.mention_stems) | set(s.mention_stems))
        t.count_total = len(t.glossary_stems)
        t.count_in_scope = sum(1 for stem in t.glossary_stems if in_scope(stem))
        t.samples = (t.samples + s.samples)[:2]
    return sorted(by_slug.values(), key=lambda c: (-c.count_in_scope, c.slug))


def apply_title_overrides(candidates: list[Candidate], titles: dict[str, str]) -> list[Candidate]:
    for c in candidates:
        if c.slug in titles and titles[c.slug].strip():
            c.title = titles[c.slug].strip()
    return candidates


class MentionIndex:
    """Which notes contain a candidate's surface forms as whole words in the body.

    One tokenising pass over every note builds postings keyed by each form's longest token (ties
    keep the first at that length) rather than its first token, since a short first word ('de',
    'a') would otherwise post nearly every note under it. This prefilter is selective when that
    longest token is rare across the corpus, and barely narrows anything when it is common; either
    way, ``confirm`` re-checks with a whole-word regex and decides correctness itself. Gene symbols
    match case-sensitively (SHANK3, not 'shank') with no plural. Other terms match
    case-insensitively and allow a trailing 's'/'es', except a short all-caps alias (length <= 3,
    such as 'NO' for nitric oxide) which stays case-sensitive and singular even for a non-gene
    candidate, so it does not swallow the everyday word it collides with.
    """

    def __init__(self, candidates: list[Candidate]):
        self.key_tokens: dict[str, set[str]] = defaultdict(set)
        self.patterns: dict[str, re.Pattern[str]] = {}
        self.postings: dict[str, set[str]] = defaultdict(set)
        for c in candidates:
            forms = sorted({c.title, *c.aliases}, key=len, reverse=True)
            for form in forms:
                tokens = TOKEN.findall(form)
                if tokens:
                    longest = max(tokens, key=len)  # max() keeps the first token on a length tie
                    self.key_tokens[longest.lower()].add(c.slug)
            if c.kind == "gene":
                alternation = "|".join(re.escape(f) for f in forms if f)
                self.patterns[c.slug] = re.compile(rf"(?<![A-Za-z0-9])(?:{alternation})(?![A-Za-z0-9])")
            else:
                short = [f for f in forms if f and SHORT_UPPER_FORM.fullmatch(f)]
                long_forms = [f for f in forms if f and not SHORT_UPPER_FORM.fullmatch(f)]
                groups = []
                if short:
                    groups.append(f"(?-i:{'|'.join(re.escape(f) for f in short)})")
                if long_forms:
                    groups.append(f"(?:{'|'.join(re.escape(f) for f in long_forms)})(?:s|es)?")
                alternation = "|".join(groups)
                self.patterns[c.slug] = re.compile(
                    rf"(?<![A-Za-z0-9])(?:{alternation})(?![A-Za-z0-9])", re.IGNORECASE)

    def add(self, stem: str, body: str) -> None:
        seen: set[str] = set()
        for token in TOKEN.findall(body):
            low = token.lower()
            if low in self.key_tokens and low not in seen:
                seen.add(low)
                for slug in self.key_tokens[low]:
                    self.postings[slug].add(stem)

    def confirm(self, slug: str, bodies: dict[str, str]) -> list[str]:
        pattern = self.patterns[slug]
        return sorted(stem for stem in self.postings.get(slug, ()) if pattern.search(bodies.get(stem, "")))


def co_occurrence(members: dict[str, set[str]], slug: str, top: int = 10) -> list[tuple[str, int]]:
    mine = members.get(slug, set())
    pairs = [(other, len(mine & stems)) for other, stems in members.items() if other != slug]
    return sorted([p for p in pairs if p[1] > 0], key=lambda p: (-p[1], p[0]))[:top]
