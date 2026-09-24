"""Which journals Byeori may collect from, and which it may look at.

Three questions, three answers (user, 2026-09-22 and 2026-09-23).

**May it be collected from OpenAlex?** ``journal_verdict`` answers that. The 65 journals in
``JOURNALS`` are the ones the Optimus scout scans daily
(`Codex/automation-ops/wiki/jobs/optimus-paper-scout.md`); ``LAB_ADDITIONS`` are titles the lab
reads that the scout does not scan (Journal of Data Science, ICLR, NeurIPS, ICML, PMLR with AISTATS
and COLT), and
``PREPRINT_SERVERS`` are bioRxiv, medRxiv and arXiv. The two lists may differ: the user, on
2026-09-23, "the Optimus list is the journals I want to find every day; Byeori's list is the papers
we want to read, whatever the Optimus list says".

**May a paper the lab uploaded be read into the wiki?** ``upload_verdict`` answers that, and it is
wider still. A PDF in the shared `to-s3` folder or one the user downloaded has already been chosen
by a person, so it is refused only for a house or title in ``DENIED_PUBLISHERS`` or
``DENIED_JOURNALS`` (MDPI and Frontiers among them, "물론 여기서도 MDPI나 프론티어스는 금지").

**May it be looked at?** ``search_verdict`` answers that, and it is deliberately wider. When the
user asks Byeori to find work on a topic, OpenAlex may return anything except the houses and
titles in ``DENIED_PUBLISHERS`` and ``DENIED_JOURNALS``, which are refused outright. Everything
else outside the 65 comes back as ``outside_list``: it may be read and reported, but the result
has to say so, because it has not been through the user's own selection.

**Matching prefers an ISSN.** Every one of the 65 carries the ISSNs the Optimus scout verified
against `api.openalex.org/sources`. A title drifts -- OpenAlex holds ``Science (New York, N.Y.)``,
``Nature reviews. Genetics`` and registers JAMA Pediatrics under its former name -- and an ISSN
does not. The title is still matched when no ISSN is known, which is most of the existing
catalogue: 5,220 of the 11,933 papers with a source note carry no journal name at all.

Changing this list decides what may be collected next. It does not reach the wiki: nothing in the
reading or search path consults a verdict, and a paper already read and noted stays (user,
2026-09-22).
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

INCLUDE = "include"
REVIEW = "review"
EXCLUDE = "exclude"

# Search verdicts, which are not collection verdicts.
ALLOWED = "allowed"            # on the collection list
OUTSIDE_LIST = "outside_list"  # may be searched and reported, with the warning that says so
FORBIDDEN = "forbidden"        # never, whatever the topic

# The lists live in one JSON file so an installing lab can replace them without editing code.
# The shipped file is this lab's list. BYEORI_JOURNAL_POLICY points at another file.
POLICY_PATH: Path = Path(__file__).with_name("policies") / "journals.json"


def load_policy(path: Path | None = None) -> dict[str, Any]:
    """Read the journal policy file; a missing file is an error that names the path."""
    chosen = path or Path(os.environ.get("BYEORI_JOURNAL_POLICY") or POLICY_PATH)
    if not chosen.is_file():
        raise FileNotFoundError(f"journal policy file not found: {chosen}")
    with chosen.open(encoding="utf-8") as handle:
        return json.load(handle)


_POLICY = load_policy()
JOURNALS: dict[str, tuple[str, str, tuple[str, ...], str]] = {
    key: (row["title"], row["family"], tuple(row["issns"]), row["source_id"])
    for key, row in _POLICY["journals"].items()}

# The preprint servers, and the display names OpenAlex gives them. Collectable, never discovered:
# the user reviews a preprint and uploads it by hand, so no automated scout reaches these. arXiv
# joined on 2026-09-23 (user); OpenAlex writes it "arXiv (Cornell University)".
PREPRINT_SERVERS: tuple[str, ...] = tuple(_POLICY["preprint_servers"])

# Titles the lab reads that the Optimus scout does not scan (user, 2026-09-23). Same shape as
# JOURNALS except that a title may carry several OpenAlex sources: NeurIPS is registered once as
# the conference and once as the "neural information processing systems" proceedings with ISSN
# 1049-5258. Ids and ISSNs read from api.openalex.org/sources on 2026-09-23.
# PMLR is on the list too (user, 2026-09-23: "PMLR 도 허용목록에 추가해줘요"). OpenAlex files few papers
# under PMLR itself (S4363608721, 7 works) and most under the conference PMLR publishes, so the
# series it publishes that OpenAlex keeps apart are listed beside it: ICML, AISTATS and COLT. A PMLR
# paper has no DOI; its DOI is left empty.
LAB_ADDITIONS: dict[str, tuple[str, str, tuple[str, ...], tuple[str, ...]]] = {
    key: (row["title"], row["family"], tuple(row["issns"]), tuple(row["source_ids"]))
    for key, row in _POLICY["lab_additions"].items()}

# Refused however relevant the topic is (user, 2026-09-22). The first five are the user's own
# words; the rest carry over llm-wiki's standing exclusions, which the user confirmed the same day.
# A publisher is named where the refusal is house-wide, because naming its journals one by one
# never finishes.
# The house names are the ones OpenAlex actually writes in ``host_organization_name``, read from
# api.openalex.org on 2026-09-22, not the short names the user and the lab say. A live search
# returned MDPI's `Nutrients` as merely outside the list because the entry said "mdpi" and OpenAlex
# says "Multidisciplinary Digital Publishing Institute"; both spellings are kept now.
DENIED_PUBLISHERS: tuple[str, ...] = tuple(_POLICY["denied_publishers"])
DENIED_JOURNALS: tuple[str, ...] = tuple(_POLICY["denied_journals"])

# Nature-portfolio titles weighed and left below the user's threshold. Kept so nobody re-adds one
# by mistake. ``scientific reports`` is also in DENIED_JOURNALS: below the threshold to collect,
# and refused outright when searching.
NATURE_PORTFOLIO_BELOW_THRESHOLD: tuple[str, ...] = tuple(_POLICY["nature_portfolio_below_threshold"])


def normalize_journal(name: str | None) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", (name or "").lower()).strip()
    text = re.sub(r"\s+", " ", text)
    return text[4:] if text.startswith("the ") else text


def _normalize_issn(value: Any) -> str:
    return re.sub(r"[^0-9x]", "", str(value or "").lower())


# family -> normalized titles, the shape the rest of the package already reads.
ALLOWLIST: dict[str, tuple[str, ...]] = {}
for _key, (_title, _family, _issns, _source_id) in JOURNALS.items():
    ALLOWLIST[_family] = ALLOWLIST.get(_family, ()) + (_key,)
for _key, (_title, _family, _issns, _source_ids) in LAB_ADDITIONS.items():
    ALLOWLIST[_family] = ALLOWLIST.get(_family, ()) + (_key,)
ALLOWLIST["Preprint server"] = PREPRINT_SERVERS

_INCLUDE_INDEX: dict[str, str] = {key: family for family, keys in ALLOWLIST.items() for key in keys}
_ISSN_INDEX: dict[str, str] = {
    _normalize_issn(issn): key for key, (_t, _f, issns, _s) in JOURNALS.items() for issn in issns
} | {_normalize_issn(issn): key for key, (_t, _f, issns, _s) in LAB_ADDITIONS.items() for issn in issns}
_SOURCE_INDEX: dict[str, str] = {source_id: key for key, (_t, _f, _i, source_id) in JOURNALS.items()} | {
    source_id: key for key, (_t, _f, _i, source_ids) in LAB_ADDITIONS.items() for source_id in source_ids}

# What an OpenAlex ``primary_location.source.id`` filter is given to keep a search on the list:
# the 65 journals and the lab's additions, under the API's limit of 100 values per filter.
SOURCE_IDS: tuple[str, ...] = tuple(_SOURCE_INDEX)
_DENIED_PUBLISHERS = {normalize_journal(name) for name in DENIED_PUBLISHERS}
_DENIED_JOURNALS = {normalize_journal(name) for name in DENIED_JOURNALS}

# Every title the user weighed and declined on 2026-09-17 has since been admitted or is simply not
# on the list; the separate same-family table is gone. What remains of that decision is the rule
# itself: a title nobody named is not collectable just because a sibling is.
SAME_FAMILY_EXCLUDED: dict[str, tuple[str, ...]] = {}


def _match(journal: str | None, issn: Iterable[str] | str | None,
           source_id: str | None = None) -> str | None:
    """The normalized title this source is: by OpenAlex id, then ISSN, then name."""
    if source_id:
        key = _SOURCE_INDEX.get(str(source_id).rsplit("/", 1)[-1])
        if key:
            return key
    values = [issn] if isinstance(issn, str) else list(issn or ())
    for value in values:
        key = _ISSN_INDEX.get(_normalize_issn(value))
        if key:
            return key
    normalized = normalize_journal(journal)
    return normalized if normalized in _INCLUDE_INDEX else None


def journal_verdict(journal: str | None, issn: Iterable[str] | str | None = None,
                    source_id: str | None = None) -> dict[str, str | None]:
    """May a paper from this source be collected? ``include`` only for the list."""
    key = _match(journal, issn, source_id)
    if key:
        family = _INCLUDE_INDEX[key]
        reason = ("on the collection list, matched by ISSN" if _match(None, issn)
                  else "on the collection list")
        return {"verdict": INCLUDE, "family": family, "reason": reason}
    normalized = normalize_journal(journal)
    if not normalized:
        return {"verdict": EXCLUDE, "family": None, "reason": "journal metadata missing"}
    if normalized in _DENIED_JOURNALS:
        return {"verdict": EXCLUDE, "family": None, "reason": "refused by the user, whatever the topic"}
    if normalized in NATURE_PORTFOLIO_BELOW_THRESHOLD:
        return {"verdict": EXCLUDE, "family": "Nature portfolio",
                "reason": "Nature-portfolio title below the impact threshold"}
    return {"verdict": EXCLUDE, "family": None, "reason": "not on the collection list"}


def search_verdict(journal: str | None, issn: Iterable[str] | str | None = None,
                   publisher: str | None = None, source_id: str | None = None) -> dict[str, str | None]:
    """May Byeori look at this source when the user asks it to find work on a topic?

    ``forbidden`` is refused whatever the topic. ``outside_list`` may be read and reported, and
    the caller has to carry ``warning`` into what it shows the user.
    """
    house = normalize_journal(publisher)
    if house and any(house == denied or house.startswith(denied + " ")
                     for denied in _DENIED_PUBLISHERS):
        return {"verdict": FORBIDDEN, "family": None, "warning": None,
                "reason": f"{publisher} is refused by the user, whatever the topic"}
    normalized = normalize_journal(journal)
    if normalized in _DENIED_JOURNALS:
        return {"verdict": FORBIDDEN, "family": None, "warning": None,
                "reason": f"{journal} is refused by the user, whatever the topic"}
    key = _match(journal, issn, source_id)
    if key:
        return {"verdict": ALLOWED, "family": _INCLUDE_INDEX[key], "warning": None,
                "reason": "on the collection list"}
    named = journal or "this source"
    return {"verdict": OUTSIDE_LIST, "family": None,
            "warning": f"{named} is not on the lab's journal list; it has not been through the user's selection.",
            "reason": "outside the collection list"}


def upload_verdict(journal: str | None, issn: Iterable[str] | str | None = None,
                   publisher: str | None = None, source_id: str | None = None) -> dict[str, str | None]:
    """May a paper the lab uploaded go into the wiki? Yes, unless its house or title is refused.

    A person chose the PDF, so the discovery list does not decide it (user, 2026-09-23); only
    ``search_verdict``'s refusals do. A paper with no journal metadata has nothing to be refused on.
    """
    searched = search_verdict(journal, issn, publisher, source_id)
    if searched["verdict"] == FORBIDDEN:
        return {"verdict": EXCLUDE, "family": None, "reason": searched["reason"]}
    if searched["verdict"] == ALLOWED:
        return {"verdict": INCLUDE, "family": searched["family"], "reason": "on the collection list"}
    return {"verdict": INCLUDE, "family": None,
            "reason": "chosen by the lab (uploaded); outside the discovery list, not refused"}


def apply_upload_policy(work: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of an uploaded paper's record with its upload verdict recorded under ``corpus``."""
    verdict = upload_verdict(work.get("source"), work.get("source_issn"), work.get("source_publisher"),
                             work.get("source_id"))
    annotated = dict(work)
    corpus = dict(annotated.get("corpus") or {})
    corpus["journal_verdict"] = verdict["verdict"]
    corpus["journal_family"] = verdict["family"]
    corpus["journal_reason"] = verdict["reason"]
    corpus["journal_rule"] = "upload"
    annotated["corpus"] = corpus
    return annotated


def apply_journal_policy(work: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``work`` with the collection verdict recorded under ``corpus``."""
    verdict = journal_verdict(work.get("source"), work.get("source_issn"), work.get("source_id"))
    annotated = dict(work)
    corpus = dict(annotated.get("corpus") or {})
    corpus["journal_verdict"] = verdict["verdict"]
    corpus["journal_family"] = verdict["family"]
    corpus["journal_reason"] = verdict["reason"]
    annotated["corpus"] = corpus
    return annotated


def apply_search_policy(work: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``work`` with the search verdict and any warning recorded under ``corpus``."""
    verdict = search_verdict(work.get("source"), work.get("source_issn"), work.get("source_publisher"),
                             work.get("source_id"))
    annotated = dict(work)
    corpus = dict(annotated.get("corpus") or {})
    corpus["search_verdict"] = verdict["verdict"]
    corpus["search_reason"] = verdict["reason"]
    corpus["search_warning"] = verdict["warning"]
    annotated["corpus"] = corpus
    return annotated


def allowed_verdicts(policy: str) -> tuple[str, ...] | None:
    """Map a search policy name to the verdicts it admits; ``None`` means no filter."""
    if policy == INCLUDE:
        return (INCLUDE,)
    if policy == REVIEW:
        return (INCLUDE, REVIEW)
    if policy == "all":
        return None
    raise ValueError("journal policy must be include, review, or all")
