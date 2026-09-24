"""Policy constants for the student question workflow (docs/LAB-QUESTION-WORKFLOW.md).

Every stored verdict, offer and approval records the revision that produced it, so a later
change of a limit or the cutoff never rewrites how an earlier record was judged. Numbers here
are the first, adjustable settings from design section 5; they are not a guarantee that enough
evidence was read.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

POLICY_REVISION = "2026-09-21-v1"

# Jev's review_candidate probability at or above which an offer may be made (design section 6).
REVIEW_CANDIDATE_CUTOFF = 0.99

# The answer worker may call the model at most three times per job, translation included: the two
# content calls (initial, then forced submit_answer after an optional lookup) plus at most one
# retry when the output limit cut a submitted answer short before the server could read it.
ANSWER_MAX_MODEL_CALLS = 3


@dataclass(frozen=True)
class PacketLimits:
    """Bounds of one evidence packet; the total byte limit is applied before section cuts.

    The numbers changed on 2026-09-22 after a measured failure. A live question ranked six
    relevant notes and the packet carried three of them: the model's four requested reads were
    admitted first, one page contributed 14,669 of the 30,000 bytes, and the three
    highest-scoring notes (BM25 143.7, 141.1, 130.1) were dropped with ``byte_budget`` while
    ranks four to six were kept. The answer then told the member its best evidence was missing.
    Three things changed: more room, a per-document share of it (``evidence_packet.build_packet``
    gives each page what is left divided by the slots still to fill), and a tighter cut for
    context sections than for the decisive ones, so background prose cannot crowd out results.
    """

    candidates: int = 8        # BM25 hits considered per search
    max_documents: int = 8     # pages actually read
    max_sections: int = 24     # sections kept in total
    section_chars: int = 6000  # characters kept per decisive section (results, limitations)
    total_bytes: int = 96000   # UTF-8 bytes of all section texts sent to the model
    searches: int = 2          # initial search plus at most one supplemental search
    context_chars: int = 2500  # characters kept per supporting section (methods, interpretation, other)

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field.name} must be a positive integer, got {value!r}")


PACKET_LIMITS = PacketLimits(8, 8, 24, 6000, 96000, 2, 2500)

# Reads returned to a client keep the existing 8,000-character cap.
READ_MAX_CHARS = 8000

# What the answer worker may pull from a paper itself when the notes do not settle a point
# (the user's rule of 2026-09-20: when the wiki has nothing, read the original). A source note is
# a reading of a paper, so a number that sits in a table or a figure caption is often only in the
# paper. These bound what one answer adds to its packet, not what anybody may ask for: the model
# names the papers, and the packet's own byte budget still applies afterwards.
ORIGINAL_MAX_PAPERS = 4       # papers whose stored text one answer may open
# One read window, the same 8,000 characters any excerpt is capped at (the execution boundary in
# AGENTS.md). Four papers at that size is 32,000 characters, about a third of the packet's own
# byte budget, so an answer that opens the maximum still leaves room for the notes.
ORIGINAL_MAX_CHARS = READ_MAX_CHARS
ASSET_MAX_CHARS = 6_000       # of that paper's figure and table text, uploaded 2026-09-22
# Supplementary tables (stored 2026-09-23 under papers/{stem}/supplementary/). One answer may make a few
# bounded reads in its single lookup; each result is one 8,000-character window, like any excerpt.
SUPPLEMENTARY_MAX_READS = 4
SUPPLEMENTARY_MAX_CHARS = READ_MAX_CHARS
SUPPLEMENTARY_SECONDS = 40    # per read; a read that runs out says how far it scanned

# Money is integer micro-USD everywhere and every call is still reserved, settled and counted
# against the job, the member's month and the lab's month. What is gone is the ceiling that
# refused a call. ``None`` means the job scope records what it spends and never stops the work.
#
# The user's decision, stated on 2026-09-20 and again on 2026-09-22: a limit put on a research
# answer is itself the defect. An answer that stops because a number was reached is worse than
# the answer's cost, and the quality of the agent and the size of the corpus are what the result
# should depend on. Observed answers settle near 0.27 USD, so the ledger stays the place to read
# what was spent.
ANSWER_JOB_CAP_MICROS: int | None = None

# Profile of a research run started by a student's consent or a professor's approval.
RESEARCH_PROFILE = {"budget_usd_micros": 5_000_000, "reread": "auto", "max_calls": 40}

OFFER_TTL_SECONDS = 604_800
APPROVAL_TTL_SECONDS = 604_800  # a professor approval stays usable as long as a student offer
LEASE_SECONDS = 960           # answer-worker lease; longer than the 900 s Lambda timeout

# A BM25 score at or above this marks an existing concept/overview as the scope of a supplement.
SCOPE_MATCH_SCORE = 20.0

JEV_MAX_INPUT_BYTES = 30_000
JEV_MODEL = "jev-1.13.0"

# Design section 6 wording, verbatim. Offers append the consent note and name their targets.
OFFER_TEMPLATES: Mapping[str, str] = MappingProxyType({
    "new_synthesis": "현재 검색한 위키에서는 이 질문을 종합한 문서를 찾지 못했습니다. 관련 논문을 바탕으로 합성 위키를 만들까요?",
    "supplement_existing": "관련 합성 위키가 있지만 이 부분을 보완할 수 있습니다. 보완할까요?",
    "consent_note": "동의하면 서버의 연구 프로필로 비동기 연구 실행이 시작됩니다.",
    "new_targets_label": "생성할 문서:",
    "existing_targets_label": "보완할 문서:",
})


def passes_cutoff(probabilities: Any) -> bool:
    """True only when the raw review_candidate probability is a finite number >= 0.99.

    Booleans, strings, NaN and missing values fail. ``confidence`` is never compared here.
    """
    if not isinstance(probabilities, Mapping):
        return False
    p = probabilities.get("review_candidate")
    return (isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p)
            and p >= REVIEW_CANDIDATE_CUTOFF)
