"""What Byeori does when the wiki cannot answer: record the gap, offer to look, wait for approval.

The rule the user wrote on 2026-09-20: never search the web; answer from the wiki, and when the
wiki has nothing, read the paper itself; when there is still nothing to answer with, say so and ask
for papers on the subject to be found and ingested. This module is the "ask" in that sentence, and
the user's decision of 2026-09-22 is that it happens on both sides at once:

- **The member is offered it.** An answer whose evidence was thin carries a ``collection_offer``.
  Accepting it records a request and nothing else: no search runs and no money is spent on a
  member's word, because collection needs the professor's approval.
- **The professor gets it anyway.** The gap is recorded whether or not anybody accepted, so a
  subject the wiki keeps failing on is visible without depending on a member noticing. A declined
  offer stays in the queue for the same reason.

So one row per question carries the whole life of the gap::

    unrequested --accept--> requested   --approve--> approved --> collected
                \\--decline-> declined  --approve--> approved
                                        \\--reject--> rejected

``approved`` is where the OpenAlex search takes over. Nothing here searches, saves a candidate or
ingests anything; it decides that a question went unanswered and who has been told.

The query kept on the row is the English one the answer worker already computed for its own
supplemental search, falling back to the standalone question. It is never the member's Korean
sentence: the lab's 65 journals are English, and a Korean query returns Korean papers, which was
measured on 2026-09-22. No model is called to write it.
"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

from byeori.lab_store import ConditionFailed, Put, StoreError, TablePort, Update, keys, new_item, now_iso

__all__ = [
    "APPROVED", "COLLECTED", "DECLINED", "GAP_STATUSES", "NotFound", "REJECTED", "REQUESTED",
    "UNREQUESTED", "approve", "gap_for_job", "gap_reason", "list_gaps", "offer_view", "record_gap",
    "reject", "respond",
]

UNREQUESTED = "unrequested"   # the wiki fell short; nobody has asked for collection yet
REQUESTED = "requested"       # the member accepted the offer
DECLINED = "declined"         # the member declined; the professor still sees it
APPROVED = "approved"         # the professor authorised collection; the search may run
REJECTED = "rejected"         # the professor decided not to collect for this question
COLLECTED = "collected"       # the search ran and its candidates were saved
GAP_STATUSES = frozenset({UNREQUESTED, REQUESTED, DECLINED, APPROVED, REJECTED, COLLECTED})

# What a member may still do, and to what.
ACCEPT, DECLINE = "accept", "decline"
DECISIONS = frozenset({ACCEPT, DECLINE})
_STATUS_OF_DECISION = {ACCEPT: REQUESTED, DECLINE: DECLINED}
MEMBER_MAY_DECIDE = frozenset({UNREQUESTED, REQUESTED, DECLINED})
PROFESSOR_MAY_DECIDE = frozenset({UNREQUESTED, REQUESTED, DECLINED, REJECTED})

# The two shapes of "the wiki did not have it".
NO_EVIDENCE = "no_evidence"   # the packet held nothing to answer from
THIN_EVIDENCE = "thin_evidence"  # an answer was written but it cited nothing

MAX_QUERY_CHARS = 300
OFFER_MESSAGE = (
    "위키에서 이 질문에 답할 근거를 충분히 찾지 못했습니다. "
    "관련 논문을 찾아 위키에 추가할 후보로 올릴까요?"
)
OFFER_NOTE = "동의하면 교수님 검토 목록에 수집 요청으로 기록됩니다. 검색과 수집은 승인 뒤에 실행됩니다."
LIST_PAGES = 5                # list_gaps reads at most this many 100-row pointer pages


class NotFound(StoreError):
    """No gap recorded for that job."""

    code = "not_found"


class InvalidTransition(StoreError):
    """The gap is not in a state that allows this decision."""

    code = "invalid_transition"


def gap_reason(answer: Mapping[str, Any]) -> str | None:
    """Why this answer counts as the wiki falling short, or ``None`` when it does not.

    Read off what the worker already recorded. ``insufficient`` is the worker's own word for a
    packet it could not answer from. An answer that cites nothing is the other case: prose was
    produced, but no page in the wiki stands behind it.
    """
    if not isinstance(answer, Mapping):
        return None
    if answer.get("evidence_state") == "insufficient":
        return NO_EVIDENCE
    if answer.get("status") in {"completed", "partial"} and not (answer.get("citations") or []):
        return THIN_EVIDENCE
    return None


def collection_query(answer: Mapping[str, Any]) -> str:
    """The English query to search with: the worker's own, else the standalone question."""
    for name in ("english_query", "standalone_question", "question"):
        value = answer.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()[:MAX_QUERY_CHARS]
    return ""


def record_gap(table: TablePort, answer: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any] | None:
    """Record that this question went unanswered. ``None`` when the answer was fine.

    The row is created once and a replay leaves the existing one alone, so re-running a job never
    resets a decision somebody already made on it.
    """
    reason = gap_reason(answer)
    if reason is None:
        return None
    job_id = answer.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("an answer needs a job_id to record its gap")
    stamp = now_iso(now)
    record = new_item(
        *keys.collection_gap(job_id), stamp, job_id=job_id, member_id=answer.get("member_id"),
        session_id=answer.get("session_id"), question=answer.get("question"),
        query=collection_query(answer), reason=reason, status=UNREQUESTED,
        evidence_state=answer.get("evidence_state"), hold_reason=answer.get("hold_reason"),
        policy_revision=answer.get("policy_revision"),
    )
    pointer = new_item(*keys.collection_gap_pointer(stamp, job_id), stamp, job_id=job_id,
                       status=UNREQUESTED, reason=reason)
    try:
        table.transact([Put(record), Put(pointer)])   # Put is create-only in lab_store
    except ConditionFailed:
        return table.get(*keys.collection_gap(job_id))
    return record


def gap_for_job(table: TablePort, job_id: str) -> dict[str, Any] | None:
    return table.get(*keys.collection_gap(job_id))


def offer_view(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """What the member is shown. ``None`` when there is no gap to offer anything about."""
    if not record:
        return None
    return {
        "job_id": record.get("job_id"),
        "status": record.get("status"),
        "reason": record.get("reason"),
        "message": OFFER_MESSAGE,
        "note": OFFER_NOTE,
        "decisions": sorted(DECISIONS),
        "decided": record.get("status") not in {UNREQUESTED},
    }


def _transition(table: TablePort, job_id: str, target: str, allowed: frozenset[str],
                changes: Mapping[str, Any], now: datetime | None) -> dict[str, Any]:
    record = table.get(*keys.collection_gap(job_id))
    if record is None:
        raise NotFound(f"no collection gap recorded for {job_id}")
    if record["status"] == target:
        return record
    if record["status"] not in allowed:
        raise InvalidTransition(f"a gap that is {record['status']} cannot become {target}")
    stamp = now_iso(now)
    updates = {"status": target, **changes}
    pointer_key = keys.collection_gap_pointer(record["created_at"], job_id)
    pointer = table.get(*pointer_key)
    operations = [Update(*keys.collection_gap(job_id), record["revision"], updates)]
    if pointer is not None:
        operations.append(Update(*pointer_key, pointer["revision"], {"status": target}))
    table.transact(operations)
    return {**record, **updates, "updated_at": stamp}


def respond(table: TablePort, job_id: str, member_id: str, decision: str, *,
            now: datetime | None = None) -> dict[str, Any]:
    """The member's answer to the offer. An accept records a request; it starts nothing."""
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {', '.join(sorted(DECISIONS))}")
    record = table.get(*keys.collection_gap(job_id))
    if record is None or (record.get("member_id") and record.get("member_id") != member_id):
        # Another member's gap is not visible as somebody else's; it simply is not found.
        raise NotFound(f"no collection gap recorded for {job_id}")
    target = _STATUS_OF_DECISION[decision]
    return _transition(table, job_id, target, MEMBER_MAY_DECIDE,
                       {"decision": decision, "decided_at": now_iso(now), "decided_by": member_id}, now)


def approve(table: TablePort, job_id: str, approved_by: str, *, query: str | None = None,
            now: datetime | None = None) -> dict[str, Any]:
    """The professor authorises collection for this question. The search runs after this."""
    changes: dict[str, Any] = {"approved_at": now_iso(now), "approved_by": approved_by}
    if query:
        changes["query"] = str(query)[:MAX_QUERY_CHARS]
    return _transition(table, job_id, APPROVED, PROFESSOR_MAY_DECIDE, changes, now)


def reject(table: TablePort, job_id: str, rejected_by: str, *, reason: str | None = None,
           now: datetime | None = None) -> dict[str, Any]:
    changes: dict[str, Any] = {"rejected_at": now_iso(now), "rejected_by": rejected_by}
    if reason:
        changes["rejected_reason"] = str(reason)[:MAX_QUERY_CHARS]
    return _transition(table, job_id, REJECTED, PROFESSOR_MAY_DECIDE, changes, now)


def mark_collected(table: TablePort, job_id: str, *, candidates: int,
                   now: datetime | None = None) -> dict[str, Any]:
    """The approved search ran and saved this many candidates."""
    return _transition(table, job_id, COLLECTED,
                       frozenset({APPROVED}), {"collected_at": now_iso(now),
                                               "candidates_saved": int(candidates)}, now)


def list_gaps(table: TablePort, *, status: str | None = None, limit: int = 50,
              start_after: str | None = None) -> dict[str, Any]:
    """The professor's queue, newest first is not guaranteed; the pointer sorts by time.

    Every gap is here, requested or not, because a subject the wiki keeps failing on should not
    depend on a member having noticed.
    """
    if status is not None and status not in GAP_STATUSES:
        raise ValueError(f"status must be one of {', '.join(sorted(GAP_STATUSES))}")
    partition = keys.collection_gap_pointer("", "")[0]
    gaps: list[dict[str, Any]] = []
    cursor = start_after
    for _page in range(LIST_PAGES):
        pointers, cursor = table.query(partition, sk_prefix="GAP#", limit=100, start_after=cursor)
        for pointer in pointers:
            if status is not None and pointer.get("status") != status:
                continue
            record = table.get(*keys.collection_gap(str(pointer.get("job_id"))))
            if record is not None:
                gaps.append(record)
            if len(gaps) >= limit:
                return {"gaps": gaps, "next_cursor": pointer["sk"], "counted": len(gaps)}
        if cursor is None:
            break
    return {"gaps": gaps, "next_cursor": None, "counted": len(gaps)}
