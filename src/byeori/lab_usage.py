"""Monthly usage totals per member, read from the budget ledger the workers already write.

Every paid call settles into three scopes: the job, ``lab:{period}`` and ``member:{id}:{period}``
(``lab_budget.scopes``). Nothing new is recorded here; this module only reads those records and the
member job pointers so an administrator can see who spent what in a month. Amounts are the
application's own estimate from the repository price table, never an AWS invoice.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from byeori import lab_budget, lab_jobs, lab_members
from byeori.lab_store import TablePort, keys

__all__ = ["PERIOD_PATTERN", "monthly_usage", "period_of"]

PERIOD_PATTERN = lab_budget._PERIOD           # noqa: SLF001 - one definition of YYYY-MM for the whole service
MAX_POINTER_PAGES = 40                        # 100 pointers per page; enough for a month of one member
JOB_KINDS = (lab_jobs.KIND_ANSWER, lab_jobs.KIND_RESEARCH)


def period_of(moment: Any) -> str:
    """``YYYY-MM`` of a timezone-aware datetime, the same period the ledger uses."""
    return lab_budget.period_for(moment)


def _scope_totals(table: TablePort, scope: str) -> dict[str, int]:
    record = table.get(*keys.budget(scope))
    if record is None:
        return {"settled_micros": 0, "reserved_micros": 0}
    return {"settled_micros": int(record.get("settled_micros") or 0),
            "reserved_micros": int(record.get("reserved_micros") or 0)}


def _counts(table: TablePort, member_id: str, period: str) -> dict[str, int]:
    """How many jobs of each kind the member started in ``period``, from the member pointers."""
    counts = {kind: 0 for kind in JOB_KINDS}
    counts["other"] = 0
    cursor: str | None = None
    prefix = f"JOB#{period}"
    for _page in range(MAX_POINTER_PAGES):
        pointers, cursor = table.query(keys.member(member_id)[0], sk_prefix=prefix, limit=100, start_after=cursor)
        for pointer in pointers:
            kind = str(pointer.get("kind") or lab_jobs.KIND_ANSWER)
            counts[kind if kind in counts else "other"] += 1
        if cursor is None:
            break
    return counts


def monthly_usage(table: TablePort, period: str, *, member_id: str | None = None) -> dict[str, Any]:
    """Per-member and lab totals for one month.

    ``member_id`` limits the report to one member; without it every registered member is listed,
    newest spend first. ``lab`` is the lab scope total, which also covers members removed from the
    registry, so it can exceed the sum of the rows; ``unattributed_micros`` reports that gap.
    """
    if not PERIOD_PATTERN.match(period or ""):
        raise ValueError("period must look like YYYY-MM")
    if member_id is not None and not isinstance(member_id, str):
        raise ValueError("member_id must be a string")
    profiles = lab_members.list_members(table)
    if member_id is not None:
        profiles = [profile for profile in profiles if profile.get("member_id") == member_id]
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        identifier = str(profile.get("member_id"))
        totals = _scope_totals(table, lab_budget.scopes.member(identifier, period))
        counts = _counts(table, identifier, period)
        rows.append({
            "member_id": identifier,
            "role": profile.get("role"),
            "active": profile.get("active"),
            "settled_micros": totals["settled_micros"],
            "reserved_micros": totals["reserved_micros"],
            "usd": round(totals["settled_micros"] / 1e6, 4),
            "answers": counts.get(lab_jobs.KIND_ANSWER, 0),
            "research": counts.get(lab_jobs.KIND_RESEARCH, 0),
        })
    rows.sort(key=lambda row: (-row["settled_micros"], row["member_id"]))
    lab = _scope_totals(table, lab_budget.scopes.lab(period))
    attributed = sum(row["settled_micros"] for row in rows)
    return {
        "period": period,
        "members": rows,
        "lab": {**lab, "usd": round(lab["settled_micros"] / 1e6, 4)},
        "unattributed_micros": max(0, lab["settled_micros"] - attributed) if member_id is None else None,
        "note": ("Amounts are this application's estimate from its price table, not an AWS invoice. "
                 "Jev triage calls are not settled into this ledger."),
    }
