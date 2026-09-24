"""Bound failure previews while retaining counts and access to complete problem text."""
from __future__ import annotations

import hashlib
import json
from collections import Counter

MAX_REPORT_BYTES = 96 * 1024
MAX_REASON_GROUPS = 20
MAX_ROW_PROBLEMS = 3
MAX_PREVIEW_BYTES = 200


def _size(value):
    return len(json.dumps(value).encode("utf-8"))


def _preview(text):
    if _size(text) <= MAX_PREVIEW_BYTES:
        return text
    suffix = "... [" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12] + "]"
    low, high = 0, min(len(text), MAX_PREVIEW_BYTES)
    while low < high:
        middle = (low + high + 1) // 2
        if _size(text[:middle] + suffix) <= MAX_PREVIEW_BYTES:
            low = middle
        else:
            high = middle - 1
    return text[:low] + suffix


def problem_detail(event, rows, *, id_key):
    """Use problem_id and problem_index on the same RPC to read a complete problem in slices."""
    ident = event["problem_id"]
    index, start, maximum = event.get("problem_index", 0), event.get("start", 0), event.get("max_chars", 4000)
    if (type(index) is not int or index < 0 or type(start) is not int or start < 0
            or type(maximum) is not int or not 1 <= maximum <= 8000):
        raise ValueError("problem_index/start must be nonnegative integers and max_chars between 1 and 8000")
    row = next((row for row in rows if row[id_key] == ident), None)
    if row is None:
        raise ValueError("Failure identifier not found")
    problems = row["problems"]
    if index >= len(problems):
        raise ValueError("Problem index is out of range")
    text = problems[index]
    excerpt = text[start:start + maximum]
    end = start + len(excerpt)
    return {id_key: ident, "problem_index": index, "problem_count": len(problems), "start": start,
            "text": excerpt, "total_chars": len(text), "next_start": end if end < len(text) else None}


def bounded_report(rows, *, offset, limit, verbose, id_key, rows_key):
    reasons = Counter(problem for row in rows for problem in row["problems"] or ["(no problem recorded)"])
    groups = reasons.most_common(MAX_REASON_GROUPS)
    result = {"failed": len(rows), "by_reason": {_preview(reason): count for reason, count in groups},
              "reason_groups_total": len(reasons), "reason_groups_omitted": max(0, len(reasons) - len(groups)),
              "reason_occurrences_total": sum(reasons.values()),
              "reason_occurrences_omitted": sum(reasons.values()) - sum(count for _, count in groups),
              "reason_previews_truncated": sum(_preview(reason) != reason for reason, _ in groups),
              rows_key: [], "offset": offset, "next_offset": None,
              "problem_detail": "Use this RPC with problem_id, problem_index, start and max_chars (up to 8000)."}
    for row in rows[offset:offset + limit]:
        if verbose:
            problems = row["problems"]
            previews = [_preview(problem) for problem in problems[:MAX_ROW_PROBLEMS]]
            entry = {**row, "problems": previews, "problem_count": len(problems),
                     "problems_omitted": max(0, len(problems) - len(previews)),
                     "problem_previews_truncated": sum(a != b for a, b in zip(previews, problems))}
        else:
            entry = row[id_key]
        result[rows_key].append(entry)
        if _size(result) > MAX_REPORT_BYTES:
            result[rows_key].pop()
            if not result[rows_key]:
                raise ValueError("Failure metadata exceeds the report budget")
            break
    end = offset + len(result[rows_key])
    result["next_offset"] = end if end < len(rows) else None
    return result
