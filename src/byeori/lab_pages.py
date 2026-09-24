"""Keep every answered lab question as Markdown in the wiki, outside the search index.

The user's decision of 2026-09-22: a student's answer already exists, so writing it as a page
costs one S3 PUT and nothing else, and that page is what llm-wiki was for. What it must not do is
enter the search index. At the lab's planned volume the questions would outnumber the source notes
within a month and take the result slots the notes need, which is the same failure the per-paper
reader layer caused in 2026-09-18.

So the pages live under ``wiki/lab-questions/``, which ``_build_wiki_index`` skips, and navigation
is by link instead of by search:

``wiki/lab-questions/{period}/{job_id}.md``
    One answered question: the question, the answer, its citations as ``[[sources/...]]`` links
    and its limitations. No member id and no conversation text, so a page shared with the lab
    never carries who asked.

``wiki/lab-questions/by-page/{folder}/{stem}.md``
    One page per cited wiki page, source note or synthesis alike, listing the questions that cited
    it, newest first. This is what an indexed page points at: the cited page carries one standing
    line to its own hub, so it is edited once in its life rather than once per question and stays
    the same length however many questions cite it.

Nothing here edits a scientific page. ``lab_store.PageWriter`` refuses every key outside
``wiki/lab-questions/``, so the answer worker cannot reach ``wiki/sources/`` even by mistake.
Publication is best effort: ``publish_answer`` returns what it wrote and what it could not, and
``lab_answer`` never fails a member's answer because a page did not save.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from byeori.lab_store import ANSWER_PAGE_PREFIX, ConditionFailed, PageWriter, now_iso

__all__ = ["HUB_PREFIX", "answer_page_key", "hub_key", "link_of", "page_link_line",
           "publish_answer", "render_answer_page"]

HUB_PREFIX = f"{ANSWER_PAGE_PREFIX}by-page/"
HUB_ATTEMPTS = 4                  # a hub is appended under its ETag; concurrent answers retry
MAX_TITLE_CHARS = 120
MAX_ENTRY_CHARS = 160
# The cited page as a wiki link: one folder and one stem, which is every indexed layer
# (sources, overviews, concepts, questions and the category pages).
_LINK = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_WHITESPACE = re.compile(r"\s+")


def link_of(key: str) -> str | None:
    """``wiki/sources/foo.md`` -> ``sources/foo``; ``None`` when the key is not an indexed page.

    A key under ``wiki/lab-questions/`` returns ``None`` too: a question does not get a hub of
    questions, and a hub never cites itself.
    """
    if not isinstance(key, str) or not key.startswith("wiki/") or not key.endswith(".md"):
        return None
    if key.startswith(ANSWER_PAGE_PREFIX):
        return None
    link = key[len("wiki/"):-len(".md")]
    return link if _LINK.match(link) else None


def answer_page_key(job_id: str, period: str) -> str:
    return f"{ANSWER_PAGE_PREFIX}{period}/{job_id}.md"


def hub_key(link: str) -> str:
    return f"{HUB_PREFIX}{link}.md"


def page_link_line(link: str) -> str:
    """The one standing line an indexed page carries, pointing at its own hub of lab questions."""
    return f"- 이 페이지를 근거로 답한 랩 질문: [[lab-questions/by-page/{link}]]"


def _one_line(value: Any, limit: int) -> str:
    text = _WHITESPACE.sub(" ", str(value or "")).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _citation_lines(citations: Sequence[Mapping[str, Any]]) -> list[str]:
    """``[[sources/stem]]`` per cited page, keeping the order and naming the sections once each."""
    sections: dict[str, list[str]] = {}
    for citation in citations or ():
        if not isinstance(citation, Mapping):
            continue
        key = citation.get("key")
        if not isinstance(key, str) or not key.startswith("wiki/") or not key.endswith(".md"):
            continue
        link = key[len("wiki/"):-len(".md")]
        section = _one_line(citation.get("section"), 80)
        named = sections.setdefault(link, [])
        if section and section not in named:
            named.append(section)
    lines = []
    for link, named in sections.items():
        lines.append(f"- [[{link}]]" + (f" — {', '.join(named)}" if named else ""))
    return lines


def render_answer_page(answer: Mapping[str, Any], *, question: str, job_id: str, period: str,
                       now: datetime | None = None) -> str:
    """The Markdown of one answered question. The member id and the conversation stay out of it."""
    title = _one_line(question, MAX_TITLE_CHARS)
    day = now_iso(now)[:10]
    front = [
        "---",
        f"title: {_quote(title)}",
        'category: "lab-questions"',
        'kind: "lab-answer"',
        "indexed: false",
        f"created: {_quote(day)}",
        f"job_id: {_quote(job_id)}",
        f"period: {_quote(period)}",
    ]
    for name in ("evidence_state", "status", "model_id", "policy_revision", "reasoning"):
        value = answer.get(name)
        if isinstance(value, str) and value:
            front.append(f"{name}: {_quote(value)}")
    front.append("---")

    body = [f"# {title}", "", "## 질문", "", _WHITESPACE.sub(" ", str(question or "")).strip(), ""]
    text = str(answer.get("answer") or "").strip()
    body += ["## 답변", "", text or "_이 질문에는 답변 본문이 저장되지 않았습니다._", ""]

    citations = _citation_lines(answer.get("citations") or ())
    body += ["## 근거", ""] + (citations or ["- _인용 없음_"]) + [""]

    limitations = [f"- {_one_line(item, 400)}" for item in (answer.get("limitations") or ())
                   if _one_line(item, 400)]
    if limitations:
        body += ["## 한계", ""] + limitations + [""]

    unresolved = [f"- {_one_line(item, 400)}" for item in (answer.get("unresolved_items") or ())
                  if _one_line(item, 400)]
    if unresolved:
        body += ["## 남은 질문", ""] + unresolved + [""]

    body += ["---", "", f"실행 기록: `runs/lab-questions/{job_id}/`", ""]
    return "\n".join(front + [""] + body)


def _render_hub(link: str, entries: Sequence[str]) -> str:
    return "\n".join([
        "---",
        f"title: {_quote(link + ' 을 근거로 답한 랩 질문')}",
        'category: "lab-questions"',
        'kind: "page-question-hub"',
        "indexed: false",
        f"page: {_quote('wiki/' + link + '.md')}",
        "---",
        "",
        f"# [[{link}]] 을 근거로 답한 질문",
        "",
        "이 목록은 검색 색인에 들어가지 않습니다. 색인된 페이지에서 질문으로 내려오는 길입니다.",
        "",
        *entries,
        "",
    ])


def _hub_entry(link: str, title: str, day: str) -> str:
    return f"- {day} [[{link}|{title}]]"


def _append_to_hub(writer: PageWriter, link: str, entry: str) -> dict[str, Any]:
    """Put ``entry`` at the top of the page's hub, creating the hub the first time."""
    key = hub_key(link)
    for _attempt in range(HUB_ATTEMPTS):
        current = writer.get_markdown(key)
        if current is None:
            try:
                writer.put_markdown(key, _render_hub(link, [entry]), create_only=True)
                return {"key": key, "outcome": "created"}
            except ConditionFailed:
                continue                      # another answer created it first; read it and append
        else:
            text, etag = current
            if entry in text:
                return {"key": key, "outcome": "unchanged"}
            lines = text.splitlines()
            marker = next((i for i, line in enumerate(lines) if line.startswith("- ")), len(lines))
            lines.insert(marker, entry)
            try:
                writer.put_markdown(key, "\n".join(lines) + "\n", expected_etag=etag)
                return {"key": key, "outcome": "appended"}
            except ConditionFailed:
                continue                      # the hub moved under us; read it again
    return {"key": key, "outcome": "conflict"}


def publish_answer(writer: PageWriter, answer: Mapping[str, Any], *, question: str, job_id: str,
                   period: str, now: datetime | None = None) -> dict[str, Any]:
    """Write the question's page and link it from the hub of every source note it cited.

    Returns ``{"page", "hubs", "errors"}``. Every failure is reported, never raised: a member's
    answer is already delivered by the time this runs and must not be undone by a page.
    """
    report: dict[str, Any] = {"page": None, "hubs": [], "errors": []}
    key = answer_page_key(job_id, period)
    try:
        writer.put_markdown(key, render_answer_page(answer, question=question, job_id=job_id,
                                                    period=period, now=now))
        report["page"] = key
    except (ConditionFailed, ValueError, Exception) as exc:  # noqa: BLE001 - reported, never raised
        report["errors"].append({"key": key, "error": type(exc).__name__, "message": str(exc)[:300]})
        return report

    entry = _hub_entry(key[len("wiki/"):-len(".md")], _one_line(question, MAX_ENTRY_CHARS),
                       now_iso(now)[:10])
    cited = dict.fromkeys(filter(None, (link_of(c.get("key")) for c in (answer.get("citations") or ())
                                        if isinstance(c, Mapping))))
    for link in cited:
        try:
            report["hubs"].append(_append_to_hub(writer, link, entry))
        except Exception as exc:  # noqa: BLE001 - one unreachable hub never loses the others
            report["errors"].append({"key": hub_key(link), "error": type(exc).__name__,
                                     "message": str(exc)[:300]})
    return report
