"""Ask Byeori the research questions llm-wiki already answered, and compare.

``select_llm_wiki_questions`` picks llm-wiki ``wiki/questions/`` pages related to autism (a link to
an asd-ndd/asd-models page, or autism in the head). ``run_question_benchmark`` asks each question
through the Lambda ``answer_question`` action (the AWS research agent stores Byeori's page under
``wiki/questions/`` in S3) and writes a local report with, per
question: seconds (search, generation), tokens and cost, page length, numeric tokens, wikilinks,
and how many of llm-wiki's cited papers exist in Byeori. llm-wiki's pages carry no timing, so the
speed column compares Byeori's end-to-end time with the local retrieval latency measured
separately. Quality is left to the reader: the report lists both pages side by side.
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import costs
from .aws_store import AwsStore
from .config import Settings

NUMBER = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?:\s?%|×)?(?![\w])")
WIKILINK = re.compile(r"\[\[([^\]|#]+)")


def _frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    fields = {}
    for line in text[4:end].splitlines():
        k, sep, v = line.partition(":")
        if sep and not line.startswith(" "):
            fields[k.strip()] = v.strip().strip('"')
    return fields, text[end + 5:]


def select_llm_wiki_questions(llm_wiki: Path, *, select: str = "autism", limit: int = 0) -> list[dict[str, Any]]:
    out = []
    for path in sorted((llm_wiki / "wiki" / "questions").glob("*.md")):
        if path.name == "index.md":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        fields, body = _frontmatter(text)
        links = WIKILINK.findall(text)
        asd = sum(1 for l in links if l.startswith(("asd-ndd/", "asd-models/")))
        if select == "autism" and not (asd or re.search(r"autis|\basd\b", text[:2500].lower())):
            continue
        title = fields.get("title") or path.stem
        if not title.endswith("?"):
            title = title.rstrip(".") + "?"
        tags = [t.strip() for t in fields.get("tags", "").strip("[]").split(",") if t.strip()]
        papers = [l.split("/")[-1] for l in links if "/" in l and not l.startswith(("overviews/", "questions/", "concepts/"))]
        out.append({"stem": path.stem, "title": title, "tags": tags[:12], "body": body, "papers": papers, "asd_links": asd})
        if limit and len(out) >= limit:
            break
    return out


def _metrics(body: str) -> dict[str, Any]:
    return {"chars": len(body), "numbers": len({m.group(0) for m in NUMBER.finditer(body)}),
            "wikilinks": len(set(WIKILINK.findall(body)))}


def run_question_benchmark(settings: Settings, llm_wiki: Path, *, select: str = "autism", limit: int = 0,
                           concurrency: int = 4, model_id: str | None = None, dry_run: bool = False) -> dict[str, Any]:
    questions = select_llm_wiki_questions(llm_wiki, select=select, limit=limit)
    run: dict[str, Any] = {"started_at": datetime.now(UTC).replace(microsecond=0).isoformat(), "selected": len(questions),
                           "dry_run": dry_run, "rows": [], "errors": []}
    if dry_run:
        run["titles"] = [q["title"] for q in questions]
        return run
    uploaded = {Path(obj["Key"]).stem for obj in AwsStore(settings).wiki_objects()
                if obj["Key"].startswith("wiki/sources/")}
    local = threading.local()

    def store() -> AwsStore:
        st = getattr(local, "store", None)
        if st is None:
            st = AwsStore(settings)
            local.store = st
        return st

    def one(q: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        result = store().answer_question(q["title"], tags=q["tags"], model_id=model_id)
        usage = result.get("usage") or {}
        est = costs.estimate_draft_usd(result.get("model_id", ""), usage)
        costs.record(settings.state_dir, {"step": "answer_question", "work_id": q["stem"], "status": result.get("status"),
                                          "model_id": result.get("model_id"), "seconds": round(time.monotonic() - started, 1),
                                          "lambda_seconds": result.get("seconds"), "input_tokens": usage.get("inputTokens"),
                                          "output_tokens": usage.get("outputTokens"), "estimated_usd": est,
                                          "basis": "Anthropic list prices applied to reported token counts; not reconciled against the AWS bill"})
        remote_metrics = store().wiki_metrics(result["question_key"])["metrics"] if result.get("status") == "answer_ready" else None
        coverage = sum(1 for p in q["papers"] if p in uploaded) / len(q["papers"]) if q["papers"] else None
        return {"stem": q["stem"], "title": q["title"], "status": result.get("status"), "problems": result.get("problems"),
                "skipped_reason": result.get("skipped_reason"),
                "search_seconds": result.get("search_seconds"), "generate_seconds": result.get("generate_seconds"),
                "wall_seconds": round(time.monotonic() - started, 1), "usage": usage, "estimated_usd": est,
                "retrieved": len(result.get("retrieved") or []), "llm_wiki_cited_papers": len(q["papers"]),
                "llm_wiki_cited_papers_in_byeori": coverage,
                "byeori": remote_metrics, "llm_wiki": _metrics(q["body"]),
                "byeori_page": f"s3://{settings.aws_bucket}/{result['question_key']}" if remote_metrics else None,
                "llm_wiki_page": f"wiki/questions/{q['stem']}.md"}

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(one, q): q["stem"] for q in questions}
        for index, future in enumerate(as_completed(futures), 1):
            stem = futures[future]
            try:
                row = future.result()
                run["rows"].append(row)
                print(f"[{index}/{len(questions)}] {stem} {row['status']} {row['wall_seconds']}s ${row['estimated_usd']}", file=sys.stderr, flush=True)
            except Exception as exc:
                run["errors"].append({"stem": stem, "error": str(exc)})
                print(f"[{index}/{len(questions)}] {stem} error: {exc}", file=sys.stderr, flush=True)
    run["finished_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    ok = [r for r in run["rows"] if r["status"] == "answer_ready"]
    skipped = [r for r in run["rows"] if r["status"] == "answer_skipped"]

    def med(values):
        values = sorted(v for v in values if v is not None)
        return values[len(values) // 2] if values else None

    run["summary"] = {
        "answered": len(ok), "skipped": len(skipped), "failed": len(run["rows"]) - len(ok) - len(skipped),
        "skipped_reasons": sorted({r.get("skipped_reason") or "unknown" for r in skipped}), "errors": len(run["errors"]),
        "median_search_seconds": med(r["search_seconds"] for r in ok), "median_generate_seconds": med(r["generate_seconds"] for r in ok),
        "median_wall_seconds": med(r["wall_seconds"] for r in ok), "total_usd": round(sum(r["estimated_usd"] or 0 for r in ok), 2),
        "median_byeori_chars": med(r["byeori"]["chars"] for r in ok), "median_llm_wiki_chars": med(r["llm_wiki"]["chars"] for r in ok),
        "median_byeori_numbers": med(r["byeori"]["numbers"] for r in ok), "median_llm_wiki_numbers": med(r["llm_wiki"]["numbers"] for r in ok),
        "median_byeori_wikilinks": med(r["byeori"]["wikilinks"] for r in ok), "median_llm_wiki_wikilinks": med(r["llm_wiki"]["wikilinks"] for r in ok),
        "median_share_of_llm_wiki_cited_papers_in_byeori": med(r["llm_wiki_cited_papers_in_byeori"] for r in ok),
    }
    report = settings.state_dir / f"question-benchmark-{run['started_at'].replace(':', '')}.json"
    report.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    run["report"] = str(report)
    return run
