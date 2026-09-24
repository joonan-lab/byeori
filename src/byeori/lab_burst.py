"""Concurrent-session validation of the lab Function URL (docs/LAB-QUESTION-WORKFLOW.md, P5).

The design requires, before the beta opens, that 25 independent sessions submit, poll and search
at once so cold/warm latency, p50/p95, queue wait, throttling and cumulative cost contention are
measured rather than inferred from a few warm serial calls. ``run_burst`` drives N sessions in
parallel threads. Each session builds its own ``LabClient`` through ``client_factory`` (the same
signed client the student MCP server uses), runs a few ``search_wiki`` calls, then submits one
``ask_byeori`` with a fresh ``request_id`` and polls ``get_byeori_answer`` until the job leaves
``received``/``queued``/``running`` or ``max_wait`` passes. Every call is recorded as metadata
only: action, outcome, error code, elapsed seconds, and for answers the status, ``usd_micros``,
usage tokens, seconds to answer and triage state. No answer text, no question text beyond an
80-character preview, no ARN and no credential ever enters the result or the receipt.

Cost: a full run with ``ask`` enabled produces about ``sessions`` answers. With the current answer
model each answer is roughly $0.20-0.25 (application estimate from the repository price table,
not an invoice) plus a few cents of Jev triage, so 25 sessions cost about $5-6. The gateway also
admits at most 20 ``ask_byeori`` intakes per member per minute, so 25 asks under one IAM identity
yield ``rate_limited`` outcomes by design; those are recorded, not retried. Run this only after
the operator has confirmed that no other heavy job (campaign, synthesis, index rebuild) shares
the account, and record the receipt under ``state/``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

__all__ = [
    "ACTIVE_STATUSES", "DEFAULT_QUESTIONS", "QUESTION_PREVIEW_CHARS", "SEARCH_QUERIES", "build_parser",
    "default_client_factory", "default_receipt_path", "latency_summary", "load_questions", "main", "percentile",
    "run_burst", "summarise", "write_receipt",
]

ACTIVE_STATUSES = frozenset({"received", "queued", "running"})
TERMINAL_STATUSES = frozenset({"completed", "partial", "failed", "outcome_unknown", "rejected_budget"})
# Poll failures that cannot change on a later poll; anything else is recorded and polling continues.
PERMANENT_POLL_ERRORS = frozenset({"not_found", "forbidden", "unauthenticated", "wrong_account", "inactive_member",
                                   "invalid_request", "unknown_action", "method_not_allowed"})
QUESTION_PREVIEW_CHARS = 80
SEARCH_LIMIT = 10
BARRIER_TIMEOUT_SECONDS = 30.0
MAX_SESSIONS = 100
MAX_SEARCHES_PER_SESSION = 10
DEFAULT_SESSIONS = 25
DEFAULT_SEARCHES = 2
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_MAX_WAIT = 600.0
COST_NOTE = ("usd_micros are application estimates from the repository price table for the answer model; "
             "Jev triage is estimated separately; neither is an AWS invoice.")
# Keys that would carry text, identity or secrets; a receipt holding one of them is refused.
FORBIDDEN_RECEIPT_KEYS = frozenset({"answer", "question", "citations", "limitations", "context", "message",
                                    "Authorization", "credentials", "secret_key", "access_key", "token",
                                    "member_id", "principal_arn"})

# Fixed English research queries; sessions draw them round-robin so warm repeats and distinct
# first hits both appear in one run.
SEARCH_QUERIES = (
    "CHD8 chromatin targets in autism",
    "SCN2A loss of function autism epilepsy",
    "de novo variant burden exome sequencing autism",
    "16p11.2 deletion cognitive phenotype",
    "polygenic risk score autism common variants",
    "female protective effect rare variants",
    "SHANK3 Phelan-McDermid syndrome",
    "prenatal cortical expression of autism risk genes",
    "sibling recurrence risk autism",
    "parental age de novo mutation rate",
)

# Twenty-five short, mutually distinct English questions about autism genetics; the default ask
# set when the operator gives no questions file.
DEFAULT_QUESTIONS = (
    "Which genes reach exome-wide significance for autism in the largest de novo variant studies?",
    "How much autism liability is attributable to common variants compared with rare de novo mutations?",
    "What phenotypes accompany CHD8 loss-of-function variants beyond autism?",
    "Does SCN2A gain versus loss of function separate epilepsy from autism presentations?",
    "What is the sibling recurrence risk of autism when the proband carries a de novo variant?",
    "How does parental age relate to the de novo mutation rate in autism cohorts?",
    "Which cell types in the developing cortex are enriched for autism risk gene expression?",
    "Do autism risk genes converge on chromatin regulation, synaptic function, or both?",
    "What does the 16p11.2 deletion contribute to autism risk and cognitive outcomes?",
    "How does a female protective effect appear in rare variant burden analyses?",
    "What is known about SHANK3 variants and outcomes in Phelan-McDermid syndrome?",
    "Do noncoding de novo variants contribute measurably to autism risk?",
    "Which autism GWAS loci have replicated across independent cohorts?",
    "How does polygenic risk for autism correlate with educational attainment?",
    "What is the evidence that ADNP variants define a recognisable syndrome?",
    "How does mosaicism in parental gametes change autism recurrence estimates?",
    "Which autism risk genes also carry evidence for schizophrenia?",
    "What fraction of autism cases receive a genetic diagnosis from exome sequencing?",
    "How do copy number variants and single nucleotide variants compare in their autism risk contribution?",
    "What does DYRK1A haploinsufficiency cause in humans and in model systems?",
    "Are autism risk genes expressed preferentially in prenatal or postnatal brain?",
    "How do rare inherited variants from unaffected parents contribute to autism?",
    "What is the role of FOXP1 variants in autism with language impairment?",
    "Do autism and ADHD share rare variant burden in the same constrained genes?",
    "Which findings link mitochondrial or metabolic genes to autism?",
)


# ---------------------------------------------------------------------------------------------
# Statistics without numpy
# ---------------------------------------------------------------------------------------------

def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile, as ``scripts/benchmark_retrieval_rpc.py`` reports it; None when empty."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * fraction))
    return ordered[min(rank, len(ordered)) - 1]


def latency_summary(values: list[float]) -> dict[str, Any]:
    return {"count": len(values), "p50": percentile(values, 0.5), "p95": percentile(values, 0.95),
            "max": max(values) if values else None}


# ---------------------------------------------------------------------------------------------
# The burst
# ---------------------------------------------------------------------------------------------

class _FirstCall:
    """Hands the ``cold`` label to exactly one call across all sessions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._taken = False

    def take(self) -> bool:
        with self._lock:
            if self._taken:
                return False
            self._taken = True
            return True


def _error_code(exc: BaseException) -> str:
    code = getattr(exc, "code", None)
    return code if isinstance(code, str) and code else f"exception:{type(exc).__name__}"


def _http_status(code: str | None) -> int | None:
    if isinstance(code, str) and code.startswith("http_") and code[5:].isdigit():
        return int(code[5:])
    return None


def _preview(text: str) -> str:
    return text[:QUESTION_PREVIEW_CHARS]


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class _Session:
    """One independent session: its own client, its calls and its single question job."""

    def __init__(self, index: int, client_factory: Callable[[], Any], first: _FirstCall,
                 clock: Callable[[], float], sleep: Callable[[float], None]) -> None:
        self.index = index
        self.client_factory = client_factory
        self.first = first
        self.clock = clock
        self.sleep = sleep
        self.calls: list[dict[str, Any]] = []
        self.job: dict[str, Any] | None = None
        self.client: Any = None

    def call(self, action: str, body: Mapping[str, Any], *, request_id: str | None = None,
             position: str = "later") -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Send one action; the record is metadata only and the envelope is returned for control flow."""
        phase = "cold" if self.first.take() else "warm"
        record: dict[str, Any] = {"session": self.index, "action": action, "phase": phase, "position": position,
                                  "ok": False, "error": None, "http_status": None, "seconds": None}
        started = self.clock()
        envelope: dict[str, Any] | None = None
        try:
            if request_id is None:
                envelope = self.client.call(action, dict(body))
            else:
                envelope = self.client.call(action, dict(body), request_id=request_id)
        except Exception as exc:  # noqa: BLE001 - every failure is a measurement here
            record["error"] = _error_code(exc)
        record["seconds"] = self.clock() - started
        if envelope is not None:
            if not isinstance(envelope, Mapping):
                record["error"] = "malformed_envelope"
                envelope = None
            elif envelope.get("ok") is True:
                record["ok"] = True
            else:
                code = envelope.get("error")
                record["error"] = code if isinstance(code, str) and code else "unknown_error"
                record["http_status"] = _http_status(record["error"])
        self.calls.append(record)
        return record, envelope

    def run(self, searches: int, question: str | None, poll_interval: float, max_wait: float) -> None:
        for number in range(searches):
            query = SEARCH_QUERIES[(self.index * searches + number) % len(SEARCH_QUERIES)]
            record, envelope = self.call("search_wiki", {"query": query, "limit": SEARCH_LIMIT},
                                         position="first" if number == 0 else "later")
            record["query_index"] = SEARCH_QUERIES.index(query)
            if envelope is not None and isinstance(envelope.get("results"), list):
                record["result_count"] = len(envelope["results"])
        if question is not None:
            self.ask_and_wait(question, poll_interval, max_wait)

    def ask_and_wait(self, question: str, poll_interval: float, max_wait: float) -> None:
        # Same shape as lab_mcp_server.new_request_id(): 32 hex characters generated before the send.
        request_id = uuid.uuid4().hex
        job: dict[str, Any] = {"session": self.index, "question_preview": _preview(question), "request_id": request_id,
                               "job_id": None, "outcome": None, "status": None, "delivery": None, "polls": 0,
                               "poll_errors": 0, "queue_wait_seconds": None, "seconds_to_answer": None,
                               "server_seconds_to_answer": None, "usd_micros": None, "usage": None,
                               "triage_status": None, "evidence_state": None, "error_code": None}
        self.job = job
        ask_started = self.clock()
        record, envelope = self.call("ask_byeori", {"question": question}, request_id=request_id)
        if envelope is None or not record["ok"]:
            job["outcome"], job["error_code"] = "ask_failed", record["error"]
            return
        job["job_id"] = envelope.get("job_id")
        job["status"] = envelope.get("status")
        job["delivery"] = envelope.get("delivery")
        if job["status"] == "running":
            job["queue_wait_seconds"] = record["seconds"]
        if job["status"] not in ACTIVE_STATUSES:
            self._settle(job, envelope, ask_started)
            return
        if not isinstance(job["job_id"], str) or not job["job_id"]:
            job["outcome"], job["error_code"] = "ask_failed", "missing_job_id"
            return
        while True:
            if self.clock() - ask_started >= max_wait:
                job["outcome"] = "timed_out_waiting"
                return
            self.sleep(poll_interval)
            record, envelope = self.call("get_byeori_answer", {"job_id": job["job_id"]})
            job["polls"] += 1
            if envelope is None or not record["ok"]:
                job["poll_errors"] += 1
                if record["error"] in PERMANENT_POLL_ERRORS:
                    job["outcome"], job["error_code"] = "poll_failed", record["error"]
                    return
                continue
            status = envelope.get("status")
            job["status"] = status
            if status == "running" and job["queue_wait_seconds"] is None:
                job["queue_wait_seconds"] = self.clock() - ask_started
            if status not in ACTIVE_STATUSES:
                self._settle(job, envelope, ask_started)
                return

    def _settle(self, job: dict[str, Any], view: Mapping[str, Any], ask_started: float) -> None:
        status = view.get("status")
        job["status"] = status
        job["outcome"] = status if status in TERMINAL_STATUSES else "other"
        job["seconds_to_answer"] = self.clock() - ask_started
        created, completed = _parse_iso(view.get("created_at")), _parse_iso(view.get("completed_at"))
        if created is not None and completed is not None:
            job["server_seconds_to_answer"] = (completed - created).total_seconds()
        micros = view.get("usd_micros")
        job["usd_micros"] = int(micros) if isinstance(micros, (int, float)) and not isinstance(micros, bool) else None
        usage = view.get("usage")
        if isinstance(usage, Mapping):
            job["usage"] = {str(key): int(value) for key, value in usage.items()
                            if isinstance(value, (int, float)) and not isinstance(value, bool)}
        job["triage_status"] = view.get("triage_status") if isinstance(view.get("triage_status"), str) else None
        job["evidence_state"] = view.get("evidence_state") if isinstance(view.get("evidence_state"), str) else None
        code = view.get("error_code") or view.get("reason")
        job["error_code"] = code if isinstance(code, str) else None


def _run_session(session: _Session, barrier: threading.Barrier, searches: int, question: str | None,
                 poll_interval: float, max_wait: float) -> _Session:
    factory_error: str | None = None
    try:
        session.client = session.client_factory()
    except Exception as exc:  # noqa: BLE001 - the session is recorded as failed, the others proceed
        factory_error = _error_code(exc)
    try:
        barrier.wait(timeout=BARRIER_TIMEOUT_SECONDS)
    except threading.BrokenBarrierError:
        pass
    if factory_error is not None:
        session.calls.append({"session": session.index, "action": "client_factory", "phase": "warm",
                              "position": "first", "ok": False, "error": factory_error, "http_status": None,
                              "seconds": 0.0})
        return session
    try:
        session.run(searches, question, poll_interval, max_wait)
    except Exception as exc:  # noqa: BLE001 - a crashed thread must not hide the other 24 sessions
        session.calls.append({"session": session.index, "action": "session", "phase": "warm", "position": "later",
                              "ok": False, "error": _error_code(exc), "http_status": None, "seconds": None})
    return session


def run_burst(client_factory: Callable[[], Any], questions: list[str] | tuple[str, ...], *, sessions: int = DEFAULT_SESSIONS,
              searches_per_session: int = DEFAULT_SEARCHES, ask: bool = True, poll_interval: float = DEFAULT_POLL_INTERVAL,
              max_wait: float = DEFAULT_MAX_WAIT, clock: Callable[[], float] = time.monotonic,
              sleep: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Run ``sessions`` independent sessions at once and aggregate their metadata.

    ``client_factory`` is called once per session, inside that session's thread, and must return an
    object with ``call(action, body, *, request_id=None)`` returning the gateway envelope. A
    ``threading.Barrier`` releases every session together after its client exists. Questions are
    drawn round-robin from ``questions``; only their first 80 characters are kept.
    """
    if not isinstance(sessions, int) or not 1 <= sessions <= MAX_SESSIONS:
        raise ValueError(f"sessions must be between 1 and {MAX_SESSIONS}")
    if not isinstance(searches_per_session, int) or not 0 <= searches_per_session <= MAX_SEARCHES_PER_SESSION:
        raise ValueError(f"searches_per_session must be between 0 and {MAX_SEARCHES_PER_SESSION}")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    if max_wait < 0:
        raise ValueError("max_wait must be zero or positive")
    questions = [q for q in questions if isinstance(q, str) and q.strip()]
    if ask and not questions:
        raise ValueError("ask requires at least one non-empty question")
    if not ask and searches_per_session == 0:
        raise ValueError("nothing to run: ask is off and searches_per_session is 0")

    started_wall = datetime.now(timezone.utc)
    started = clock()
    first = _FirstCall()
    barrier = threading.Barrier(sessions)
    plan = [_Session(index, client_factory, first, clock, sleep) for index in range(sessions)]
    with ThreadPoolExecutor(max_workers=sessions) as pool:
        futures = [pool.submit(_run_session, session, barrier, searches_per_session,
                               questions[session.index % len(questions)] if ask else None, poll_interval, max_wait)
                   for session in plan]
        finished_sessions = [future.result() for future in futures]
    elapsed = clock() - started
    finished_wall = datetime.now(timezone.utc)

    calls = [record for session in finished_sessions for record in session.calls]
    jobs = [session.job for session in finished_sessions if session.job is not None]
    return {
        "kind": "lab_burst",
        "started_at": started_wall.isoformat(),
        "finished_at": finished_wall.isoformat(),
        "elapsed_seconds": elapsed,
        "sessions": sessions,
        "searches_per_session": searches_per_session,
        "ask": ask,
        "poll_interval": poll_interval,
        "max_wait": max_wait,
        "call_count": len(calls),
        "latency": _latency_by_action(calls),
        "latency_split": _latency_split(calls),
        "errors": _errors_by_code(calls),
        "errors_by_action": _errors_by_action(calls),
        "session_failures": sum(1 for record in calls if record["action"] in ("client_factory", "session")),
        "answers": _answer_counts(jobs, sessions if ask else 0),
        "total_usd_micros": sum(job["usd_micros"] for job in jobs if isinstance(job["usd_micros"], int)),
        "usage_tokens": _usage_totals(jobs),
        "queue_wait_seconds": latency_summary([job["queue_wait_seconds"] for job in jobs
                                               if isinstance(job["queue_wait_seconds"], (int, float))]),
        "queue_wait_unobserved": sum(1 for job in jobs if job["outcome"] in TERMINAL_STATUSES
                                     and job["queue_wait_seconds"] is None),
        "seconds_to_answer": latency_summary([job["seconds_to_answer"] for job in jobs
                                              if isinstance(job["seconds_to_answer"], (int, float))]),
        "server_seconds_to_answer": latency_summary([job["server_seconds_to_answer"] for job in jobs
                                                     if isinstance(job["server_seconds_to_answer"], (int, float))]),
        "triage": dict(Counter(job["triage_status"] or "unknown" for job in jobs if job["outcome"] in TERMINAL_STATUSES)),
        "question_jobs": [{key: value for key, value in job.items() if key != "request_id"} for job in jobs],
        "calls": calls,
        "cost_note": COST_NOTE,
    }


def _latency_by_action(calls: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_action: dict[str, list[float]] = {}
    for record in calls:
        if isinstance(record.get("seconds"), (int, float)):
            by_action.setdefault(record["action"], []).append(float(record["seconds"]))
    return {action: latency_summary(values) for action, values in sorted(by_action.items())}


def _latency_split(calls: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    timed = [record for record in calls if isinstance(record.get("seconds"), (int, float))]
    return {
        "cold": latency_summary([r["seconds"] for r in timed if r["phase"] == "cold"]),
        "warm": latency_summary([r["seconds"] for r in timed if r["phase"] == "warm"]),
        "first_search": latency_summary([r["seconds"] for r in timed
                                         if r["action"] == "search_wiki" and r["position"] == "first"]),
        "later_search": latency_summary([r["seconds"] for r in timed
                                         if r["action"] == "search_wiki" and r["position"] == "later"]),
    }


def _errors_by_code(calls: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record["error"] for record in calls if record.get("error")).items()))


def _errors_by_action(calls: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(record["action"] for record in calls if record.get("error")).items()))


def _answer_counts(jobs: list[dict[str, Any]], submitted: int) -> dict[str, int]:
    counts = {"submitted": submitted, "completed": 0, "partial": 0, "failed": 0, "outcome_unknown": 0,
              "rejected_budget": 0, "timed_out_waiting": 0, "ask_failed": 0, "poll_failed": 0, "other": 0}
    for job in jobs:
        outcome = job.get("outcome") or "other"
        counts[outcome if outcome in counts else "other"] += 1
    return counts


def _usage_totals(jobs: list[dict[str, Any]]) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for job in jobs:
        if isinstance(job.get("usage"), Mapping):
            totals.update({key: int(value) for key, value in job["usage"].items()})
    return dict(sorted(totals.items()))


# ---------------------------------------------------------------------------------------------
# Receipt and summary
# ---------------------------------------------------------------------------------------------

def _check_metadata_only(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_RECEIPT_KEYS:
                raise ValueError(f"receipt would carry {key!r} at {path}; the burst receipt is metadata only")
            _check_metadata_only(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for number, item in enumerate(value):
            _check_metadata_only(item, f"{path}[{number}]")
    elif isinstance(value, str):
        if value.startswith("arn:") or value.startswith("AKIA") or value.startswith("ASIA") or "Signature=" in value:
            raise ValueError(f"receipt would carry an ARN or credential at {path}")


def write_receipt(result: Mapping[str, Any], path: str | Path) -> Path:
    """Write the metadata-only receipt as JSON; refuses text, identity and credential fields."""
    _check_metadata_only(result)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return target


def default_receipt_path(moment: datetime | None = None, state_dir: str | Path = "state") -> Path:
    stamp = (moment or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return Path(state_dir) / f"lab-burst-{stamp}.json"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def summarise(result: Mapping[str, Any]) -> str:
    """A short terminal table: per-action latency, cold/warm split, answers, errors and cost."""
    lines = [
        f"lab burst: {result['sessions']} sessions, {result['searches_per_session']} searches each, "
        f"ask={'on' if result['ask'] else 'off'}, {result['started_at']} -> {result['finished_at']} "
        f"({_fmt(float(result['elapsed_seconds']))} s)",
        f"{'action':<20}{'count':>7}{'p50':>9}{'p95':>9}{'max':>9}{'errors':>8}",
    ]
    errors_by_action = result.get("errors_by_action", {})
    for action, summary in result["latency"].items():
        lines.append(f"{action:<20}{summary['count']:>7}{_fmt(summary['p50']):>9}{_fmt(summary['p95']):>9}"
                     f"{_fmt(summary['max']):>9}{errors_by_action.get(action, 0):>8}")
    split = result["latency_split"]
    lines.append(f"cold (first call of the run): {_fmt(split['cold']['max'])} s; warm p50/p95: "
                 f"{_fmt(split['warm']['p50'])}/{_fmt(split['warm']['p95'])} s")
    lines.append(f"first search per session p95: {_fmt(split['first_search']['p95'])} s; later searches p95: "
                 f"{_fmt(split['later_search']['p95'])} s")
    answers = result["answers"]
    lines.append("answers: " + ", ".join(f"{key} {value}" for key, value in answers.items()))
    lines.append(f"queue wait (ask -> first running) p50/p95: {_fmt(result['queue_wait_seconds']['p50'])}/"
                 f"{_fmt(result['queue_wait_seconds']['p95'])} s ({result.get('queue_wait_unobserved', 0)} unobserved); "
                 f"seconds to answer p50/p95: {_fmt(result['seconds_to_answer']['p50'])}/"
                 f"{_fmt(result['seconds_to_answer']['p95'])} s")
    if result.get("triage"):
        lines.append("triage: " + ", ".join(f"{key} {value}" for key, value in sorted(result["triage"].items())))
    errors = result["errors"]
    lines.append("errors by code: " + (", ".join(f"{code}={count}" for code, count in errors.items()) if errors else "none"))
    micros = int(result["total_usd_micros"])
    lines.append(f"total usd_micros: {micros:,} (~${micros / 1_000_000:.2f}; application estimate, not an invoice)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# CLI wiring (scripts/lab_burst.py calls main)
# ---------------------------------------------------------------------------------------------

def load_questions(path: str | Path) -> list[str]:
    """One question per line; blank lines and ``#`` comments are skipped."""
    questions = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    questions = [line for line in questions if line and not line.startswith("#")]
    if not questions:
        raise ValueError(f"{path} holds no questions")
    return questions


def default_client_factory(environ: Mapping[str, str] | None = None) -> Callable[[], Any]:
    """Per-session ``LabClient`` builder using the student MCP client's configuration and credential chain.

    Each session gets its own transport (connection pool); the credentials are the operator's own
    profile, so all sessions share one IAM identity unless the operator runs several processes
    under different ``AWS_PROFILE`` values.
    """
    from byeori import lab_mcp_server as student

    config = student.LabConfig.from_env(environ)
    credentials = student.load_credentials()
    if credentials is None:
        raise student.LabConfigError("No AWS credentials found; set AWS_PROFILE to the operator's own profile")

    def factory() -> Any:
        return student.LabClient.from_config(config, credentials, student.default_transport())

    return factory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab_burst",
        description="Run N independent lab sessions at once against LAB_FUNCTION_URL and write a metadata-only "
                    "receipt under state/. With ask enabled each session costs one answer (about $0.20-0.25).")
    parser.add_argument("--sessions", type=int, default=DEFAULT_SESSIONS, help=f"independent sessions (default {DEFAULT_SESSIONS})")
    parser.add_argument("--searches", type=int, default=DEFAULT_SEARCHES, help=f"search_wiki calls per session (default {DEFAULT_SEARCHES})")
    parser.add_argument("--no-ask", action="store_true", help="searches only; submit no question")
    parser.add_argument("--max-wait", type=float, default=DEFAULT_MAX_WAIT, help="seconds to wait for each answer (default 600)")
    parser.add_argument("--poll-interval", type=float, default=DEFAULT_POLL_INTERVAL, help="seconds between polls (default 2)")
    parser.add_argument("--questions-file", type=Path, default=None, help="one question per line; default: built-in 25")
    parser.add_argument("--output", type=Path, default=None, help="receipt path (default state/lab-burst-<UTC stamp>.json)")
    return parser


def main(argv: list[str] | None = None, *, environ: Mapping[str, str] | None = None,
         client_factory: Callable[[], Any] | None = None, out: Callable[[str], Any] = print) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.sessions <= MAX_SESSIONS:
        parser.error(f"--sessions must be between 1 and {MAX_SESSIONS}")
    if not 0 <= args.searches <= MAX_SEARCHES_PER_SESSION:
        parser.error(f"--searches must be between 0 and {MAX_SEARCHES_PER_SESSION}")
    environ = os.environ if environ is None else environ
    if not (environ.get("LAB_FUNCTION_URL") or "").strip():
        parser.exit(2, "lab_burst: LAB_FUNCTION_URL is not set; export the deployed GatewayUrl before running\n")
    try:
        questions = load_questions(args.questions_file) if args.questions_file else list(DEFAULT_QUESTIONS)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"lab_burst: {exc}\n")
    if client_factory is None:
        try:
            client_factory = default_client_factory(environ)
        except Exception as exc:  # noqa: BLE001 - configuration refusal, reported and stopped
            parser.exit(2, f"lab_burst: {exc}\n")
    result = run_burst(client_factory, questions, sessions=args.sessions, searches_per_session=args.searches,
                       ask=not args.no_ask, poll_interval=args.poll_interval, max_wait=args.max_wait)
    path = write_receipt(result, args.output or default_receipt_path())
    out(summarise(result))
    out(f"receipt: {path}")
    return 0
