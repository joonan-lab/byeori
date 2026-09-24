"""The burst runner measures concurrent sessions from scripted envelopes; no AWS, no network."""
from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from byeori import lab_burst

LONG_QUESTION = ("Does the female protective effect in autism appear as a higher rare variant burden in affected "
                 "females across every cohort that has measured it, or only in simplex families?")
assert len(LONG_QUESTION) > lab_burst.QUESTION_PREVIEW_CHARS


# ---------------------------------------------------------------------------------------------
# Fakes: a per-thread clock and a client whose envelopes and latencies are scripted per session
# ---------------------------------------------------------------------------------------------

class FakeClock:
    """Each thread has its own timeline, so concurrent sessions do not disturb one another's timing."""

    def __init__(self) -> None:
        self._local = threading.local()

    def now(self) -> float:
        return getattr(self._local, "value", 0.0)

    def advance(self, seconds: float) -> None:
        self._local.value = self.now() + seconds

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


def ok(action: str, **fields: Any) -> dict[str, Any]:
    return {"ok": True, "action": action, **fields}


def failure(code: str) -> dict[str, Any]:
    return {"ok": False, "error": code, "message": "scripted failure"}


COMPLETED = ok("get_byeori_answer", job_id="job-a", status="completed", usd_micros=200_000,
               usage={"inputTokens": 20_000, "outputTokens": 3_000}, triage_status="triage_pending",
               evidence_state="sufficient", created_at="2026-09-21T14:00:00+00:00",
               completed_at="2026-09-21T14:00:50+00:00", answer="never copied", citations=[], limitations=[])


class Script:
    """Per-action outcome queues with a default once a queue is exhausted; latency is per action."""

    def __init__(self, outcomes: dict[str, list[Any]], defaults: dict[str, Any] | None = None,
                 latency: dict[str, list[float] | float] | None = None) -> None:
        self.outcomes = {action: list(items) for action, items in outcomes.items()}
        self.defaults = defaults or {}
        self.latency = latency or {}

    def next(self, action: str) -> tuple[Any, float]:
        queue = self.outcomes.get(action, [])
        outcome = queue.pop(0) if queue else self.defaults.get(action, failure("unscripted_action"))
        latency = self.latency.get(action, 0.1)
        if isinstance(latency, list):
            seconds = latency.pop(0) if len(latency) > 1 else latency[0]
        else:
            seconds = latency
        return outcome, seconds


class FakeLabClient:
    def __init__(self, clock: FakeClock, script: Script) -> None:
        self.clock = clock
        self.script = script
        self.sent: list[tuple[str, dict[str, Any], str | None]] = []

    def call(self, action: str, body: dict[str, Any], *, request_id: str | None = None) -> dict[str, Any]:
        self.sent.append((action, dict(body), request_id))
        outcome, seconds = self.script.next(action)
        self.clock.advance(seconds)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Factory:
    """Hands one scripted client to each session (thread-safe) and keeps them for inspection."""

    def __init__(self, clock: FakeClock, scripts: list[Script]) -> None:
        self.clock = clock
        self.scripts = list(scripts)
        self.clients: list[FakeLabClient] = []
        self._lock = threading.Lock()

    def __call__(self) -> FakeLabClient:
        with self._lock:
            if not self.scripts:
                raise AssertionError("more sessions than scripts")
            client = FakeLabClient(self.clock, self.scripts.pop(0))
            self.clients.append(client)
            return client


class NetworkDown(Exception):
    code = "network"


SEARCH_LATENCY = [0.5, 0.2]  # first search of a session, then later ones


def completed_script() -> Script:
    return Script({"search_wiki": [], "ask_byeori": [ok("ask_byeori", job_id="job-a", status="queued", delivery="sent")],
                   "get_byeori_answer": [ok("get_byeori_answer", job_id="job-a", status="queued"),
                                         ok("get_byeori_answer", job_id="job-a", status="running"), COMPLETED]},
                  defaults={"search_wiki": ok("search_wiki", results=[{"key": "wiki/sources/a.md"}], index_etag='"e"')},
                  latency={"search_wiki": list(SEARCH_LATENCY), "ask_byeori": 0.3, "get_byeori_answer": 0.1})


def rate_limited_script() -> Script:
    return Script({"ask_byeori": [failure("rate_limited")]},
                  defaults={"search_wiki": ok("search_wiki", results=[], index_etag='"e"')},
                  latency={"search_wiki": list(SEARCH_LATENCY), "ask_byeori": 0.3})


def never_finishes_script() -> Script:
    return Script({"ask_byeori": [ok("ask_byeori", job_id="job-c", status="queued", delivery="pending")]},
                  defaults={"search_wiki": ok("search_wiki", results=[], index_etag='"e"'),
                            "get_byeori_answer": ok("get_byeori_answer", job_id="job-c", status="running")},
                  latency={"search_wiki": list(SEARCH_LATENCY), "ask_byeori": 0.3, "get_byeori_answer": 0.1})


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def run_three(clock: FakeClock, questions=None, **overrides: Any) -> tuple[dict[str, Any], Factory]:
    factory = Factory(clock, [completed_script(), rate_limited_script(), never_finishes_script()])
    options = dict(sessions=3, searches_per_session=2, ask=True, poll_interval=2.0, max_wait=10.0,
                   clock=clock.now, sleep=clock.sleep)
    options.update(overrides)
    result = lab_burst.run_burst(factory, questions or list(lab_burst.DEFAULT_QUESTIONS), **options)
    return result, factory


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------

def test_percentile_is_nearest_rank_without_numpy():
    assert lab_burst.percentile([5, 1, 4, 2, 3], 0.5) == 3
    assert lab_burst.percentile([5, 1, 4, 2, 3], 0.95) == 5
    assert lab_burst.percentile([7.0], 0.95) == 7.0
    assert lab_burst.percentile([], 0.5) is None
    assert lab_burst.latency_summary([]) == {"count": 0, "p50": None, "p95": None, "max": None}
    assert lab_burst.latency_summary([0.2, 0.5, 0.2]) == {"count": 3, "p50": 0.2, "p95": 0.5, "max": 0.5}


# ---------------------------------------------------------------------------------------------
# run_burst with three scripted sessions
# ---------------------------------------------------------------------------------------------

def test_three_sessions_count_calls_and_latency_percentiles(clock):
    result, factory = run_three(clock)
    assert result["sessions"] == 3 and len(factory.clients) == 3, "one client per session"
    search = result["latency"]["search_wiki"]
    assert search == {"count": 6, "p50": pytest.approx(0.2), "p95": pytest.approx(0.5), "max": pytest.approx(0.5)}
    assert result["latency"]["ask_byeori"]["count"] == 3
    assert result["latency"]["ask_byeori"]["p95"] == pytest.approx(0.3)
    polls = result["latency"]["get_byeori_answer"]
    assert polls["count"] == 3 + 5, "three polls to completion plus five before the 10 s wait expires"
    assert polls["p95"] == pytest.approx(0.1)
    assert result["call_count"] == 6 + 3 + 8
    assert result["started_at"].endswith("+00:00") and result["finished_at"] >= result["started_at"]


def test_cold_is_the_first_call_of_the_run_and_first_search_is_per_session(clock):
    result, _ = run_three(clock)
    split = result["latency_split"]
    assert split["cold"]["count"] == 1 and split["cold"]["max"] == pytest.approx(0.5)
    assert split["warm"]["count"] == result["call_count"] - 1
    assert split["first_search"] == {"count": 3, "p50": pytest.approx(0.5), "p95": pytest.approx(0.5), "max": pytest.approx(0.5)}
    assert split["later_search"]["count"] == 3 and split["later_search"]["p95"] == pytest.approx(0.2)
    cold = [record for record in result["calls"] if record["phase"] == "cold"]
    assert len(cold) == 1 and cold[0]["action"] == "search_wiki" and cold[0]["position"] == "first"


def test_answers_cost_queue_wait_and_timeout_are_aggregated(clock):
    result, _ = run_three(clock)
    assert result["answers"] == {"submitted": 3, "completed": 1, "partial": 0, "failed": 0, "outcome_unknown": 0,
                                 "rejected_budget": 0, "timed_out_waiting": 1, "ask_failed": 1, "poll_failed": 0,
                                 "other": 0}
    assert result["total_usd_micros"] == 200_000
    assert result["usage_tokens"] == {"inputTokens": 20_000, "outputTokens": 3_000}
    assert result["errors"] == {"rate_limited": 1}
    assert result["errors_by_action"] == {"ask_byeori": 1}
    assert result["triage"] == {"triage_pending": 1}
    # completed job: ask 0.3 s, sleep 2, poll 0.1 (queued), sleep 2, poll 0.1 (running) -> 4.5 s to the first
    # running poll; the job that never finishes shows running on its first poll at 2.4 s and counts too.
    assert result["queue_wait_seconds"] == {"count": 2, "p50": pytest.approx(2.4), "p95": pytest.approx(4.5),
                                            "max": pytest.approx(4.5)}
    assert result["queue_wait_unobserved"] == 0
    assert result["seconds_to_answer"]["p95"] == pytest.approx(6.6)
    assert result["server_seconds_to_answer"]["max"] == pytest.approx(50.0)
    by_outcome = {job["outcome"]: job for job in result["question_jobs"]}
    assert set(by_outcome) == {"completed", "ask_failed", "timed_out_waiting"}
    assert by_outcome["completed"]["job_id"] == "job-a" and by_outcome["completed"]["delivery"] == "sent"
    assert by_outcome["timed_out_waiting"]["job_id"] == "job-c" and by_outcome["timed_out_waiting"]["polls"] == 5
    assert by_outcome["ask_failed"]["job_id"] is None and by_outcome["ask_failed"]["error_code"] == "rate_limited"
    assert all("request_id" not in job and "member_id" not in job for job in result["question_jobs"])


def test_each_session_sends_its_own_fresh_request_id_and_round_robin_questions(clock):
    questions = ["first question?", "second question?", "third question?"]
    result, factory = run_three(clock, questions=questions)
    asks = [sent for client in factory.clients for sent in client.sent if sent[0] == "ask_byeori"]
    assert len(asks) == 3
    request_ids = {request_id for _, _, request_id in asks}
    assert len(request_ids) == 3 and all(len(rid) == 32 and int(rid, 16) >= 0 for rid in request_ids)
    assert {body["question"] for _, body, _ in asks} == set(questions)
    assert all(set(body) == {"question"} for _, body, _ in asks)
    searches = [sent for client in factory.clients for sent in client.sent if sent[0] == "search_wiki"]
    assert all(body == {"query": body["query"], "limit": 10} for _, body, _ in searches)
    assert all(body["query"] in lab_burst.SEARCH_QUERIES for _, body, _ in searches)
    assert len({body["query"] for _, body, _ in searches}) == 6, "round-robin covers six distinct queries"
    polls = [sent for client in factory.clients for sent in client.sent if sent[0] == "get_byeori_answer"]
    assert all(request_id is None and set(body) == {"job_id"} for _, body, request_id in polls)
    assert {job["question_preview"] for job in result["question_jobs"]} == set(questions)


def test_no_ask_runs_searches_only(clock):
    factory = Factory(clock, [completed_script(), completed_script()])
    result = lab_burst.run_burst(factory, [], sessions=2, searches_per_session=1, ask=False,
                                 clock=clock.now, sleep=clock.sleep)
    assert result["call_count"] == 2 and set(result["latency"]) == {"search_wiki"}
    assert result["answers"]["submitted"] == 0 and result["question_jobs"] == []
    assert result["total_usd_micros"] == 0


# ---------------------------------------------------------------------------------------------
# Error envelopes and exceptions are counted by code
# ---------------------------------------------------------------------------------------------

def test_error_envelopes_and_client_exceptions_are_counted_by_code(clock):
    scripts = [
        Script({"search_wiki": [failure("http_502"), failure("conflict")],
                "ask_byeori": [failure("internal")]}),
        Script({"search_wiki": [NetworkDown("unreachable"), ok("search_wiki", results=[], index_etag='"e"')],
                "ask_byeori": [ok("ask_byeori", job_id="job-x", status="rejected_budget", reason="budget_exceeded")]}),
        Script({"search_wiki": [failure("rate_limited"), failure("rate_limited")],
                "ask_byeori": [ok("ask_byeori", job_id="job-y", status="queued")],
                "get_byeori_answer": [failure("internal"), failure("not_found")]}),
    ]
    factory = Factory(clock, scripts)
    result = lab_burst.run_burst(factory, ["q?"], sessions=3, searches_per_session=2, ask=True,
                                 poll_interval=2.0, max_wait=60.0, clock=clock.now, sleep=clock.sleep)
    assert result["errors"] == {"conflict": 1, "http_502": 1, "internal": 2, "network": 1, "not_found": 1, "rate_limited": 2}
    assert result["errors_by_action"] == {"ask_byeori": 1, "get_byeori_answer": 2, "search_wiki": 5}
    http = [record for record in result["calls"] if record["error"] == "http_502"]
    assert http[0]["http_status"] == 502 and http[0]["ok"] is False
    assert result["answers"]["ask_failed"] == 1
    assert result["answers"]["rejected_budget"] == 1
    assert result["answers"]["poll_failed"] == 1, "a transient internal error keeps polling; not_found stops it"
    rejected = next(job for job in result["question_jobs"] if job["outcome"] == "rejected_budget")
    assert rejected["error_code"] == "budget_exceeded" and rejected["usd_micros"] is None
    assert result["session_failures"] == 0


def test_a_failing_client_factory_or_crashing_session_does_not_hide_the_others(clock):
    calls = {"count": 0}
    lock = threading.Lock()
    good = Factory(clock, [completed_script()])

    def factory():
        with lock:
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("no credentials for this profile")
        return good()

    result = lab_burst.run_burst(factory, ["q?"], sessions=2, searches_per_session=1, ask=True,
                                 poll_interval=2.0, max_wait=10.0, clock=clock.now, sleep=clock.sleep)
    assert result["session_failures"] == 1
    assert result["errors"]["exception:RuntimeError"] == 1
    assert result["answers"]["completed"] == 1 and result["answers"]["submitted"] == 2


@pytest.mark.parametrize("kwargs", [dict(sessions=0), dict(sessions=101), dict(searches_per_session=-1),
                                    dict(poll_interval=0), dict(max_wait=-1), dict(ask=False, searches_per_session=0)])
def test_invalid_options_are_refused_before_any_call(clock, kwargs):
    factory = Factory(clock, [completed_script()])
    options = dict(sessions=1, searches_per_session=1, ask=True, clock=clock.now, sleep=clock.sleep)
    options.update(kwargs)
    with pytest.raises(ValueError):
        lab_burst.run_burst(factory, ["q?"], **options)
    assert factory.clients == []


def test_ask_without_questions_is_refused(clock):
    with pytest.raises(ValueError):
        lab_burst.run_burst(Factory(clock, [completed_script()]), ["", "   "], sessions=1, ask=True,
                            clock=clock.now, sleep=clock.sleep)


# ---------------------------------------------------------------------------------------------
# Receipt and summary
# ---------------------------------------------------------------------------------------------

def test_write_receipt_keeps_only_an_eighty_character_preview_of_the_question(clock, tmp_path):
    result, _ = run_three(clock, questions=[LONG_QUESTION])
    path = lab_burst.write_receipt(result, tmp_path / "state" / "lab-burst-test.json")
    text = path.read_text(encoding="utf-8")
    assert LONG_QUESTION not in text
    assert "never copied" not in text, "answer text never reaches the receipt"
    loaded = json.loads(text)
    previews = [job["question_preview"] for job in loaded["question_jobs"]]
    assert previews and all(len(preview) <= 80 for preview in previews)
    assert all(LONG_QUESTION.startswith(preview) for preview in previews)
    assert "arn:" not in text and "Authorization" not in text
    assert loaded["kind"] == "lab_burst" and loaded["total_usd_micros"] == 200_000
    assert loaded["cost_note"].endswith("neither is an AWS invoice.")


@pytest.mark.parametrize("poison", [
    {"kind": "lab_burst", "answer": "full answer text"},
    {"kind": "lab_burst", "question_jobs": [{"job_id": "j", "question": "full question text"}]},
    {"kind": "lab_burst", "calls": [{"message": "server text"}]},
    {"kind": "lab_burst", "stack": "arn:aws:lambda:ap-northeast-2:123456789012:function:x"},
    {"kind": "lab_burst", "headers": ["AWS4-HMAC-SHA256 Credential=AKIA.../lambda/aws4_request, Signature=abc"]},
])
def test_write_receipt_refuses_text_identity_and_credential_fields(tmp_path, poison):
    target = tmp_path / "poisoned.json"
    with pytest.raises(ValueError):
        lab_burst.write_receipt(poison, target)
    assert not target.exists()


def test_default_receipt_path_uses_the_state_directory_and_a_utc_stamp():
    from datetime import datetime, timezone
    path = lab_burst.default_receipt_path(datetime(2026, 9, 21, 14, 5, 9, tzinfo=timezone.utc))
    assert str(path) == "state/lab-burst-20260921T140509Z.json"


def test_summarise_shows_p95_values_answers_errors_and_cost(clock):
    result, _ = run_three(clock)
    text = lab_burst.summarise(result)
    assert "p95" in text
    assert f"{result['latency']['search_wiki']['p95']:.2f}" in text
    assert f"{result['latency']['get_byeori_answer']['p95']:.2f}" in text
    assert "search_wiki" in text and "ask_byeori" in text and "get_byeori_answer" in text
    assert "completed 1" in text and "timed_out_waiting 1" in text and "ask_failed 1" in text
    assert "rate_limited=1" in text
    assert "200,000" in text and "$0.20" in text and "not an invoice" in text
    assert "cold" in text and "warm" in text
    assert "never copied" not in text


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

def test_parser_parses_the_documented_flags(tmp_path):
    parser = lab_burst.build_parser()
    args = parser.parse_args(["--sessions", "25", "--searches", "2", "--no-ask", "--max-wait", "600",
                              "--questions-file", str(tmp_path / "q.txt")])
    assert args.sessions == 25 and args.searches == 2 and args.no_ask is True
    assert args.max_wait == 600.0 and args.poll_interval == 2.0
    assert args.questions_file == tmp_path / "q.txt" and args.output is None
    defaults = parser.parse_args([])
    assert defaults.sessions == 25 and defaults.searches == 2 and defaults.no_ask is False
    assert defaults.max_wait == 600.0 and defaults.questions_file is None


def test_main_refuses_when_lab_function_url_is_unset(capsys):
    with pytest.raises(SystemExit) as raised:
        lab_burst.main(["--sessions", "1"], environ={}, client_factory=lambda: pytest.fail("no client is built"))
    assert raised.value.code == 2
    assert "LAB_FUNCTION_URL" in capsys.readouterr().err


def test_main_refuses_out_of_range_sessions(capsys):
    with pytest.raises(SystemExit):
        lab_burst.main(["--sessions", "101"], environ={"LAB_FUNCTION_URL": "https://x.lambda-url.ap-northeast-2.on.aws/"},
                       client_factory=lambda: pytest.fail("no client is built"))
    assert "--sessions" in capsys.readouterr().err


def test_main_runs_with_an_injected_factory_and_writes_the_receipt(clock, tmp_path):
    # --no-ask never sleeps, so the real monotonic clock inside main() is harmless here.
    factory = Factory(clock, [completed_script(), completed_script()])
    output = tmp_path / "state" / "lab-burst-cli.json"
    printed: list[str] = []
    code = lab_burst.main(["--sessions", "2", "--searches", "1", "--no-ask", "--output", str(output)],
                          environ={"LAB_FUNCTION_URL": "https://abcdefghij1234567890abcdefghij12.lambda-url.ap-northeast-2.on.aws/"},
                          client_factory=factory, out=printed.append)
    assert code == 0 and output.exists()
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["sessions"] == 2 and loaded["ask"] is False and loaded["call_count"] == 2
    assert printed[-1] == f"receipt: {output}"
    assert "search_wiki" in printed[0]


def test_load_questions_skips_blank_lines_and_comments(tmp_path):
    path = tmp_path / "q.txt"
    path.write_text("# operator questions\n\nWhich genes reach exome-wide significance?\n  \nWhat does CHD8 loss cause?\n")
    assert lab_burst.load_questions(path) == ["Which genes reach exome-wide significance?", "What does CHD8 loss cause?"]
    (tmp_path / "empty.txt").write_text("# nothing\n")
    with pytest.raises(ValueError):
        lab_burst.load_questions(tmp_path / "empty.txt")


def test_default_questions_are_twenty_five_distinct_english_questions():
    questions = lab_burst.DEFAULT_QUESTIONS
    assert len(questions) == 25 and len(set(questions)) == 25
    assert all(question.endswith("?") and question.isascii() and len(question) < 120 for question in questions)
    assert len(lab_burst.SEARCH_QUERIES) == len(set(lab_burst.SEARCH_QUERIES)) >= 8


# ---------------------------------------------------------------------------------------------
# Boundary of this module
# ---------------------------------------------------------------------------------------------

def test_module_imports_no_client_library_or_campaign_module_at_top_level():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(lab_burst))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not imported & {"httpx", "mcp", "boto3", "numpy", "byeori.aws_store", "byeori.question_agent",
                           "byeori.ingest_lambda", "byeori.agent_cache", "byeori.config"}, imported
