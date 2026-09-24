"""Student MCP client for the lab Function URL (docs/LAB-QUESTION-WORKFLOW.md, section 4).

The server owns identity, evidence selection, model calls and every record. This client signs
one POST per tool call with the student's own AWS credentials (SigV4, service ``lambda``),
generates the ``request_id`` before the first send so a network retry reuses it, and exposes
the six student tools only. Startup refuses admin configuration so a student session can never
reach the campaign's ingest function or table. Run as ``python -m byeori.lab_mcp_server``.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from mcp.server.fastmcp import FastMCP

ADMIN_ENVIRONMENT = ("AWS_KIRO_WIKI_INGEST_FUNCTION", "AWS_KIRO_WIKI_TABLE")
FUNCTION_URL_PATTERN = re.compile(
    r"^https://(?P<host>[a-z0-9]+\.lambda-url\.(?P<region>[a-z]{2}(?:-gov)?-[a-z]+-\d)\.on\.aws)/?$")
SERVICE = "lambda"
MAX_ATTEMPTS = 2
# A Function URL answers a throttled or briefly unavailable Lambda with a bare HTTP status and no gateway
# envelope; those are retried with backoff. The gateway's own refusals (``rate_limited`` and the rest)
# arrive as envelopes and are returned to the caller unchanged.
THROTTLE_STATUSES = frozenset({429, 502, 503, 504})
THROTTLE_BACKOFF_SECONDS = (0.5, 1.0, 2.0)
THROTTLE_JITTER_SECONDS = 0.25
REQUEST_TIMEOUT_SECONDS = 60.0
POLL_INTERVAL_SECONDS = 2
WAIT_MAX_SECONDS = 120
ACTIVE_STATUSES = frozenset({"received", "queued", "running"})
CONTEXT_MAX_ITEMS = 8
CONTEXT_MAX_CHARS = 4000
CONTEXT_ROLES = frozenset({"user", "assistant"})
SEARCH_MAX_LIMIT = 30
READ_MAX_CHARS = 8000
MESSAGE_MAX_CHARS = 300

_sleep = time.sleep


class LabConfigError(Exception):
    """The environment does not describe a student session; the server refuses to start."""


class LabClientError(Exception):
    """The Function URL could not be reached; ``code`` is stable (``network``)."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class Transport(Protocol):
    def post(self, url: str, *, content: bytes, headers: dict[str, str]) -> Any: ...


@dataclass(frozen=True)
class LabConfig:
    """Function URL and signing region taken from the environment, nothing else."""

    function_url: str
    region: str

    @classmethod
    def from_env(cls, environ: Any = None) -> "LabConfig":
        environ = os.environ if environ is None else environ
        configured = [name for name in ADMIN_ENVIRONMENT if environ.get(name)]
        if configured:
            raise LabConfigError(
                f"{', '.join(configured)} is set: this is the administrator configuration. The student "
                "server needs only LAB_FUNCTION_URL and AWS_PROFILE; unset the administrator variables.")
        raw = (environ.get("LAB_FUNCTION_URL") or "").strip()
        match = FUNCTION_URL_PATTERN.match(raw)
        if not match:
            raise LabConfigError(
                "LAB_FUNCTION_URL must be the lab Function URL, https://<id>.lambda-url.<region>.on.aws/, "
                f"not {raw!r}")
        url_region = match.group("region")
        region = environ.get("LAB_REGION") or environ.get("AWS_REGION") or environ.get("AWS_DEFAULT_REGION") or url_region
        if region != url_region:
            raise LabConfigError(
                f"LAB_REGION/AWS_REGION is {region!r} but LAB_FUNCTION_URL is in {url_region!r}; "
                "the signature must use the function's region")
        return cls(function_url=f"https://{match.group('host')}/", region=region)


def load_credentials() -> Any:
    """The student's own credentials from the standard chain (AWS_PROFILE); no secrets are configured here."""
    import botocore.session
    return botocore.session.Session().get_credentials()


def default_transport() -> httpx.Client:
    return httpx.Client(timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=10.0), follow_redirects=False)


@dataclass
class LabClient:
    """Signs and sends one action per call; the transport and credentials are injectable for tests."""

    function_url: str
    region: str
    credentials: Any
    transport: Transport
    max_attempts: int = MAX_ATTEMPTS
    sleep: Callable[[float], None] = time.sleep

    @classmethod
    def from_config(cls, config: LabConfig, credentials: Any, transport: Transport | None = None) -> "LabClient":
        return cls(function_url=config.function_url, region=config.region, credentials=credentials,
                   transport=transport or default_transport())

    def call(self, action: str, body: dict[str, Any], *, request_id: str | None = None) -> dict[str, Any]:
        """POST ``{"action": ..., **body}`` with the same bytes on every attempt.

        A transport failure is retried once (``max_attempts``). A bare throttle status from the
        Function URL (``THROTTLE_STATUSES`` without a gateway envelope) is retried after each of
        ``THROTTLE_BACKOFF_SECONDS`` with jitter, then returned as ``http_<status>``. Retrying is
        safe because ``ask_byeori`` carries its ``request_id`` and every other action is a read.
        """
        payload = {"action": action, **body}
        if request_id is not None:
            payload["request_id"] = request_id
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        failure: Exception | None = None
        failures = throttles = 0
        while True:
            headers = self.sign(data)
            try:
                response = self.transport.post(self.function_url, content=data, headers=headers)
            except (httpx.TransportError, OSError) as exc:
                failure = exc
                failures += 1
                if failures >= self.max_attempts:
                    raise LabClientError("network", f"{self.function_url} unreachable after {self.max_attempts} "
                                                    f"attempts ({type(failure).__name__})") from exc
                continue
            decoded = decode_response(response.status_code, response.content)
            throttled = (response.status_code in THROTTLE_STATUSES
                         and decoded.get("error") == f"http_{response.status_code}")
            if throttled and throttles < len(THROTTLE_BACKOFF_SECONDS):
                self.sleep(THROTTLE_BACKOFF_SECONDS[throttles] + random.uniform(0, THROTTLE_JITTER_SECONDS))
                throttles += 1
                continue
            return decoded

    def sign(self, data: bytes) -> dict[str, str]:
        credentials = self.credentials
        if hasattr(credentials, "get_frozen_credentials"):
            credentials = credentials.get_frozen_credentials()
        host = httpx.URL(self.function_url).host
        request = AWSRequest(method="POST", url=self.function_url, data=data,
                             headers={"Content-Type": "application/json", "Host": host})
        SigV4Auth(credentials, SERVICE, self.region).add_auth(request)
        return {name: value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for name, value in request.headers.items()}


def decode_response(status: int, raw: bytes) -> dict[str, Any]:
    """The gateway envelope as-is; anything else becomes a failure envelope with a bounded message."""
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        parsed = None
    if isinstance(parsed, dict) and "ok" in parsed:
        return parsed
    text = raw.decode("utf-8", errors="replace").strip()
    return {"ok": False, "error": f"http_{status}", "message": text[:MESSAGE_MAX_CHARS]}


_CLIENT: LabClient | None = None


def client() -> LabClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LabClient.from_config(LabConfig.from_env(), _require_credentials())
    return _CLIENT


def _require_credentials() -> Any:
    credentials = load_credentials()
    if credentials is None:
        raise LabConfigError("No AWS credentials found; set AWS_PROFILE to the student's own profile")
    return credentials


def new_request_id() -> str:
    return uuid.uuid4().hex


def _validated_context(context: list[dict[str, str]] | None) -> list[dict[str, str]] | None:
    if context is None:
        return None
    if not isinstance(context, list) or len(context) > CONTEXT_MAX_ITEMS:
        raise ValueError(f"context holds at most {CONTEXT_MAX_ITEMS} prior turns")
    cleaned, total = [], 0
    for item in context:
        if not isinstance(item, dict) or not isinstance(item.get("role"), str) or not isinstance(item.get("text"), str):
            raise ValueError("each context item is {'role': 'user'|'assistant', 'text': str}")
        if item["role"] not in CONTEXT_ROLES:
            raise ValueError("context roles are user or assistant; instructions are not conversation context")
        total += len(item["text"])
        cleaned.append({"role": item["role"], "text": item["text"]})
    if total > CONTEXT_MAX_CHARS:
        raise ValueError(f"context holds at most {CONTEXT_MAX_CHARS} characters in total")
    return cleaned


mcp = FastMCP(
    "Byeori Lab",
    instructions=(
        "Byeori answers research questions from the lab's evidence wiki in AWS. Use ask_byeori to "
        "submit a question; it returns a job_id at once because the answer is produced asynchronously. "
        "Use get_byeori_answer with that job_id to collect the answer, its citations and limitations, "
        "and any synthesis_offer the server attached. Only the student decides on an offer: show its "
        "message, wait for an explicit yes or no, then call respond_to_synthesis_offer with the offer's "
        "offer_id, revision and hash. Use search_wiki, read_wiki_page and wiki_backlinks for lookups. "
        "Every tool works on the shared wiki in read-only fashion; nothing here writes a page. "
        "Routing: whenever paper knowledge, literature evidence or a prior-work check is needed, search "
        "this wiki first (search_wiki with a full English question, then read_wiki_page) before any web "
        "search or external database; when nothing is found, say so instead of searching the web unasked. "
        "Mark questions that carry unpublished manuscripts or experiment data with private_material=true. "
        "read_source returns the stored full text behind a wiki/sources/ note for checking a claim against "
        "the paper itself; request_paper records a paper the wiki lacks for the administrator to add."
    ),
)


@mcp.tool()
def ask_byeori(question: str, session_id: str | None = None, parent_job_id: str | None = None,
               context: list[dict[str, str]] | None = None, private_material: bool = False) -> dict[str, Any]:
    """Submit a research question to Byeori; the answer is prepared asynchronously in AWS.

    Returns job_id, session_id, turn, status (queued or rejected_budget) and poll_after_seconds.
    Pass the server's session_id and the previous job_id as parent_job_id for a follow-up question,
    with at most 8 prior turns ({role: user|assistant, text}) totalling 4,000 characters as context,
    so "and in other populations?" keeps its meaning. The question text is sent verbatim.

    A question or context that contains an unpublished manuscript or lab experiment data must set
    private_material to true: the server still answers it from the wiki, but the question is never
    sent to the external Jev judge and no synthesis offer is made from it. The flag cannot be
    added after submission, so set it on the first call.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(private_material, bool):
        raise ValueError("private_material must be true or false")
    body: dict[str, Any] = {"question": question}
    if session_id:
        body["session_id"] = session_id
    if parent_job_id:
        body["parent_job_id"] = parent_job_id
    cleaned = _validated_context(context)
    if cleaned:
        body["context"] = cleaned
    if private_material:
        body["private_material"] = True
    return client().call("ask_byeori", body, request_id=new_request_id())


@mcp.tool()
def get_byeori_answer(job_id: str, wait_seconds: int = 0) -> dict[str, Any]:
    """Collect the answer for a job_id returned by ask_byeori.

    Returns status, answer, citations, limitations, evidence_state, unresolved_items, usage,
    triage_status and, when the server issued one, synthesis_offer. With wait_seconds above zero
    the call polls every 2 seconds while the job is still queued or running, up to that many
    seconds (at most 120). A completed answer whose triage_status is still triage_pending is
    returned as it is; call again later to see the triage result or an offer.
    """
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a non-empty string")
    remaining_polls = max(0, min(int(wait_seconds), WAIT_MAX_SECONDS)) // POLL_INTERVAL_SECONDS
    result = client().call("get_byeori_answer", {"job_id": job_id})
    while remaining_polls > 0 and result.get("ok") is True and result.get("status") in ACTIVE_STATUSES:
        remaining_polls -= 1
        _sleep(POLL_INTERVAL_SECONDS)
        result = client().call("get_byeori_answer", {"job_id": job_id})
    return result


@mcp.tool()
def respond_to_synthesis_offer(offer_id: str, revision: int, offer_hash: str, decision: str) -> dict[str, Any]:
    """Record the student's decision on a synthesis offer the server attached to an answer.

    Call this only after the student has read the offer message and explicitly said yes or no;
    never decide on their behalf. offer_id, revision and offer_hash come from synthesis_offer in
    get_byeori_answer (offer_hash is its hash field); decision is accept or decline. Accepting
    starts an asynchronous research run under the server's research profile and returns its
    execution_id; declining is recorded. The server refuses another student's offer, a stale
    revision or hash, and an expired offer.
    """
    if decision not in ("accept", "decline"):
        raise ValueError("decision must be accept or decline")
    if not offer_id or not offer_hash:
        raise ValueError("offer_id and offer_hash come from the synthesis_offer of get_byeori_answer")
    body = {"offer_id": offer_id, "revision": int(revision), "hash": offer_hash, "decision": decision}
    return client().call("respond_to_synthesis_offer", body, request_id=new_request_id())


@mcp.tool()
def respond_to_collection_offer(job_id: str, decision: str) -> dict[str, Any]:
    """Record the student's decision on a collection offer attached to an answer.

    An answer the wiki could not stand behind carries collection_offer in get_byeori_answer. Call
    this only after the student has read its message and explicitly said yes or no; never decide
    on their behalf. job_id is the answer's job; decision is accept or decline.

    Accepting records a request and starts nothing: no search runs, no paper is collected and no
    money is spent, because collecting papers is the administrator's to approve. Declining is
    recorded too, and either way the question stays in the administrator's queue, so a subject the
    wiki keeps failing on is visible. Another student's offer is refused as not found.
    """
    if decision not in ("accept", "decline"):
        raise ValueError("decision must be accept or decline")
    if not job_id:
        raise ValueError("job_id comes from the collection_offer of get_byeori_answer")
    return client().call("respond_to_collection_offer", {"job_id": job_id, "decision": decision},
                         request_id=new_request_id())


@mcp.tool()
def search_wiki(query: str, limit: int = 10, doc_type: str | None = None) -> dict[str, Any]:
    """BM25 search over the lab wiki in AWS: evidence notes, concept and overview pages.

    Each hit names the page key, title, doc_type, section and a snippet; read a hit with
    read_wiki_page. limit is at most 30. doc_type narrows to note, concept, overview or paper.
    """
    if not 1 <= int(limit) <= SEARCH_MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {SEARCH_MAX_LIMIT}")
    body: dict[str, Any] = {"query": query, "limit": int(limit)}
    if doc_type:
        body["doc_type"] = doc_type
    return client().call("search_wiki", body)


@mcp.tool()
def read_wiki_page(key: str, section: str | None = None, start: int = 0, max_chars: int = 4000) -> dict[str, Any]:
    """Read a page outline, or at most 8,000 characters of one named section.

    Without section the server returns metadata and section names with an empty text. Give the
    exact section heading to read an excerpt, then continue from next_start while has_more is true.
    """
    if not 0 < int(max_chars) <= READ_MAX_CHARS:
        raise ValueError(f"max_chars must be between 1 and {READ_MAX_CHARS}")
    if int(start) < 0:
        raise ValueError("start must be zero or positive")
    body: dict[str, Any] = {"key": key, "start": int(start), "max_chars": int(max_chars)}
    if section:
        body["section"] = section
    return client().call("read_wiki_page", body)


@mcp.tool()
def wiki_backlinks(key: str) -> dict[str, Any]:
    """Which pages cite the page at this key, from the wiki's link table in AWS."""
    return client().call("wiki_backlinks", {"key": key})


@mcp.tool()
def read_source(key: str, section: str | None = None, start: int = 0, max_chars: int = 4000) -> dict[str, Any]:
    """Read the stored full text of the paper behind a wiki/sources/ note: its outline, or one section.

    Use this to check a claim against the original paper's own text (methods, results, tables as
    extracted) instead of the note's summary. Without section the server returns the paper's section
    names; give one heading to read at most 8,000 characters and continue from next_start while
    has_more is true. The PDF file itself is not returned.
    """
    if not 0 < int(max_chars) <= READ_MAX_CHARS:
        raise ValueError(f"max_chars must be between 1 and {READ_MAX_CHARS}")
    if int(start) < 0:
        raise ValueError("start must be zero or positive")
    body: dict[str, Any] = {"key": key, "start": int(start), "max_chars": int(max_chars)}
    if section:
        body["section"] = section
    return client().call("read_source", body)


@mcp.tool()
def request_paper(reason: str, doi: str | None = None, title: str | None = None) -> dict[str, Any]:
    """Ask the lab to add a paper the wiki does not have yet; give its DOI (preferred) or title and why it is needed.

    The server first checks the wiki: if the DOI is already there it returns those notes and records
    nothing. Otherwise it records the request for the administrator, who ingests the paper under the
    lab's journal and licence rules; the paper is not fetched here. Do not cite a requested paper as
    evidence until its note appears in search_wiki.
    """
    body: dict[str, Any] = {"reason": reason}
    if doi:
        body["doi"] = doi
    if title:
        body["title"] = title
    return client().call("request_paper", body, request_id=new_request_id())


def main() -> None:
    """Run the MCP server, or ``byeori-lab setup [--apply]`` to review and route the member's rule files."""
    global _CLIENT
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        from . import lab_setup as _setup      # the same module name in the byeori-lab package
        raise SystemExit(_setup.main(sys.argv[2:]))
    try:
        _CLIENT = LabClient.from_config(LabConfig.from_env(), _require_credentials())
    except LabConfigError as exc:
        raise SystemExit(f"byeori-lab-mcp: {exc}") from exc
    mcp.run()


if __name__ == "__main__":
    main()
